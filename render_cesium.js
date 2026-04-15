const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { startServer } = require('./serve_earth');

function loadDotEnv(dotenvPath) {
    if (!fs.existsSync(dotenvPath)) return;
    const lines = fs.readFileSync(dotenvPath, 'utf8').split(/\r?\n/);
    for (const raw of lines) {
        const line = raw.trim();
        if (!line || line.startsWith('#')) continue;
        const eq = line.indexOf('=');
        if (eq <= 0) continue;
        const key = line.slice(0, eq).trim();
        let value = line.slice(eq + 1).trim();
        if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
        }
        if (key && !(key in process.env)) {
            process.env[key] = value;
        }
    }
}

loadDotEnv(path.resolve(__dirname, '.env'));

const args = process.argv.slice(2);
const lon = parseFloat(args[0] || -74.0060);
const lat = parseFloat(args[1] || 40.7128);
const duration = parseFloat(args[2] || 5.0);
const outDir = args[3] || './nyc_zoom_frames';
const renderMode = String(args[4] || 'auto').toLowerCase();

const fps = 30;
const totalFrames = Math.floor(duration * fps);

if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
}

(async () => {
    let browser = null;
    let serverHandle = null;
    let capturedFrames = 0;
    try {
        console.log(`[CesiumRenderer] Starting cinematic zoom for [${lat}, ${lon}]`);
        serverHandle = await startServer({ requestedPort: 0, host: '127.0.0.1' });
        browser = await chromium.launch({
            headless: true,
            args: [
                '--disable-gpu',
                '--disable-gpu-compositing',
                '--use-gl=angle',
                '--use-angle=swiftshader',
                '--disable-webgl-draft-extensions',
                '--disable-webgl-image-chromium',
                '--use-gl=swiftshader',
                '--ignore-gpu-blocklist',
                '--disable-dev-shm-usage' // Extra force for software rendering
            ]
        });
        const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });

        page.on('console', msg => console.log(`[PAGE] ${msg.text()}`));
        let workerWarned = false;
        page.on('pageerror', err => {
            const message = String(err && err.message ? err.message : err);
            if (message.includes("Failed to execute 'importScripts'") && message.includes('blob:null/')) {
                if (!workerWarned) {
                    console.log('[PAGE WARN] Cesium worker import error detected on software rendering; continuing with fallback path.');
                    workerWarned = true;
                }
                return;
            }
            console.log(`[PAGE ERR] ${message}`);
        });

        const htmlUrl = new URL(`http://${serverHandle.host}:${serverHandle.port}/earth_digital_twin.html`);
        if (process.env.CESIUM_ION_TOKEN) {
            htmlUrl.searchParams.set('cesium_ion_token', process.env.CESIUM_ION_TOKEN);
        }
        if (process.env.MAPBOX_TOKEN) {
            htmlUrl.searchParams.set('mapbox_token', process.env.MAPBOX_TOKEN);
        }
        htmlUrl.searchParams.set('render_mode', renderMode);
        htmlUrl.searchParams.set('automation', '1');
        htmlUrl.searchParams.set('hide_ui', '1');
        await page.goto(htmlUrl.toString());

        await page.waitForFunction(() => typeof window.setCamera === 'function' && typeof window.checkIdle === 'function', { timeout: 30000 });

        // Initial wait for tiles and atmosphere to initialize
        console.log(`[CesiumRenderer] Waiting for initial load (up to 30s)...`);
        try {
            await page.waitForFunction('window.checkIdle() === true', { timeout: 30000 });
            console.log(`[CesiumRenderer] Initial load complete.`);
        } catch (e) {
            console.warn(`[CesiumRenderer] Initial load timed out, proceeding anyway...`);
        }

        console.log(`[CesiumRenderer] Rendering ${totalFrames} frames...`);

        for (let i = 0; i < totalFrames; i++) {
            const t = i / Math.max(1, totalFrames - 1); // 0.0 to 1.0

            // Multi-stage GTA Ease:
            // We want a very fast initial drop, then a slow down as we emerge into the 3D city.
            let alt;
            if (t < 0.4) {
                // Stage 1: Fast drop from Space to Cloud level (10km)
                const st = t / 0.4;
                const ease = 1 - Math.pow(1 - st, 3); // cubic ease out
                alt = 20000000 - (20000000 - 10000) * ease;
            } else {
                // Stage 2: Cinematic drift from Cloud to Drone level (150m)
                const st = (t - 0.4) / 0.6;
                const ease = st; // linear drift
                alt = 10000 - (10000 - 150) * ease;
            }

            const heading = 45 + (t * 20); // slight rotation
            const pitch = -90 + (t * 65);   // transition from top-down to oblique

            try {
                await page.evaluate(({ lat, lon, alt, heading, pitch }) => {
                    window.setCamera(lat, lon, alt, heading, pitch, 0);
                }, { lat, lon, alt, heading, pitch });

                // Wait for frame to be stable and tiles to load
                try {
                    await page.waitForFunction('window.checkIdle() === true', { timeout: 10000 });
                } catch (e) {
                    // Proceed if it takes too long
                }
                await page.waitForTimeout(200); // Extra buffer for software rendering draw

                const framePath = path.join(outDir, `frame_${String(i).padStart(4, '0')}.jpg`);
                await page.screenshot({ path: framePath, type: 'jpeg', quality: 95 });
                capturedFrames++;
            } catch (frameErr) {
                const msg = String(frameErr && frameErr.message ? frameErr.message : frameErr);
                console.log(`\n[CesiumRenderer] Frame capture interrupted at ${i}/${totalFrames}: ${msg}`);
                break;
            }

            if (i % 10 === 0) {
                process.stdout.write(`\r  Frame ${i}/${totalFrames} [ ${(i / totalFrames * 100).toFixed(1)}% ] `);
            }
        }

        console.log(`\n[CesiumRenderer] Finished capture (${capturedFrames}/${totalFrames}).`);
        if (capturedFrames === 0) {
            process.exitCode = 1;
        }
    } catch (err) {
        console.error(`[CesiumRenderer] Fatal error: ${String(err && err.message ? err.message : err)}`);
        process.exitCode = 1;
    } finally {
        try {
            if (browser) {
                await browser.close();
            }
        } catch (_) {
            // ignore close errors
        }
        try {
            if (serverHandle && serverHandle.server) {
                await new Promise((resolve, reject) => {
                    serverHandle.server.close((error) => error ? reject(error) : resolve());
                });
            }
        } catch (_) {
            // ignore close errors
        }
    }
})();
