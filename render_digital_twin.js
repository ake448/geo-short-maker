const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { startServer } = require('./serve_earth');
const { buildShotTimeline, listShotPresets, resolveShotPreset } = require('./digital_twin_shots');

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

function parseArgs(argv) {
    const options = {
        lat: 40.7128,
        lon: -74.0060,
        duration: 5,
        fps: 30,
        shot: 'orbit',
        outDir: './earth_twin_frames',
        renderMode: 'auto',
        labels: undefined,
        roads: undefined,
        buildings: undefined,
        lighting: undefined,
        dryRun: false,
        waitMs: 180,
        timeoutMs: 12000,
        quality: 92
    };

    const args = [...argv];
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        const next = () => args[++i];
        switch (arg) {
            case '--lat': options.lat = Number(next()); break;
            case '--lon': options.lon = Number(next()); break;
            case '--duration': options.duration = Number(next()); break;
            case '--fps': options.fps = Number(next()); break;
            case '--shot': options.shot = String(next() || 'orbit'); break;
            case '--out-dir': options.outDir = String(next() || options.outDir); break;
            case '--render-mode': options.renderMode = String(next() || 'auto'); break;
            case '--labels': options.labels = true; break;
            case '--no-labels': options.labels = false; break;
            case '--roads': options.roads = true; break;
            case '--no-roads': options.roads = false; break;
            case '--buildings': options.buildings = true; break;
            case '--no-buildings': options.buildings = false; break;
            case '--lighting': options.lighting = true; break;
            case '--no-lighting': options.lighting = false; break;
            case '--wait-ms': options.waitMs = Number(next()); break;
            case '--timeout-ms': options.timeoutMs = Number(next()); break;
            case '--quality': options.quality = Number(next()); break;
            case '--dry-run': options.dryRun = true; break;
            case '--list-shots':
                console.log(listShotPresets().join('\n'));
                process.exit(0);
                break;
            case '--help':
            case '-h':
                printHelp();
                process.exit(0);
                break;
            default:
                if (arg.startsWith('--')) {
                    throw new Error(`Unknown argument: ${arg}`);
                }
                break;
        }
    }
    return options;
}

function printHelp() {
    console.log([
        'Usage: node render_digital_twin.js --lat <lat> --lon <lon> [options]',
        '',
        'Options:',
        '  --shot <name>           Shot preset: ' + listShotPresets().join(', '),
        '  --duration <seconds>    Timeline duration, default 5',
        '  --fps <number>          Frames per second, default 30',
        '  --out-dir <path>        Output frames directory',
        '  --render-mode <mode>    auto | day | night',
        '  --labels / --no-labels  Override labels visibility',
        '  --roads / --no-roads    Override roads overlay',
        '  --buildings / --no-buildings  Override building overlay',
        '  --lighting / --no-lighting    Override lighting',
        '  --wait-ms <ms>          Extra wait after idle, default 180',
        '  --timeout-ms <ms>       Idle wait timeout per frame, default 12000',
        '  --quality <0-100>       JPEG quality, default 92',
        '  --dry-run               Print timeline and exit',
        '  --list-shots            List presets and exit'
    ].join('\n'));
}

function ensureDir(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function buildCaptureOverrides(options) {
    return {
        ...(typeof options.labels === 'boolean' ? { labels: options.labels } : {}),
        ...(typeof options.roads === 'boolean' ? { roads: options.roads } : {}),
        ...(typeof options.buildings === 'boolean' ? { buildings: options.buildings } : {}),
        ...(typeof options.lighting === 'boolean' ? { lighting: options.lighting } : {})
    };
}

async function main() {
    loadDotEnv(path.resolve(__dirname, '.env'));
    const options = parseArgs(process.argv.slice(2));
    const outDir = path.resolve(__dirname, options.outDir);
    ensureDir(outDir);

    const timeline = buildShotTimeline({
        shot: options.shot,
        lat: options.lat,
        lon: options.lon,
        duration: options.duration,
        fps: options.fps,
        renderMode: options.renderMode,
        captureOptions: buildCaptureOverrides(options)
    });

    const manifestPath = path.join(outDir, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify({
        shot: timeline.shot,
        description: resolveShotPreset(options.shot).description,
        lat: options.lat,
        lon: options.lon,
        duration: timeline.duration,
        fps: timeline.fps,
        totalFrames: timeline.totalFrames,
        renderMode: options.renderMode,
        captureOptions: timeline.captureOptions
    }, null, 2));

    if (options.dryRun) {
        console.log(JSON.stringify({
            manifest: manifestPath,
            totalFrames: timeline.totalFrames,
            firstFrame: timeline.frames[0],
            midFrame: timeline.frames[Math.floor(timeline.frames.length / 2)],
            lastFrame: timeline.frames[timeline.frames.length - 1]
        }, null, 2));
        return;
    }

    let browser = null;
    let serverHandle = null;
    let capturedFrames = 0;

    try {
        serverHandle = await startServer({ requestedPort: 0, host: '127.0.0.1' });
        browser = await chromium.launch({
            headless: true,
            args: [
                '--disable-gpu',
                '--disable-gpu-compositing',
                '--disable-dev-shm-usage',
                '--ignore-gpu-blocklist',
                '--use-gl=angle',
                '--use-angle=swiftshader'
            ]
        });

        const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
        page.on('console', (msg) => console.log(`[PAGE] ${msg.text()}`));
        page.on('pageerror', (err) => console.log(`[PAGE ERR] ${String(err && err.message ? err.message : err)}`));

        const url = new URL(`http://${serverHandle.host}:${serverHandle.port}/earth_digital_twin.html`);
        url.searchParams.set('automation', '1');
        url.searchParams.set('hide_ui', '1');
        url.searchParams.set('render_mode', String(options.renderMode || 'auto'));
        if (process.env.CESIUM_ION_TOKEN) {
            url.searchParams.set('cesium_ion_token', process.env.CESIUM_ION_TOKEN);
        }
        if (process.env.MAPBOX_TOKEN) {
            url.searchParams.set('mapbox_token', process.env.MAPBOX_TOKEN);
        }

        await page.goto(url.toString(), { waitUntil: 'domcontentloaded', timeout: 60000 });
        await page.waitForFunction(() => typeof window.setCamera === 'function' && typeof window.checkIdle === 'function' && typeof window.setCaptureOptions === 'function', { timeout: 45000 });
        await page.evaluate((captureOptions) => window.setCaptureOptions(captureOptions), timeline.captureOptions);

        for (const frame of timeline.frames) {
            await page.evaluate((camera) => {
                window.setCamera(camera.lat, camera.lon, camera.height, camera.heading, camera.pitch, 0, camera.roll || 0);
            }, frame);

            try {
                await page.waitForFunction(() => window.checkIdle() === true, { timeout: Math.max(1000, options.timeoutMs) });
            } catch (_) {
                // Continue with best-effort capture.
            }

            if (options.waitMs > 0) {
                await page.waitForTimeout(options.waitMs);
            }

            const framePath = path.join(outDir, `frame_${String(frame.index).padStart(4, '0')}.jpg`);
            await page.screenshot({ path: framePath, type: 'jpeg', quality: Math.max(10, Math.min(100, options.quality)) });
            capturedFrames++;
            if (frame.index % 10 === 0) {
                process.stdout.write(`\r  Frame ${frame.index + 1}/${timeline.totalFrames}`);
            }
        }

        console.log(`\n[DigitalTwin] Finished capture (${capturedFrames}/${timeline.totalFrames})`);
        if (capturedFrames === 0) {
            process.exitCode = 1;
        }
    } finally {
        if (browser) {
            await browser.close().catch(() => { });
        }
        if (serverHandle && serverHandle.server) {
            await new Promise((resolve) => serverHandle.server.close(() => resolve()));
        }
    }
}

main().catch((error) => {
    console.error(`[DigitalTwin] ${error && error.message ? error.message : error}`);
    process.exitCode = 1;
});
