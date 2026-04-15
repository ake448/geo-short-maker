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

async function main() {
    loadDotEnv(path.resolve(__dirname, '.env'));

    const examples = [
        {
            name: 'new_york_day',
            lat: 40.7128,
            lon: -74.0060,
            height: 22000,
            heading: 24,
            pitch: -58,
            mode: 'day',
            options: { labels: true, roads: true, buildings: true, lighting: true, nightMode: false }
        },
        {
            name: 'tokyo_night',
            lat: 35.6762,
            lon: 139.6503,
            height: 18000,
            heading: 36,
            pitch: -62,
            mode: 'night',
            options: { labels: true, roads: true, buildings: true, lighting: false, nightMode: true }
        },
        {
            name: 'cairo_day',
            lat: 30.0444,
            lon: 31.2357,
            height: 26000,
            heading: 18,
            pitch: -54,
            mode: 'day',
            options: { labels: true, roads: false, buildings: false, lighting: true, nightMode: false }
        }
    ];

    const outputDir = path.resolve(__dirname, 'examples', 'earth_twin_shots');
    fs.mkdirSync(outputDir, { recursive: true });

    let browser = null;
    let serverHandle = null;

    try {
        serverHandle = await startServer({ requestedPort: 0, host: '127.0.0.1' });
        browser = await chromium.launch({
            headless: true,
            args: ['--disable-dev-shm-usage']
        });

        const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
        page.on('console', (msg) => console.log(`[PAGE] ${msg.text()}`));

        for (const example of examples) {
            const url = new URL(`http://${serverHandle.host}:${serverHandle.port}/earth_digital_twin.html`);
            url.searchParams.set('automation', '1');
            url.searchParams.set('hide_ui', '1');
            url.searchParams.set('render_mode', example.mode);
            if (process.env.CESIUM_ION_TOKEN) {
                url.searchParams.set('cesium_ion_token', process.env.CESIUM_ION_TOKEN);
            }
            if (process.env.MAPBOX_TOKEN) {
                url.searchParams.set('mapbox_token', process.env.MAPBOX_TOKEN);
            }

            await page.goto(url.toString(), { waitUntil: 'domcontentloaded' });
            await page.waitForFunction(() => !!window.earthTwin && typeof window.setCamera === 'function', { timeout: 30000 });
            await page.evaluate((options) => window.setCaptureOptions(options), example.options);
            await page.evaluate((camera) => window.setCamera(camera.lat, camera.lon, camera.height, camera.heading, camera.pitch, 0), example);
            await page.waitForFunction(() => window.checkIdle() === true, { timeout: 30000 }).catch(() => null);
            await page.waitForTimeout(1200);

            const outputPath = path.join(outputDir, `${example.name}.jpg`);
            await page.screenshot({ path: outputPath, type: 'jpeg', quality: 95 });
            console.log(`[Example] Saved ${outputPath}`);
        }
    } finally {
        if (browser) {
            await browser.close().catch(() => {});
        }
        if (serverHandle?.server) {
            await new Promise((resolve) => serverHandle.server.close(() => resolve()));
        }
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
