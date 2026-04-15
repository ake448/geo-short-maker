const DEFAULT_CAPTURE_OPTIONS = Object.freeze({
    labels: false,
    roads: false,
    buildings: false,
    lighting: true,
    nightMode: false
});

const SHOT_PRESETS = Object.freeze({
    orbit: {
        description: 'High-altitude orbital establishing move around a region.',
        captureOptions: { labels: false, roads: false, buildings: false, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 22000000, heading: 0, pitch: -90, roll: 0 },
            { t: 0.24, height: 9000000, heading: 18, pitch: -86, roll: 0 },
            { t: 0.58, height: 2200000, heading: 72, pitch: -68, roll: 0 },
            { t: 1.0, height: 420000, heading: 138, pitch: -56, roll: 0 }
        ]
    },
    flyover: {
        description: 'Regional-to-local oblique flyover suited for geography beats.',
        captureOptions: { labels: false, roads: false, buildings: false, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 520000, heading: 26, pitch: -60, roll: 0 },
            { t: 0.35, height: 240000, heading: 44, pitch: -58, roll: 0 },
            { t: 0.7, height: 120000, heading: 68, pitch: -54, roll: 0 },
            { t: 1.0, height: 80000, heading: 92, pitch: -48, roll: 0 }
        ]
    },
    zoom: {
        description: 'Vertical descent from orbital context into a target area.',
        captureOptions: { labels: false, roads: false, buildings: false, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 18000000, heading: 6, pitch: -90, roll: 0 },
            { t: 0.3, height: 4600000, heading: 12, pitch: -84, roll: 0 },
            { t: 0.72, height: 520000, heading: 22, pitch: -70, roll: 0 },
            { t: 1.0, height: 120000, heading: 34, pitch: -58, roll: 0 }
        ]
    },
    close_oblique: {
        description: 'Low-altitude terrain-forward oblique move.',
        captureOptions: { labels: false, roads: false, buildings: false, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 60000, heading: 18, pitch: -47, roll: 0 },
            { t: 0.45, height: 42000, heading: 34, pitch: -43, roll: 0 },
            { t: 1.0, height: 30000, heading: 54, pitch: -40, roll: 0 }
        ]
    },
    gtazoom: {
        description: 'Fast city descent into roads and buildings.',
        captureOptions: { labels: false, roads: true, buildings: true, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 16000000, heading: 20, pitch: -90, roll: 0 },
            { t: 0.22, height: 2800000, heading: 28, pitch: -84, roll: 0 },
            { t: 0.52, height: 220000, heading: 42, pitch: -66, roll: 0 },
            { t: 0.78, height: 12000, heading: 58, pitch: -54, roll: 0 },
            { t: 1.0, height: 2200, heading: 74, pitch: -38, roll: 0 }
        ]
    },
    skyline_descend: {
        description: 'Metro hero shot with buildings enabled near the end.',
        captureOptions: { labels: false, roads: true, buildings: true, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 320000, heading: 14, pitch: -62, roll: 0 },
            { t: 0.4, height: 90000, heading: 28, pitch: -55, roll: 0 },
            { t: 0.78, height: 18000, heading: 40, pitch: -48, roll: 0 },
            { t: 1.0, height: 3500, heading: 52, pitch: -34, roll: 0 }
        ]
    },
    night_city: {
        description: 'Night-time city reveal with lights, roads, and buildings.',
        captureOptions: { labels: true, roads: true, buildings: true, lighting: false, nightMode: true },
        keyframes: [
            { t: 0.0, height: 1800000, heading: 18, pitch: -78, roll: 0 },
            { t: 0.34, height: 260000, heading: 28, pitch: -66, roll: 0 },
            { t: 0.7, height: 22000, heading: 44, pitch: -52, roll: 0 },
            { t: 1.0, height: 5000, heading: 58, pitch: -38, roll: 0 }
        ]
    },
    region_reveal: {
        description: 'Wide region intro for country or province facts.',
        captureOptions: { labels: true, roads: false, buildings: false, lighting: true, nightMode: false },
        keyframes: [
            { t: 0.0, height: 12000000, heading: 0, pitch: -88, roll: 0 },
            { t: 0.4, height: 2600000, heading: 12, pitch: -78, roll: 0 },
            { t: 0.75, height: 680000, heading: 24, pitch: -64, roll: 0 },
            { t: 1.0, height: 240000, heading: 36, pitch: -56, roll: 0 }
        ]
    }
});

function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function easeInOutCubic(t) {
    const x = clamp01(t);
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

function interpolateFrame(a, b, t) {
    const eased = easeInOutCubic(t);
    return {
        height: lerp(a.height, b.height, eased),
        heading: lerp(a.heading, b.heading, eased),
        pitch: lerp(a.pitch, b.pitch, eased),
        roll: lerp(a.roll || 0, b.roll || 0, eased)
    };
}

function resolveShotPreset(name) {
    const key = String(name || 'orbit').trim().toLowerCase();
    return SHOT_PRESETS[key] || SHOT_PRESETS.orbit;
}

function listShotPresets() {
    return Object.keys(SHOT_PRESETS);
}

function mergeCaptureOptions(baseOptions, overrides) {
    return {
        ...DEFAULT_CAPTURE_OPTIONS,
        ...(baseOptions || {}),
        ...(overrides || {})
    };
}

function buildShotTimeline({ shot = 'orbit', lat, lon, duration = 5, fps = 30, captureOptions = {}, renderMode = 'auto' }) {
    const preset = resolveShotPreset(shot);
    const safeDuration = Math.max(0.5, Number(duration) || 5);
    const safeFps = Math.max(1, Number(fps) || 30);
    const totalFrames = Math.max(1, Math.floor(safeDuration * safeFps));
    const keyframes = [...preset.keyframes].sort((a, b) => a.t - b.t);
    const options = mergeCaptureOptions(preset.captureOptions, captureOptions);
    if (String(renderMode).toLowerCase() === 'night') {
        options.nightMode = true;
        options.lighting = false;
    }

    const frames = [];
    for (let i = 0; i < totalFrames; i++) {
        const p = totalFrames === 1 ? 1 : i / (totalFrames - 1);
        let left = keyframes[0];
        let right = keyframes[keyframes.length - 1];
        for (let j = 0; j < keyframes.length - 1; j++) {
            if (p >= keyframes[j].t && p <= keyframes[j + 1].t) {
                left = keyframes[j];
                right = keyframes[j + 1];
                break;
            }
        }
        const localT = left === right ? 0 : (p - left.t) / Math.max(1e-6, right.t - left.t);
        const frame = interpolateFrame(left, right, localT);
        frames.push({
            index: i,
            progress: p,
            lat: Number(lat),
            lon: Number(lon),
            ...frame,
            captureOptions: options
        });
    }

    return {
        shot: shot in SHOT_PRESETS ? shot : 'orbit',
        description: preset.description,
        duration: safeDuration,
        fps: safeFps,
        totalFrames,
        captureOptions: options,
        frames
    };
}

module.exports = {
    DEFAULT_CAPTURE_OPTIONS,
    SHOT_PRESETS,
    buildShotTimeline,
    listShotPresets,
    mergeCaptureOptions,
    resolveShotPreset
};
