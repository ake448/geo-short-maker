const runtimeConfig = window.RUNTIME_CONFIG || {};
const queryParams = new URLSearchParams(window.location.search);
const CESIUM_ION_TOKEN = queryParams.get('cesium_ion_token') || runtimeConfig.cesiumIonToken || '';
const MAPBOX_TOKEN = queryParams.get('mapbox_token') || runtimeConfig.mapboxToken || '';
const RENDER_MODE = (queryParams.get('render_mode') || 'day').toLowerCase();
const AUTOMATION_MODE = ['1', 'true', 'yes'].includes(String(queryParams.get('automation') || '').toLowerCase());
const HIDE_UI = AUTOMATION_MODE || ['1', 'true', 'yes'].includes(String(queryParams.get('hide_ui') || '').toLowerCase());

if (CESIUM_ION_TOKEN) {
    Cesium.Ion.defaultAccessToken = CESIUM_ION_TOKEN;
}

if (HIDE_UI) {
    document.body.classList.add('capture-mode');
}

const elements = {
    searchForm: document.getElementById('searchForm'),
    searchInput: document.getElementById('searchInput'),
    searchResults: document.getElementById('searchResults'),
    buildingsToggle: document.getElementById('buildingsToggle'),
    roadsToggle: document.getElementById('roadsToggle'),
    labelsToggle: document.getElementById('labelsToggle'),
    lightingToggle: document.getElementById('lightingToggle'),
    nightModeToggle: document.getElementById('nightModeToggle'),
    loader: document.getElementById('loader'),
    terrainChip: document.getElementById('terrainChip'),
    modeChip: document.getElementById('modeChip'),
    overlayChip: document.getElementById('overlayChip'),
    networkChip: document.getElementById('networkChip'),
    altitudeValue: document.getElementById('altitudeValue'),
    pitchValue: document.getElementById('pitchValue'),
    latitudeValue: document.getElementById('latitudeValue'),
    longitudeValue: document.getElementById('longitudeValue'),
    roadsValue: document.getElementById('roadsValue'),
    buildingsValue: document.getElementById('buildingsValue')
};

const viewer = new Cesium.Viewer('cesiumContainer', {
    sceneMode: Cesium.SceneMode.SCENE3D,
    scene3DOnly: true,
    baseLayer: false,
    imageryProvider: false,
    animation: false,
    timeline: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    navigationHelpButton: false,
    infoBox: false,
    fullscreenButton: false,
    selectionIndicator: false,
    vrButton: false,
    shouldAnimate: false,
    requestRenderMode: true,
    maximumRenderTimeChange: Infinity,
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    creditContainer: document.getElementById('cesiumCredits'),
    contextOptions: {
        requestWebgl2: true,
        webgl: {
            alpha: false,
            antialias: true,
            powerPreference: 'high-performance'
        }
    }
});

viewer.imageryLayers.removeAll();

const imageryLayer = viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    credit: 'Esri, Maxar, Earthstar Geographics, and the GIS User Community'
}));

const darkMapLayer = MAPBOX_TOKEN ? viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
    url: `https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/256/{z}/{x}/{y}?access_token=${MAPBOX_TOKEN}`,
    credit: 'Mapbox'
})) : null;

if (darkMapLayer) {
    darkMapLayer.show = false;
    darkMapLayer.alpha = 1;
}

const nightLightsLayer = viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
    url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_CityLights_2012/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg',
    credit: 'NASA Black Marble / GIBS',
    tilingScheme: new Cesium.WebMercatorTilingScheme(),
    maximumLevel: 8
}));
nightLightsLayer.show = false;
nightLightsLayer.alpha = 0;

const labelsLayer = viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
    url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    credit: 'Esri'
}));
labelsLayer.alpha = 0.92;

viewer.scene.highDynamicRange = false;
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.dynamicAtmosphereLighting = true;
viewer.scene.globe.dynamicAtmosphereLightingFromSun = true;
viewer.scene.globe.showGroundAtmosphere = true;
viewer.scene.globe.depthTestAgainstTerrain = true;
viewer.scene.skyAtmosphere.show = true;
viewer.scene.fog.enabled = true;
viewer.scene.fog.density = 0.00008;
viewer.scene.globe.maximumScreenSpaceError = 3;
viewer.scene.globe.tileCacheSize = 450;
viewer.scene.postProcessStages.fxaa.enabled = false;
viewer.resolutionScale = Math.min(window.devicePixelRatio || 1, 1);

viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
viewer.camera.percentageChanged = 0.02;
viewer.scene.screenSpaceCameraController.minimumZoomDistance = 40;
viewer.scene.screenSpaceCameraController.maximumZoomDistance = 40000000;
viewer.scene.screenSpaceCameraController.enableCollisionDetection = true;

const roadsSource = new Cesium.CustomDataSource('roads');
const buildingsSource = new Cesium.CustomDataSource('buildings');
viewer.dataSources.add(roadsSource);
viewer.dataSources.add(buildingsSource);

const state = {
    terrainLabel: 'Ellipsoid',
    terrainReady: false,
    nightMode: false,
    automationMode: AUTOMATION_MODE,
    lastOverlayKey: null,
    overlayCache: new Map(),
    pendingOverlayController: null,
    searchAbortController: null,
    roadsCount: 0,
    buildingsCount: 0,
    lastSearchResults: [],
    performanceBucket: null,
    overlayMode: 'roads+buildings',
    overlayRequestInFlight: false,
    lastRenderTick: 0
};

const roadStyleMap = {
    motorway: { width: 4.5, color: '#ffd36c' },
    trunk: { width: 4.0, color: '#ffc266' },
    primary: { width: 3.5, color: '#f8b35b' },
    secondary: { width: 3.0, color: '#f59f58' },
    tertiary: { width: 2.5, color: '#f6c975' },
    residential: { width: 2.0, color: '#d8eefb' },
    service: { width: 1.6, color: '#b9d5e5' },
    unclassified: { width: 1.8, color: '#cfe4f0' }
};

function debounce(fn, delayMs) {
    let timerId = null;
    return (...args) => {
        window.clearTimeout(timerId);
        timerId = window.setTimeout(() => fn(...args), delayMs);
    };
}

function setChip(target, text, stateName) {
    target.textContent = text;
    target.dataset.state = stateName || '';
}

function getNightLightAlpha() {
    const height = viewer.camera.positionCartographic.height;
    if (height > 12000000) {
        return 0.95;
    }
    if (height > 2000000) {
        return 0.82;
    }
    if (height > 200000) {
        return 0.68;
    }
    return 0.45;
}

function applyNightMode() {
    state.nightMode = !!elements.nightModeToggle.checked;
    imageryLayer.alpha = state.nightMode ? (darkMapLayer ? 0.12 : 0.24) : 1;
    imageryLayer.brightness = state.nightMode ? 0.24 : 1.0;
    imageryLayer.contrast = state.nightMode ? 1.04 : 1.0;
    imageryLayer.gamma = state.nightMode ? 0.92 : 1.0;
    imageryLayer.saturation = state.nightMode ? 0.18 : 1.0;
    imageryLayer.hue = 0;

    if (darkMapLayer) {
        darkMapLayer.show = state.nightMode;
        darkMapLayer.alpha = state.nightMode ? 0.96 : 0;
        darkMapLayer.brightness = 0.9;
        darkMapLayer.contrast = 1.02;
        darkMapLayer.saturation = 0.16;
    }

    nightLightsLayer.show = state.nightMode;
    nightLightsLayer.alpha = state.nightMode ? getNightLightAlpha() : 0;
    labelsLayer.alpha = state.nightMode ? 0.78 : 0.92;
    viewer.scene.skyAtmosphere.hueShift = 0;
    viewer.scene.skyAtmosphere.saturationShift = state.nightMode ? -0.92 : 0;
    viewer.scene.skyAtmosphere.brightnessShift = state.nightMode ? -0.58 : 0;
    viewer.scene.globe.atmosphereBrightnessShift = state.nightMode ? -0.28 : 0;
    viewer.scene.backgroundColor = state.nightMode ? Cesium.Color.fromCssColorString('#040404') : Cesium.Color.BLACK;
    viewer.scene.globe.enableLighting = elements.lightingToggle.checked && !state.nightMode;
    setChip(elements.modeChip, state.nightMode ? 'Mode: night lights' : 'Mode: day', state.nightMode ? 'active' : '');
    viewer.scene.requestRender();
}

function updatePerformanceProfile() {
    const height = viewer.camera.positionCartographic.height;
    let bucket = 'local';
    let resolutionScale = 1;
    let sse = 2.2;
    let fogDensity = 0.00005;

    if (height > 4000000) {
        bucket = 'orbital';
        resolutionScale = 0.7;
        sse = 5.5;
        fogDensity = 0.00018;
    } else if (height > 250000) {
        bucket = 'regional';
        resolutionScale = 0.82;
        sse = 4;
        fogDensity = 0.0001;
    } else if (height > 30000) {
        bucket = 'metro';
        resolutionScale = 0.92;
        sse = 3.2;
        fogDensity = 0.00008;
    }

    if (state.performanceBucket !== bucket) {
        state.performanceBucket = bucket;
        viewer.resolutionScale = resolutionScale;
        viewer.scene.globe.maximumScreenSpaceError = sse;
        viewer.scene.fog.density = fogDensity;
        viewer.scene.requestRender();
    }

    if (state.nightMode) {
        nightLightsLayer.alpha = getNightLightAlpha();
    }
}

function formatNumber(value, digits = 2) {
    return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function formatAltitude(meters) {
    if (!Number.isFinite(meters)) {
        return '—';
    }
    if (meters >= 1000000) {
        return `${(meters / 1000000).toFixed(2)} Mm`;
    }
    if (meters >= 1000) {
        return `${(meters / 1000).toFixed(2)} km`;
    }
    return `${meters.toFixed(0)} m`;
}

function hideLoader() {
    elements.loader.classList.add('hidden');
    window.setTimeout(() => {
        elements.loader.style.display = 'none';
    }, 220);
}

async function initializeTerrain() {
    try {
        if (CESIUM_ION_TOKEN && typeof Cesium.createWorldTerrainAsync === 'function') {
            viewer.terrainProvider = await Cesium.createWorldTerrainAsync({
                requestVertexNormals: true,
                requestWaterMask: true
            });
            state.terrainLabel = 'Cesium World Terrain';
            state.terrainReady = true;
            setChip(elements.terrainChip, 'Terrain: Cesium World Terrain', 'active');
            return;
        }
    } catch (error) {
        console.warn('Falling back from Cesium terrain:', error);
    }

    state.terrainReady = true;
    setChip(elements.terrainChip, 'Terrain: ellipsoid fallback', 'warn');
}

function updateCameraMetrics() {
    const cartographic = viewer.camera.positionCartographic;
    elements.altitudeValue.textContent = formatAltitude(cartographic.height);
    elements.pitchValue.textContent = `${formatNumber(Cesium.Math.toDegrees(viewer.camera.pitch), 1)}°`;
    elements.latitudeValue.textContent = `${formatNumber(Cesium.Math.toDegrees(cartographic.latitude), 4)}°`;
    elements.longitudeValue.textContent = `${formatNumber(Cesium.Math.toDegrees(cartographic.longitude), 4)}°`;
    elements.roadsValue.textContent = state.roadsCount.toLocaleString();
    elements.buildingsValue.textContent = state.buildingsCount.toLocaleString();
}

function getCameraState() {
    const cartographic = viewer.camera.positionCartographic;
    return {
        lat: Cesium.Math.toDegrees(cartographic.latitude),
        lon: Cesium.Math.toDegrees(cartographic.longitude),
        height: cartographic.height,
        heading: Cesium.Math.toDegrees(viewer.camera.heading),
        pitch: Cesium.Math.toDegrees(viewer.camera.pitch),
        roll: Cesium.Math.toDegrees(viewer.camera.roll),
        roadsCount: state.roadsCount,
        buildingsCount: state.buildingsCount,
        terrainReady: state.terrainReady,
        overlayMode: state.overlayMode,
        nightMode: state.nightMode
    };
}

function isLoaderVisible() {
    return elements.loader.style.display !== 'none' && !elements.loader.classList.contains('hidden');
}

function isSceneIdle() {
    const globeReady = !!viewer.scene.globe.tilesLoaded;
    return state.terrainReady && globeReady && !state.overlayRequestInFlight && !isLoaderVisible();
}

function getViewportCenterCartographic() {
    const canvas = viewer.scene.canvas;
    const center = new Cesium.Cartesian2(canvas.clientWidth / 2, canvas.clientHeight / 2);
    const ray = viewer.camera.getPickRay(center);
    let point = null;

    if (ray) {
        point = viewer.scene.globe.pick(ray, viewer.scene);
    }

    if (!point) {
        point = viewer.camera.pickEllipsoid(center, viewer.scene.globe.ellipsoid);
    }

    if (!point) {
        return viewer.camera.positionCartographic;
    }

    return Cesium.Cartographic.fromCartesian(point);
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function getOverlayBBox() {
    const center = getViewportCenterCartographic();
    const lat = Cesium.Math.toDegrees(center.latitude);
    const lon = Cesium.Math.toDegrees(center.longitude);
    const height = viewer.camera.positionCartographic.height;
    const halfSideKm = clamp(height / 3600, 0.22, 7);
    const latDelta = halfSideKm / 111;
    const lonScale = Math.max(Math.cos(Cesium.Math.toRadians(Math.max(8, Math.abs(lat)))), 0.18);
    const lonDelta = halfSideKm / (111 * lonScale);

    return {
        south: clamp(lat - latDelta, -85, 85),
        west: clamp(lon - lonDelta, -179.9, 179.9),
        north: clamp(lat + latDelta, -85, 85),
        east: clamp(lon + lonDelta, -179.9, 179.9)
    };
}

function buildOverlayKey(bbox) {
    return [bbox.south, bbox.west, bbox.north, bbox.east, state.overlayMode].map((value) => typeof value === 'number' ? value.toFixed(3) : value).join(':');
}

function estimateBuildingHeight(tags) {
    if (!tags) {
        return 16;
    }

    const numericHeight = Number.parseFloat(tags.height || tags['building:height']);
    if (Number.isFinite(numericHeight) && numericHeight > 2) {
        return numericHeight;
    }

    const levels = Number.parseFloat(tags['building:levels']);
    if (Number.isFinite(levels) && levels > 0) {
        return Math.max(8, levels * 3.2);
    }

    const buildingType = String(tags.building || '').toLowerCase();
    if (buildingType.includes('industrial') || buildingType.includes('warehouse')) {
        return 18;
    }
    if (buildingType.includes('commercial') || buildingType.includes('office')) {
        return 28;
    }
    return 14;
}

function applyOverlayData(elementsData) {
    roadsSource.entities.removeAll();
    buildingsSource.entities.removeAll();

    let roadsCount = 0;
    let buildingsCount = 0;

    if (elements.roadsToggle.checked) {
        for (const item of elementsData) {
            if (item.type !== 'way' || !item.tags || !item.tags.highway || !Array.isArray(item.geometry) || item.geometry.length < 2) {
                continue;
            }

            const style = roadStyleMap[item.tags.highway] || roadStyleMap.unclassified;
            const positions = [];
            for (const point of item.geometry) {
                positions.push(point.lon, point.lat);
            }

            roadsSource.entities.add({
                polyline: {
                    positions: Cesium.Cartesian3.fromDegreesArray(positions),
                    clampToGround: true,
                    width: style.width,
                    material: Cesium.Color.fromCssColorString(style.color).withAlpha(0.9)
                }
            });
            roadsCount += 1;
        }
    }

    if (elements.buildingsToggle.checked) {
        for (const item of elementsData) {
            if (item.type !== 'way' || !item.tags || !item.tags.building || !Array.isArray(item.geometry) || item.geometry.length < 3) {
                continue;
            }

            const geometry = item.geometry.slice();
            const first = geometry[0];
            const last = geometry[geometry.length - 1];
            if (first.lat !== last.lat || first.lon !== last.lon) {
                geometry.push(first);
            }

            const hierarchy = [];
            for (const point of geometry) {
                hierarchy.push(point.lon, point.lat);
            }

            const buildingHeight = estimateBuildingHeight(item.tags);
            buildingsSource.entities.add({
                polygon: {
                    hierarchy: Cesium.Cartesian3.fromDegreesArray(hierarchy),
                    material: Cesium.Color.fromCssColorString('#87a7c7').withAlpha(0.66),
                    outline: false,
                    height: 0,
                    extrudedHeight: buildingHeight,
                    shadows: Cesium.ShadowMode.DISABLED
                }
            });
            buildingsCount += 1;
        }
    }

    state.roadsCount = roadsCount;
    state.buildingsCount = buildingsCount;
    updateCameraMetrics();

    if (roadsCount || buildingsCount) {
        setChip(elements.overlayChip, `Overlay: ${roadsCount} roads · ${buildingsCount} buildings`, 'active');
    } else {
        setChip(elements.overlayChip, 'Overlay: none in view', 'warn');
    }

    viewer.scene.requestRender();
}

function clearOverlayData(reason) {
    roadsSource.entities.removeAll();
    buildingsSource.entities.removeAll();
    state.roadsCount = 0;
    state.buildingsCount = 0;
    updateCameraMetrics();
    setChip(elements.overlayChip, `Overlay: ${reason}`, '');
    viewer.scene.requestRender();
}

async function fetchOverlayData(bbox, key) {
    if (state.pendingOverlayController) {
        state.pendingOverlayController.abort();
    }

    const controller = new AbortController();
    state.pendingOverlayController = controller;
    state.overlayRequestInFlight = true;
    setChip(elements.networkChip, 'Network: loading streets', 'active');

    const queryParts = [];
    if (state.overlayMode.includes('roads')) {
        queryParts.push(`way["highway"]["area"!="yes"](${bbox.south},${bbox.west},${bbox.north},${bbox.east});`);
    }
    if (state.overlayMode.includes('buildings')) {
        queryParts.push(`way["building"](${bbox.south},${bbox.west},${bbox.north},${bbox.east});`);
    }

    const overpassQuery = `
        [out:json][timeout:18];
        (
          ${queryParts.join('\n          ')}
        );
        out body geom qt;
    `;

    const response = await fetch('https://overpass-api.de/api/interpreter', {
        method: 'POST',
        body: overpassQuery,
        signal: controller.signal,
        headers: {
            'Content-Type': 'text/plain;charset=UTF-8'
        }
    });

    if (!response.ok) {
        throw new Error(`Overpass returned ${response.status}`);
    }

    try {
        const payload = await response.json();
        state.overlayCache.set(key, payload.elements || []);
        return payload.elements || [];
    } finally {
        if (state.pendingOverlayController === controller) {
            state.pendingOverlayController = null;
        }
        state.overlayRequestInFlight = false;
    }
}

const refreshOverlayData = debounce(async () => {
    const height = viewer.camera.positionCartographic.height;

    const roadsAllowed = elements.roadsToggle.checked && height <= 18000;
    const buildingsAllowed = elements.buildingsToggle.checked && height <= 6500;
    state.overlayMode = [roadsAllowed ? 'roads' : '', buildingsAllowed ? 'buildings' : ''].filter(Boolean).join('+') || 'none';

    if (!roadsAllowed && !buildingsAllowed) {
        state.lastOverlayKey = null;
        state.overlayRequestInFlight = false;
        clearOverlayData('zoom in for streets');
        setChip(elements.networkChip, 'Network: idle', '');
        return;
    }

    const bbox = getOverlayBBox();
    const key = buildOverlayKey(bbox);

    if (key === state.lastOverlayKey) {
        return;
    }

    state.lastOverlayKey = key;

    if (state.overlayCache.has(key)) {
        applyOverlayData(state.overlayCache.get(key));
        setChip(elements.networkChip, 'Network: cache hit', 'active');
        return;
    }

    try {
        const elementsData = await fetchOverlayData(bbox, key);
        applyOverlayData(elementsData);
        setChip(elements.networkChip, 'Network: live OSM data', 'active');
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        console.warn('Overlay fetch failed:', error);
        state.overlayRequestInFlight = false;
        setChip(elements.networkChip, 'Network: overlay unavailable', 'error');
        setChip(elements.overlayChip, 'Overlay: failed to load', 'error');
    }
}, 450);

function flyToLocation(lat, lon, height = 16000, heading = 18, pitch = -55) {
    viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
        orientation: {
            heading: Cesium.Math.toRadians(heading),
            pitch: Cesium.Math.toRadians(pitch),
            roll: 0
        },
        duration: 2.8,
        complete: () => {
            refreshOverlayData();
            viewer.scene.requestRender();
        }
    });
}

function setCamera(lat, lon, height = 16000, heading = 18, pitch = -55, durationSeconds = 0, roll = 0) {
    const destination = Cesium.Cartesian3.fromDegrees(Number(lon), Number(lat), Number(height));
    const orientation = {
        heading: Cesium.Math.toRadians(Number(heading) || 0),
        pitch: Cesium.Math.toRadians(Number(pitch) || -90),
        roll: Cesium.Math.toRadians(Number(roll) || 0)
    };

    if (Number(durationSeconds) > 0) {
        viewer.camera.flyTo({
            destination,
            orientation,
            duration: Number(durationSeconds),
            complete: () => {
                updateCameraMetrics();
                updatePerformanceProfile();
                refreshOverlayData();
                viewer.scene.requestRender();
            }
        });
        return;
    }

    viewer.camera.setView({
        destination,
        orientation
    });
    updateCameraMetrics();
    updatePerformanceProfile();
    refreshOverlayData();
    viewer.scene.requestRender();
}

function setCaptureOptions(options = {}) {
    if (typeof options.labels === 'boolean') {
        elements.labelsToggle.checked = options.labels;
    }
    if (typeof options.roads === 'boolean') {
        elements.roadsToggle.checked = options.roads;
    }
    if (typeof options.buildings === 'boolean') {
        elements.buildingsToggle.checked = options.buildings;
    }
    if (typeof options.lighting === 'boolean') {
        elements.lightingToggle.checked = options.lighting;
    }
    if (typeof options.nightMode === 'boolean') {
        elements.nightModeToggle.checked = options.nightMode;
    }
    syncLayerVisibility();
}

window.setCamera = setCamera;
window.checkIdle = isSceneIdle;
window.getCameraState = getCameraState;
window.setCaptureOptions = setCaptureOptions;
window.earthTwin = {
    viewer,
    state,
    setCamera,
    checkIdle: isSceneIdle,
    getCameraState,
    setCaptureOptions,
    flyToLocation,
    runSearch
};

function renderSearchResults(results) {
    state.lastSearchResults = results;

    if (!results.length) {
        elements.searchResults.innerHTML = '<div class="results-empty">No matches found. Try a broader city, region, or landmark.</div>';
        return;
    }

    elements.searchResults.innerHTML = '';
    for (const result of results) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'search-result';
        button.textContent = result.display_name;
        button.addEventListener('click', () => {
            const height = Number.parseFloat(result.type === 'city' || result.type === 'administrative' ? 28000 : 14000);
            flyToLocation(Number.parseFloat(result.lat), Number.parseFloat(result.lon), height);
        });
        elements.searchResults.appendChild(button);
    }
}

async function runSearch(query) {
    if (!query.trim()) {
        renderSearchResults([]);
        return;
    }

    if (state.searchAbortController) {
        state.searchAbortController.abort();
    }

    const controller = new AbortController();
    state.searchAbortController = controller;
    setChip(elements.networkChip, 'Network: searching', 'active');

    try {
        const url = new URL('https://nominatim.openstreetmap.org/search');
        url.searchParams.set('format', 'jsonv2');
        url.searchParams.set('limit', '6');
        url.searchParams.set('q', query.trim());

        const response = await fetch(url, {
            signal: controller.signal,
            headers: {
                Accept: 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Nominatim returned ${response.status}`);
        }

        const results = await response.json();
        renderSearchResults(results);
        setChip(elements.networkChip, 'Network: search ready', 'active');
        if (results.length) {
            const first = results[0];
            const height = Number.parseFloat(first.type === 'city' || first.type === 'administrative' ? 28000 : 14000);
            flyToLocation(Number.parseFloat(first.lat), Number.parseFloat(first.lon), height);
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        console.warn('Search failed:', error);
        setChip(elements.networkChip, 'Network: search failed', 'error');
        elements.searchResults.innerHTML = '<div class="results-empty">Search is temporarily unavailable.</div>';
    }
}

function syncLayerVisibility() {
    labelsLayer.show = elements.labelsToggle.checked;
    applyNightMode();

    if (!elements.roadsToggle.checked && !elements.buildingsToggle.checked) {
        clearOverlayData('layers disabled');
        return;
    }

    if (state.lastOverlayKey && state.overlayCache.has(state.lastOverlayKey)) {
        applyOverlayData(state.overlayCache.get(state.lastOverlayKey));
    } else {
        refreshOverlayData();
    }
}

function bindUi() {
    elements.searchForm.addEventListener('submit', (event) => {
        event.preventDefault();
        runSearch(elements.searchInput.value);
    });

    document.querySelectorAll('.preset-btn').forEach((button) => {
        button.addEventListener('click', () => {
            flyToLocation(
                Number.parseFloat(button.dataset.lat),
                Number.parseFloat(button.dataset.lon),
                Number.parseFloat(button.dataset.height),
                Number.parseFloat(button.dataset.heading),
                Number.parseFloat(button.dataset.pitch)
            );
        });
    });

    [elements.buildingsToggle, elements.roadsToggle, elements.labelsToggle, elements.lightingToggle, elements.nightModeToggle].forEach((input) => {
        input.addEventListener('change', syncLayerVisibility);
    });

    viewer.camera.changed.addEventListener(updateCameraMetrics);
    viewer.camera.changed.addEventListener(updatePerformanceProfile);
    viewer.camera.moveEnd.addEventListener(() => {
        updatePerformanceProfile();
        refreshOverlayData();
    });
    window.addEventListener('resize', () => viewer.scene.requestRender());
}

async function initialize() {
    bindUi();
    updateCameraMetrics();
    updatePerformanceProfile();

    await initializeTerrain();

    viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(-20, 22, 23000000),
        orientation: {
            heading: 0,
            pitch: Cesium.Math.toRadians(-90),
            roll: 0
        }
    });

    if (RENDER_MODE === 'night') {
        elements.nightModeToggle.checked = true;
    }

    applyNightMode();
    hideLoader();
    setChip(elements.overlayChip, 'Overlay: zoom in for streets', '');
    setChip(elements.networkChip, 'Network: idle', '');
    refreshOverlayData();
    viewer.scene.requestRender();
    window.dispatchEvent(new CustomEvent('earth-twin-ready', { detail: getCameraState() }));
}

initialize().catch((error) => {
    console.error('Earth digital twin initialization failed:', error);
    setChip(elements.terrainChip, 'Terrain: failed to initialize', 'error');
    setChip(elements.networkChip, 'Network: failed to initialize', 'error');
    hideLoader();
});