const http = require('http');
const fs = require('fs');
const path = require('path');

const rootDir = __dirname;
const defaultFile = 'earth_digital_twin.html';
const port = Number(process.env.PORT || 8080);

function loadDotEnv(dotenvPath) {
    if (!fs.existsSync(dotenvPath)) {
        return;
    }

    const lines = fs.readFileSync(dotenvPath, 'utf8').split(/\r?\n/);
    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) {
            continue;
        }

        const separator = line.indexOf('=');
        if (separator <= 0) {
            continue;
        }

        const key = line.slice(0, separator).trim();
        let value = line.slice(separator + 1).trim();
        if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
        }

        if (key && !(key in process.env)) {
            process.env[key] = value;
        }
    }
}

loadDotEnv(path.join(rootDir, '.env'));

const mimeTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

function send(response, statusCode, body, contentType) {
    response.writeHead(statusCode, {
        'Content-Type': contentType,
        'Cache-Control': 'no-cache'
    });
    response.end(body);
}

function buildRuntimeConfigScript() {
    const config = {
        cesiumIonToken: process.env.CESIUM_ION_TOKEN || '',
        mapboxToken: process.env.MAPBOX_TOKEN || ''
    };

    return `window.RUNTIME_CONFIG = ${JSON.stringify(config)};`;
}

function resolveRequestPath(requestUrl) {
    const url = new URL(requestUrl, 'http://localhost');
    const decodedPath = decodeURIComponent(url.pathname);
    const safePath = decodedPath === '/' ? `/${defaultFile}` : decodedPath;
    const fullPath = path.normalize(path.join(rootDir, safePath));

    if (!fullPath.startsWith(rootDir)) {
        return null;
    }

    return fullPath;
}

function createServer() {
    return http.createServer((request, response) => {
        const url = new URL(request.url, 'http://localhost');

        if (url.pathname === '/runtime-config.js') {
            send(response, 200, buildRuntimeConfigScript(), 'application/javascript; charset=utf-8');
            return;
        }

        const filePath = resolveRequestPath(request.url);
        if (!filePath) {
            send(response, 403, 'Forbidden', 'text/plain; charset=utf-8');
            return;
        }

        fs.stat(filePath, (statError, stats) => {
            if (statError) {
                send(response, 404, 'Not found', 'text/plain; charset=utf-8');
                return;
            }

            const finalPath = stats.isDirectory() ? path.join(filePath, defaultFile) : filePath;
            const extension = path.extname(finalPath).toLowerCase();
            const contentType = mimeTypes[extension] || 'application/octet-stream';

            fs.readFile(finalPath, (readError, file) => {
                if (readError) {
                    send(response, 500, 'Failed to read file', 'text/plain; charset=utf-8');
                    return;
                }
                send(response, 200, file, contentType);
            });
        });
    });
}

function startServer(options = {}) {
    const { requestedPort = port, host = '127.0.0.1' } = options;
    const server = createServer();

    return new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(requestedPort, host, () => {
            const address = server.address();
            resolve({
                server,
                host,
                port: typeof address === 'object' && address ? address.port : requestedPort
            });
        });
    });
}

if (require.main === module) {
    startServer()
        .then(({ host, port: activePort }) => {
            console.log(`Earth Digital Twin is available at http://${host}:${activePort}`);
        })
        .catch((error) => {
            console.error(`Failed to start Earth Digital Twin server: ${error && error.message ? error.message : error}`);
            process.exitCode = 1;
        });
}

module.exports = {
    createServer,
    startServer
};