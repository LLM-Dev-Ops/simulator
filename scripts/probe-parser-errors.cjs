'use strict';
const http = require('node:http');
const path = require('node:path');
const { once } = require('node:events');
const root = path.resolve(process.argv[2] || '.');
process.env.CORS_ALLOWED_ORIGINS = 'https://console.example.com';
const { handler } = require(path.join(root, 'index.js'));
const marker = 'PRIVATE_BODY_SENTINEL';
const whatif = '/v1/simulator/what-if';
const scenario = '/v1/simulator/scenario';
async function main() {
  const server = handler.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const request = (route, body, headers = {}, method = 'POST') => new Promise((resolve, reject) => {
    const req = http.request({ hostname: '127.0.0.1', port: server.address().port,
      path: route, method, headers: { 'Content-Type': 'application/json', ...headers } }, (res) => {
      let text = '';
      res.on('data', (chunk) => { text += chunk; });
      res.on('end', () => { let json; try { json = JSON.parse(text); } catch {}
        resolve({ status: res.statusCode, headers: res.headers, text, json }); });
    });
    req.setTimeout(5000, () => req.destroy(new Error('probe socket timeout')));
    req.on('error', reject);
    req.end(body);
  });
  try {
    const definitions = [
      [whatif, '{"input":"' + marker, {}, 400, 'INVALID_JSON'],
      [scenario, '{"scenario_id":"' + marker, {}, 400, 'INVALID_JSON'],
      [whatif, JSON.stringify({ input: marker + 'x'.repeat(1048576) }), {}, 413, 'PAYLOAD_TOO_LARGE'],
      [whatif, '{}', { 'Content-Type': 'application/json; charset=iso-8859-1' }, 415, 'UNSUPPORTED_MEDIA_TYPE'],
      [scenario, '{}', { 'Content-Encoding': 'snappy' }, 415, 'UNSUPPORTED_MEDIA_TYPE'],
      [whatif, '42', {}, 400, 'INVALID_JSON'],
      [whatif, 'not-compressed', { 'Content-Encoding': 'gzip' }, 400, 'INVALID_REQUEST'],
      [scenario, 'not-compressed', { 'Content-Encoding': 'deflate' }, 400, 'INVALID_REQUEST'],
    ];
    const holdout = [];
    const statuses = [];
    for (const [route, body, headers, status, code] of definitions) {
      const r = await request(route, body, headers);
      statuses.push(r.status);
      holdout.push(r.status === status && r.headers['content-type']?.includes('application/json') &&
        r.headers['x-content-type-options'] === 'nosniff' && r.json?.error?.code === code &&
        typeof r.json?.execution_metadata?.trace_id === 'string' &&
        !r.text.includes(marker) && !r.text.includes('node_modules') && !r.text.includes('<!DOCTYPE'));
    }
    const health = await request('/health', undefined, {}, 'GET');
    const valid = await request(whatif, JSON.stringify({ input: 'cache optimization', domain: 'performance' }));
    const invalid = await request(whatif, '{}');
    const missing = await request('/not-a-route', undefined, {}, 'GET');
    const allowed = await request('/health', undefined, { Origin: 'https://console.example.com' }, 'GET');
    const denied = await request('/health', undefined, { Origin: 'https://untrusted.example.com' }, 'GET');
    const anchor = [
      health.status === 200 && health.json?.status === 'healthy',
      valid.status === 200 && valid.json?.signal?.signal_type === 'hypothesis_signal',
      invalid.status === 400 && invalid.json?.error?.code === 'INVALID_REQUEST',
      missing.status === 404 && missing.json?.error?.code === 'NOT_FOUND',
      allowed.headers['access-control-allow-origin'] === 'https://console.example.com' &&
        allowed.headers['access-control-allow-credentials'] === 'true',
      denied.status === 200 && !denied.headers['access-control-allow-origin'] &&
        !denied.headers['access-control-allow-credentials'],
    ];
    console.log(JSON.stringify({ holdout, anchor, statuses }));
    if (!process.argv.includes('--json') && ![...holdout, ...anchor].every(Boolean)) process.exitCode = 1;
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}
main().catch((err) => { console.error(err); process.exitCode = 1; });
