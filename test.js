'use strict';

const http = require('http');
const { handler } = require('./index');

const PORT = 0; // Random available port
let server;
let baseUrl;
let passed = 0;
let failed = 0;

function request(method, path, body) {
  return new Promise((resolve, reject) => {
    const opts = {
      hostname: '127.0.0.1',
      port: new URL(baseUrl).port,
      path,
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    const req = http.request(opts, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        resolve({ status: res.statusCode, body: JSON.parse(data) });
      });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function assert(condition, label) {
  if (condition) {
    console.log(`  PASS: ${label}`);
    passed++;
  } else {
    console.error(`  FAIL: ${label}`);
    failed++;
  }
}

async function runTests() {
  // ---- Health ----
  console.log('\n[1] GET /health');
  const h = await request('GET', '/health');
  assert(h.status === 200, 'status 200');
  assert(h.body.status === 'healthy', 'status healthy');
  assert(h.body.service === 'simulator-agents', 'service name');
  assert(Array.isArray(h.body.agents) && h.body.agents.includes('what-if') && h.body.agents.includes('scenario'), 'agents list');
  assert(h.body.execution_metadata && h.body.execution_metadata.service === 'simulator-agents', 'execution_metadata present');
  assert(Array.isArray(h.body.layers_executed), 'layers_executed present');

  // ---- What-If Agent ----
  console.log('\n[2] POST /v1/simulator/what-if (valid)');
  const w = await request('POST', '/v1/simulator/what-if', {
    input: 'What if we increase cache TTL to 600s?',
    domain: 'performance',
    constraints: ['budget < 100ms'],
    evidence: ['Previous TTL of 300s showed 80% hit rate'],
  });
  assert(w.status === 200, 'status 200');
  assert(w.body.signal.signal_type === 'hypothesis_signal', 'signal_type');
  assert(w.body.signal.source_agent === 'what-if', 'source_agent');
  assert(w.body.signal.payload.type === 'Hypothesis', 'payload type');
  assert(typeof w.body.signal.payload.data.confidence === 'number', 'confidence is number');
  assert(w.body.signal.tokens_used <= 2500, 'tokens within budget');
  assert(w.body.execution_metadata.service === 'simulator-agents', 'execution_metadata.service');
  assert(w.body.execution_metadata.trace_id, 'execution_metadata.trace_id');
  assert(w.body.execution_metadata.execution_id, 'execution_metadata.execution_id');
  assert(w.body.layers_executed.length === 2, 'layers_executed length');
  assert(w.body.layers_executed[0].layer === 'AGENT_ROUTING', 'layer[0] AGENT_ROUTING');
  assert(w.body.layers_executed[1].layer === 'SIMULATOR_WHAT_IF', 'layer[1] SIMULATOR_WHAT_IF');
  assert(typeof w.body.layers_executed[1].duration_ms === 'number', 'duration_ms present');

  console.log('\n[3] POST /v1/simulator/what-if (invalid - missing input)');
  const wi = await request('POST', '/v1/simulator/what-if', { domain: 'test' });
  assert(wi.status === 400, 'status 400');
  assert(wi.body.error.code === 'INVALID_REQUEST', 'error code');
  assert(wi.body.execution_metadata, 'execution_metadata on error');

  // ---- Scenario Agent ----
  console.log('\n[4] POST /v1/simulator/scenario (valid)');
  const s = await request('POST', '/v1/simulator/scenario', {
    scenario_id: 'scenario-perf-001',
    parameters: { cache_ttl: 600, max_connections: 50 },
    expected_outcomes: ['Reduced latency', 'Higher throughput'],
  });
  assert(s.status === 200, 'status 200');
  assert(s.body.signal.signal_type === 'simulation_outcome_signal', 'signal_type');
  assert(s.body.signal.source_agent === 'scenario', 'source_agent');
  assert(s.body.signal.payload.type === 'SimulationOutcome', 'payload type');
  assert(s.body.signal.payload.data.scenario_id === 'scenario-perf-001', 'scenario_id matches');
  assert(typeof s.body.signal.payload.data.success_probability === 'number', 'success_probability');
  assert(Array.isArray(s.body.signal.payload.data.risk_factors), 'risk_factors');
  assert(Array.isArray(s.body.signal.payload.data.recommendations), 'recommendations');
  assert(s.body.execution_metadata.service === 'simulator-agents', 'execution_metadata.service');
  assert(s.body.layers_executed[1].layer === 'SIMULATOR_SCENARIO', 'layer SIMULATOR_SCENARIO');

  console.log('\n[5] POST /v1/simulator/scenario (invalid - missing scenario_id)');
  const si = await request('POST', '/v1/simulator/scenario', { parameters: {} });
  assert(si.status === 400, 'status 400');
  assert(si.body.error.code === 'INVALID_REQUEST', 'error code');

  // ---- 404 ----
  console.log('\n[6] GET /nonexistent');
  const n = await request('GET', '/nonexistent');
  assert(n.status === 404, 'status 404');
  assert(n.body.execution_metadata, 'execution_metadata on 404');

  // ---- Correlation ID passthrough ----
  console.log('\n[7] X-Correlation-ID passthrough');
  const opts = {
    hostname: '127.0.0.1',
    port: new URL(baseUrl).port,
    path: '/v1/simulator/what-if',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Correlation-ID': 'test-trace-abc-123',
    },
  };
  const cid = await new Promise((resolve, reject) => {
    const req = http.request(opts, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(data) }));
    });
    req.on('error', reject);
    req.write(JSON.stringify({ input: 'test', domain: 'test' }));
    req.end();
  });
  assert(cid.body.execution_metadata.trace_id === 'test-trace-abc-123', 'correlation ID passthrough');
}

// Boot server and run
server = handler.listen(PORT, '127.0.0.1', async () => {
  const addr = server.address();
  baseUrl = `http://127.0.0.1:${addr.port}`;
  console.log(`Test server on ${baseUrl}`);

  try {
    await runTests();
  } catch (err) {
    console.error('Test error:', err);
    failed++;
  }

  console.log(`\n========================================`);
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log(`========================================\n`);

  server.close(() => process.exit(failed > 0 ? 1 : 0));
});
