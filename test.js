'use strict';

const http = require('http');
const { handler } = require('./index');

const PORT = 0; // Random available port
let server;
let baseUrl;
let passed = 0;
let failed = 0;

function request(method, path, body, extraHeaders) {
  return new Promise((resolve, reject) => {
    const opts = {
      hostname: '127.0.0.1',
      port: new URL(baseUrl).port,
      path,
      method,
      headers: { 'Content-Type': 'application/json', ...extraHeaders },
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

  // Dynamic analysis assertions
  const wd = w.body.signal.payload.data;
  assert(wd.hypothesis.length > 20, 'hypothesis is substantive (not a template)');
  assert(wd.hypothesis.includes('performance'), 'hypothesis references the domain');
  assert(Array.isArray(wd.alternatives) && wd.alternatives.length === 3, 'three outcome scenarios (best/expected/worst)');
  assert(wd.alternatives[0].toLowerCase().includes('best'), 'first alternative is best case');
  assert(wd.alternatives[1].toLowerCase().includes('expected'), 'second alternative is expected case');
  assert(wd.alternatives[2].toLowerCase().includes('worst'), 'third alternative is worst case');
  assert(wd.reasoning.includes('impact metrics') || wd.reasoning.includes('Impact metrics'), 'reasoning includes quantified impact metrics');
  assert(wd.reasoning.includes('performance'), 'reasoning references domain');

  console.log('\n[3] POST /v1/simulator/what-if (invalid - missing input)');
  const wi = await request('POST', '/v1/simulator/what-if', { domain: 'test' });
  assert(wi.status === 400, 'status 400');
  assert(wi.body.error.code === 'INVALID_REQUEST', 'error code');
  assert(wi.body.execution_metadata, 'execution_metadata on error');

  // ---- What-If with risk keywords ----
  console.log('\n[4] POST /v1/simulator/what-if (risk-heavy input)');
  const wr = await request('POST', '/v1/simulator/what-if', {
    input: 'What if we migrate the legacy authentication system to a distributed microservices architecture with encryption?',
    domain: 'security',
    constraints: [],
    evidence: [],
  });
  assert(wr.status === 200, 'status 200');
  const wrd = wr.body.signal.payload.data;
  assert(wrd.hypothesis.includes('risk'), 'risk-heavy input produces risk-aware hypothesis');
  assert(wrd.confidence > 0 && wrd.confidence < 1, 'confidence is bounded (0,1)');

  // ---- Scenario Agent ----
  console.log('\n[5] POST /v1/simulator/scenario (valid)');
  const s = await request('POST', '/v1/simulator/scenario', {
    scenario_id: 'sim-perf-001',
    parameters: { description: 'Migrate legacy monolith to Kubernetes microservices with caching layer' },
    expected_outcomes: ['Reduced latency', 'Higher throughput'],
  });
  assert(s.status === 200, 'status 200');
  assert(s.body.signal.signal_type === 'simulation_outcome_signal', 'signal_type');
  assert(s.body.signal.source_agent === 'scenario', 'source_agent');
  assert(s.body.signal.payload.type === 'SimulationOutcome', 'payload type');
  assert(s.body.signal.payload.data.scenario_id === 'sim-perf-001', 'scenario_id matches');
  assert(typeof s.body.signal.payload.data.success_probability === 'number', 'success_probability');
  assert(Array.isArray(s.body.signal.payload.data.risk_factors), 'risk_factors');
  assert(Array.isArray(s.body.signal.payload.data.recommendations), 'recommendations');
  assert(s.body.execution_metadata.service === 'simulator-agents', 'execution_metadata.service');
  assert(s.body.layers_executed[1].layer === 'SIMULATOR_SCENARIO', 'layer SIMULATOR_SCENARIO');

  // Dynamic analysis assertions for scenario
  const sd = s.body.signal.payload.data;
  assert(sd.success_probability !== 0.75, 'success_probability is NOT the static 0.75 placeholder');
  assert(sd.success_probability > 0 && sd.success_probability < 1, 'success_probability is bounded (0,1)');
  assert(sd.risk_factors.length > 0, 'risk_factors are populated');
  assert(sd.risk_factors[0] !== 'latency_variance', 'risk_factors are NOT hardcoded placeholders');
  assert(sd.risk_factors.some((r) => r.includes('complexity') || r.includes('scalability') || r.includes('technology')), 'risk_factors reflect input content');
  assert(Array.isArray(sd.evidence_items), 'evidence_items present');
  assert(sd.evidence_items.length > 0, 'evidence_items are populated');
  assert(typeof sd.timeline_estimate === 'string' && sd.timeline_estimate.length > 0, 'timeline_estimate present');
  assert(sd.outcome.includes('sim-perf-001'), 'outcome references scenario_id');
  assert(sd.recommendations.length > 0, 'recommendations populated');
  assert(sd.recommendations[0] !== 'Consider caching frequent queries', 'recommendations are NOT hardcoded');

  // ---- Scenario with simple parameters (no risk keywords) ----
  console.log('\n[6] POST /v1/simulator/scenario (simple input)');
  const ss = await request('POST', '/v1/simulator/scenario', {
    scenario_id: 'sim-simple-001',
    parameters: { description: 'Add a new status field to the user profile' },
  });
  assert(ss.status === 200, 'status 200');
  const ssd = ss.body.signal.payload.data;
  assert(ssd.success_probability > sd.success_probability, 'simple scenario has higher success probability than complex one');
  assert(ssd.risk_factors.includes('standard_operational_risk'), 'simple scenario has standard risk factor');

  console.log('\n[7] POST /v1/simulator/scenario (invalid - missing scenario_id)');
  const si = await request('POST', '/v1/simulator/scenario', { parameters: {} });
  assert(si.status === 400, 'status 400');
  assert(si.body.error.code === 'INVALID_REQUEST', 'error code');

  // ---- 404 ----
  console.log('\n[8] GET /nonexistent');
  const n = await request('GET', '/nonexistent');
  assert(n.status === 404, 'status 404');
  assert(n.body.execution_metadata, 'execution_metadata on 404');

  // ---- Correlation ID passthrough ----
  console.log('\n[9] X-Correlation-ID passthrough');
  const cid = await request('POST', '/v1/simulator/what-if',
    { input: 'test', domain: 'test' },
    { 'X-Correlation-ID': 'test-trace-abc-123' }
  );
  assert(cid.body.execution_metadata.trace_id === 'test-trace-abc-123', 'correlation ID passthrough');

  // ---- What-If with lots of evidence boosts confidence ----
  console.log('\n[10] POST /v1/simulator/what-if (evidence boosts confidence)');
  const we = await request('POST', '/v1/simulator/what-if', {
    input: 'What if we enable batch processing?',
    domain: 'throughput',
    constraints: [],
    evidence: ['Batch processing reduced latency by 40%', 'Throughput increased 3x in staging', 'Memory usage stayed flat', 'No error rate increase'],
  });
  assert(we.status === 200, 'status 200');
  const wed = we.body.signal.payload.data;
  assert(wed.confidence > w.body.signal.payload.data.confidence, 'more evidence produces higher confidence');
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
