'use strict';

const crypto = require('crypto');
const express = require('express');
const cors = require('cors');

const { validateWhatIfRequest, validateScenarioRequest } = require('./schemas/validate');

// ============================================================================
// Constants
// ============================================================================

const SERVICE_NAME = 'simulator-agents';
const AGENTS = ['what-if', 'scenario'];
const MAX_TOKENS = 2500;
const MAX_LATENCY_MS = 5000;

// ============================================================================
// Execution Metadata Builder
// ============================================================================

function buildExecutionMetadata(req) {
  return {
    trace_id: req.headers['x-correlation-id'] || crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    service: SERVICE_NAME,
    execution_id: crypto.randomUUID(),
  };
}

function buildLayersExecuted(agentName, durationMs) {
  return [
    { layer: 'AGENT_ROUTING', status: 'completed' },
    { layer: `SIMULATOR_${agentName}`, status: 'completed', duration_ms: durationMs },
  ];
}

// ============================================================================
// What-If Simulation Agent Handler
// ============================================================================

function handleWhatIf(req, res) {
  const start = Date.now();
  const executionMetadata = buildExecutionMetadata(req);

  const validation = validateWhatIfRequest(req.body);
  if (!validation.valid) {
    const durationMs = Date.now() - start;
    return res.status(400).json({
      error: {
        code: 'INVALID_REQUEST',
        message: 'Request validation failed',
        details: validation.errors,
      },
      execution_metadata: executionMetadata,
      layers_executed: [
        { layer: 'AGENT_ROUTING', status: 'completed' },
        { layer: 'SIMULATOR_WHAT_IF', status: 'failed', duration_ms: durationMs },
      ],
    });
  }

  const { input, domain, constraints = [], evidence = [] } = req.body;
  const correlationId = executionMetadata.trace_id;

  // Emit hypothesis signal (mirrors IntelligenceConsumer.emit_hypothesis)
  const tokensUsed = Math.min(Math.ceil(input.length / 4), MAX_TOKENS);
  const confidence = evidence.length > 0 ? Math.min(0.5 + evidence.length * 0.1, 0.95) : 0.5;

  const signal = {
    signal_id: `sig_${Date.now().toString(16)}`,
    signal_type: 'hypothesis_signal',
    timestamp_ms: Date.now(),
    source_agent: 'what-if',
    correlation_id: correlationId,
    payload: {
      type: 'Hypothesis',
      data: {
        hypothesis: `Based on ${evidence.length} evidence items from domain '${domain}'`,
        confidence,
        evidence,
        alternatives: [],
        reasoning: `What-if analysis for domain '${domain}' with ${constraints.length} constraints`,
      },
    },
    latency_ms: Date.now() - start,
    tokens_used: tokensUsed,
  };

  const durationMs = Date.now() - start;

  return res.status(200).json({
    signal,
    execution_metadata: executionMetadata,
    layers_executed: buildLayersExecuted('WHAT_IF', durationMs),
  });
}

// ============================================================================
// Scenario Generator Agent Handler
// ============================================================================

function handleScenario(req, res) {
  const start = Date.now();
  const executionMetadata = buildExecutionMetadata(req);

  const validation = validateScenarioRequest(req.body);
  if (!validation.valid) {
    const durationMs = Date.now() - start;
    return res.status(400).json({
      error: {
        code: 'INVALID_REQUEST',
        message: 'Request validation failed',
        details: validation.errors,
      },
      execution_metadata: executionMetadata,
      layers_executed: [
        { layer: 'AGENT_ROUTING', status: 'completed' },
        { layer: 'SIMULATOR_SCENARIO', status: 'failed', duration_ms: durationMs },
      ],
    });
  }

  const { scenario_id, parameters, expected_outcomes = [] } = req.body;
  const correlationId = executionMetadata.trace_id;

  // Emit simulation outcome signal (mirrors IntelligenceConsumer.emit_simulation_outcome)
  const tokensUsed = Math.min(
    Math.ceil(JSON.stringify(parameters).length / 4),
    MAX_TOKENS
  );
  const successProbability = expected_outcomes.length > 0
    ? Math.min(0.5 + expected_outcomes.length * 0.05, 0.95)
    : 0.75;

  const signal = {
    signal_id: `sig_${Date.now().toString(16)}`,
    signal_type: 'simulation_outcome_signal',
    timestamp_ms: Date.now(),
    source_agent: 'scenario',
    correlation_id: correlationId,
    payload: {
      type: 'SimulationOutcome',
      data: {
        scenario_id,
        parameters,
        outcome: `Simulation completed for scenario '${scenario_id}'`,
        success_probability: successProbability,
        risk_factors: ['latency_variance', 'token_consumption'],
        recommendations: [
          'Consider caching frequent queries',
          'Monitor token budget utilization',
        ],
      },
    },
    latency_ms: Date.now() - start,
    tokens_used: tokensUsed,
  };

  const durationMs = Date.now() - start;

  return res.status(200).json({
    signal,
    execution_metadata: executionMetadata,
    layers_executed: buildLayersExecuted('SCENARIO', durationMs),
  });
}

// ============================================================================
// Health Endpoint
// ============================================================================

function handleHealth(req, res) {
  const executionMetadata = buildExecutionMetadata(req);

  return res.status(200).json({
    status: 'healthy',
    service: SERVICE_NAME,
    agents: AGENTS,
    version: process.env.K_REVISION || '1.0.0',
    uptime_seconds: Math.floor(process.uptime()),
    execution_metadata: executionMetadata,
    layers_executed: [
      { layer: 'AGENT_ROUTING', status: 'completed' },
    ],
  });
}

// ============================================================================
// Express App
// ============================================================================

const app = express();

app.use(cors({
  origin: true,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Correlation-ID'],
  exposedHeaders: ['X-Correlation-ID'],
  credentials: true,
  maxAge: 3600,
}));

app.use(express.json({ limit: '1mb' }));

// Routes
app.post('/v1/simulator/what-if', handleWhatIf);
app.post('/v1/simulator/scenario', handleScenario);
app.get('/health', handleHealth);

// 404 fallback
app.use((req, res) => {
  const executionMetadata = buildExecutionMetadata(req);
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: `Route ${req.method} ${req.path} not found`,
      available_routes: [
        'POST /v1/simulator/what-if',
        'POST /v1/simulator/scenario',
        'GET /health',
      ],
    },
    execution_metadata: executionMetadata,
    layers_executed: [
      { layer: 'AGENT_ROUTING', status: 'completed' },
    ],
  });
});

// Cloud Function entry point
exports.handler = app;
