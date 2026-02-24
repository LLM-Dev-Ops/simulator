'use strict';

const crypto = require('crypto');
const https = require('https');
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
// Keyword dictionaries for rule-based analysis
// ============================================================================

const COMPLEXITY_KEYWORDS = [
  'migration', 'rewrite', 'overhaul', 'redesign', 'distributed', 'real-time',
  'legacy', 'monolith', 'refactor', 'replatform', 'decompose',
];
const RISK_KEYWORDS = [
  'security', 'authentication', 'encryption', 'compliance', 'gdpr', 'hipaa',
  'pci', 'vulnerability', 'breach', 'downtime', 'outage', 'data-loss',
];
const SCALE_KEYWORDS = [
  'scale', 'scaling', 'high-availability', 'load-balancing', 'multi-region',
  'concurrent', 'throughput', 'capacity', 'horizontal', 'vertical', 'shard',
];
const TECH_KEYWORDS = [
  'kubernetes', 'microservices', 'serverless', 'machine-learning', 'blockchain',
  'ai', 'ml', 'deep-learning', 'gpu', 'container', 'kafka', 'grpc',
];
const POSITIVE_KEYWORDS = [
  'cache', 'caching', 'optimize', 'index', 'cdn', 'compress', 'batch', 'async',
  'parallel', 'incremental', 'automate', 'simplify', 'streamline',
];

function matchKeywords(text, keywords) {
  const lower = text.toLowerCase();
  return keywords.filter((k) => lower.includes(k));
}

// ============================================================================
// Claude API Integration
// ============================================================================

function callClaude(apiKey, systemPrompt, userPrompt) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1024,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
    });

    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
    };

    const timer = setTimeout(() => {
      req.destroy(new Error('Claude API timeout'));
    }, MAX_LATENCY_MS - 500);

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        clearTimeout(timer);
        try {
          const parsed = JSON.parse(data);
          if (res.statusCode !== 200) {
            reject(new Error(`Claude API ${res.statusCode}: ${parsed.error?.message || data}`));
            return;
          }
          resolve(parsed.content?.[0]?.text || '');
        } catch (err) {
          reject(new Error(`Failed to parse Claude response: ${err.message}`));
        }
      });
    });

    req.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
    req.write(body);
    req.end();
  });
}

function parseJsonFromText(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) {
    try { return JSON.parse(fenced[1].trim()); } catch { /* fall through */ }
  }
  const braced = text.match(/\{[\s\S]*\}/);
  if (braced) {
    try { return JSON.parse(braced[0]); } catch { /* fall through */ }
  }
  return null;
}

// ============================================================================
// Claude-Powered Analysis
// ============================================================================

async function analyzeScenarioWithClaude(scenarioId, parameters, expectedOutcomes, apiKey) {
  const systemPrompt = `You are a platform simulation analyst. Analyze the given scenario and return ONLY valid JSON (no markdown fences) with these fields:
- outcome (string): concise analysis summary
- success_probability (number 0.0-1.0): realistic probability based on the scenario
- risk_factors (string[]): specific identified risks
- recommendations (string[]): actionable recommendations (signals, not decisions)
- evidence_items (string[]): evidence supporting your analysis
- timeline_estimate (string): estimated duration like "2-4 weeks"`;

  const userPrompt = `Scenario ID: ${scenarioId}
Parameters: ${JSON.stringify(parameters)}
Expected outcomes: ${expectedOutcomes.length > 0 ? expectedOutcomes.join('; ') : 'none specified'}

Provide a realistic platform-level what-if analysis.`;

  const text = await callClaude(apiKey, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);
  if (!parsed || typeof parsed.success_probability !== 'number') {
    throw new Error('Claude returned unparseable scenario analysis');
  }
  return {
    outcome: String(parsed.outcome || ''),
    success_probability: Math.max(0, Math.min(1, parsed.success_probability)),
    risk_factors: Array.isArray(parsed.risk_factors) ? parsed.risk_factors.map(String) : [],
    recommendations: Array.isArray(parsed.recommendations) ? parsed.recommendations.map(String) : [],
    evidence_items: Array.isArray(parsed.evidence_items) ? parsed.evidence_items.map(String) : [],
    timeline_estimate: String(parsed.timeline_estimate || 'unknown'),
  };
}

async function analyzeWhatIfWithClaude(input, domain, constraints, evidence, apiKey) {
  const systemPrompt = `You are a platform what-if analyst. Analyze the query and return ONLY valid JSON (no markdown fences) with these fields:
- hypothesis (string): primary hypothesis statement
- confidence (number 0.0-1.0): confidence in the hypothesis
- alternatives (string[]): exactly 3 entries for best-case, expected-case, worst-case scenarios, each starting with "Best case:", "Expected case:", or "Worst case:" and including a probability estimate
- reasoning (string): detailed reasoning with quantified impact metrics`;

  const userPrompt = `Input: ${input}
Domain: ${domain}
Constraints: ${constraints.length > 0 ? constraints.join('; ') : 'none'}
Evidence: ${evidence.length > 0 ? evidence.join('; ') : 'none'}

Provide multi-scenario impact analysis with quantified metrics.`;

  const text = await callClaude(apiKey, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);
  if (!parsed || typeof parsed.confidence !== 'number') {
    throw new Error('Claude returned unparseable what-if analysis');
  }
  return {
    hypothesis: String(parsed.hypothesis || ''),
    confidence: Math.max(0, Math.min(1, parsed.confidence)),
    evidence,
    alternatives: Array.isArray(parsed.alternatives) ? parsed.alternatives.map(String) : [],
    reasoning: String(parsed.reasoning || ''),
  };
}

// ============================================================================
// Rule-Based Analysis (fallback when no API key)
// ============================================================================

function analyzeScenarioRuleBased(scenarioId, parameters, expectedOutcomes) {
  const description = parameters.description || JSON.stringify(parameters);

  const complexityHits = matchKeywords(description, COMPLEXITY_KEYWORDS);
  const riskHits = matchKeywords(description, RISK_KEYWORDS);
  const scaleHits = matchKeywords(description, SCALE_KEYWORDS);
  const techHits = matchKeywords(description, TECH_KEYWORDS);
  const positiveHits = matchKeywords(description, POSITIVE_KEYWORDS);

  // Weighted success probability
  let probability = 0.78;
  probability -= complexityHits.length * 0.07;
  probability -= riskHits.length * 0.05;
  probability -= scaleHits.length * 0.06;
  probability -= techHits.length * 0.03;
  probability += positiveHits.length * 0.04;
  if (expectedOutcomes.length > 0) {
    probability += Math.min(expectedOutcomes.length * 0.02, 0.08);
  }
  const successProbability = Math.round(Math.max(0.12, Math.min(0.95, probability)) * 100) / 100;

  // Risk factors from detected patterns
  const riskFactors = [];
  if (complexityHits.length > 0) riskFactors.push(`implementation_complexity: ${complexityHits.join(', ')}`);
  if (riskHits.length > 0) riskFactors.push(`security_compliance: ${riskHits.join(', ')}`);
  if (scaleHits.length > 0) riskFactors.push(`scalability_concerns: ${scaleHits.join(', ')}`);
  if (techHits.length > 0) riskFactors.push(`technology_risk: ${techHits.join(', ')}`);
  if (riskFactors.length === 0) riskFactors.push('standard_operational_risk');

  // Evidence items from analysis
  const evidenceItems = [];
  if (complexityHits.length > 0) evidenceItems.push(`Detected ${complexityHits.length} complexity indicator(s): ${complexityHits.join(', ')}`);
  if (riskHits.length > 0) evidenceItems.push(`Found ${riskHits.length} risk-related factor(s): ${riskHits.join(', ')}`);
  if (scaleHits.length > 0) evidenceItems.push(`Identified ${scaleHits.length} scalability consideration(s): ${scaleHits.join(', ')}`);
  if (positiveHits.length > 0) evidenceItems.push(`Found ${positiveHits.length} optimization indicator(s): ${positiveHits.join(', ')}`);
  if (expectedOutcomes.length > 0) evidenceItems.push(`${expectedOutcomes.length} expected outcome(s) provided for validation`);
  const paramKeys = Object.keys(parameters).filter((k) => k !== 'description');
  if (paramKeys.length > 0) evidenceItems.push(`${paramKeys.length} simulation parameter(s): ${paramKeys.join(', ')}`);
  if (evidenceItems.length === 0) evidenceItems.push('Baseline analysis with no specific indicators detected');

  // Timeline estimate
  const totalFactors = complexityHits.length + riskHits.length + scaleHits.length + techHits.length;
  let timelineEstimate;
  if (totalFactors === 0) timelineEstimate = '1-2 weeks';
  else if (totalFactors <= 2) timelineEstimate = '2-4 weeks';
  else if (totalFactors <= 4) timelineEstimate = '4-8 weeks';
  else timelineEstimate = '8-16 weeks';

  // Recommendations
  const recommendations = [];
  if (complexityHits.length > 1) recommendations.push('Break implementation into phased milestones to manage complexity');
  if (riskHits.length > 0) recommendations.push('Conduct security and compliance review before deployment');
  if (scaleHits.length > 0) recommendations.push('Validate scalability assumptions with load testing');
  if (techHits.length > 0) recommendations.push('Prototype technology integration to surface unknowns early');
  if (positiveHits.length > 0) recommendations.push(`Prioritize identified optimizations: ${positiveHits.join(', ')}`);
  if (recommendations.length === 0) recommendations.push('Proceed with standard monitoring and incremental rollout');

  // Outcome description
  const outlook = successProbability >= 0.7 ? 'favorable' : successProbability >= 0.4 ? 'moderate' : 'challenging';
  const outcomeLines = [`Scenario '${scenarioId}' analysis: ${outlook} outlook (p=${successProbability}).`];
  outcomeLines.push(`Estimated timeline: ${timelineEstimate}.`);
  if (totalFactors > 0) {
    const topFactors = [...complexityHits, ...riskHits, ...scaleHits, ...techHits].slice(0, 4);
    outcomeLines.push(`Key factors: ${topFactors.join(', ')}.`);
  }

  return {
    outcome: outcomeLines.join(' '),
    success_probability: successProbability,
    risk_factors: riskFactors,
    recommendations,
    evidence_items: evidenceItems,
    timeline_estimate: timelineEstimate,
  };
}

function analyzeWhatIfRuleBased(input, domain, constraints, evidence) {
  const allText = [input, domain, ...constraints, ...evidence].join(' ');

  const complexityHits = matchKeywords(allText, COMPLEXITY_KEYWORDS);
  const riskHits = matchKeywords(allText, RISK_KEYWORDS);
  const scaleHits = matchKeywords(allText, SCALE_KEYWORDS);
  const techHits = matchKeywords(allText, TECH_KEYWORDS);
  const positiveHits = matchKeywords(allText, POSITIVE_KEYWORDS);

  const totalRisk = complexityHits.length + riskHits.length + scaleHits.length;
  const totalPositive = positiveHits.length + techHits.length;

  // Confidence from evidence quality and input specificity
  let confidence = 0.45;
  confidence += Math.min(evidence.length * 0.08, 0.3);
  confidence += Math.min(constraints.length * 0.03, 0.1);
  if (input.length > 200) confidence += 0.05;
  confidence = Math.round(Math.max(0.2, Math.min(0.95, confidence)) * 100) / 100;

  // Hypothesis based on balance of factors
  let hypothesis;
  if (totalPositive > totalRisk) {
    hypothesis = `The proposed change in '${domain}' is likely to yield net positive outcomes. ` +
      `Analysis indicates ${totalPositive} favorable factor(s) against ${totalRisk} risk factor(s), ` +
      `suggesting the change should proceed with standard monitoring.`;
  } else if (totalRisk > totalPositive) {
    hypothesis = `The proposed change in '${domain}' carries elevated risk. ` +
      `Analysis identifies ${totalRisk} risk factor(s) outweighing ${totalPositive} positive indicator(s), ` +
      `suggesting phased implementation with rollback capability.`;
  } else {
    hypothesis = `The proposed change in '${domain}' shows balanced risk-reward profile. ` +
      `Analysis found ${totalRisk} risk and ${totalPositive} opportunity factor(s), ` +
      `suggesting a controlled experiment approach to validate outcomes.`;
  }

  // Best/Expected/Worst case alternatives
  const bestProb = Math.round(Math.min(0.95, 0.60 + totalPositive * 0.05) * 100) / 100;
  const expectedProb = Math.round(Math.max(0.25, 0.50 + (totalPositive - totalRisk) * 0.05) * 100) / 100;
  const worstProb = Math.round(Math.max(0.05, 0.25 - totalRisk * 0.03) * 100) / 100;

  const bestDetail = positiveHits.length > 0
    ? `${positiveHits.join(', ')} deliver expected gains`
    : 'changes meet objectives with minimal friction';
  const worstDetail = riskHits.length > 0
    ? `${riskHits.join(', ')} introduce complications`
    : 'unforeseen integration issues delay benefits';

  const alternatives = [
    `Best case (p=${bestProb}): ${totalPositive > 0 ? 'significant' : 'moderate'} improvement in ${domain} — ${bestDetail}`,
    `Expected case (p=${expectedProb}): partial improvement in ${domain} — some benefits realized, ${totalRisk > 0 ? 'risk factors require active monitoring' : 'standard operational adjustments needed'}`,
    `Worst case (p=${worstProb}): ${totalRisk > 0 ? 'potential degradation' : 'minimal improvement'} — ${worstDetail}`,
  ];

  // Reasoning with quantified impact metrics
  const totalFactors = totalRisk + totalPositive;
  const riskScore = totalFactors > 0 ? Math.round(totalRisk / totalFactors * 100) : 50;
  const opportunityScore = totalFactors > 0 ? Math.round(totalPositive / totalFactors * 100) : 50;

  const reasoningParts = [];
  reasoningParts.push(`Domain: ${domain}. Input analysis identified ${totalRisk} risk factor(s) and ${totalPositive} positive indicator(s).`);
  if (constraints.length > 0) reasoningParts.push(`Operating under ${constraints.length} constraint(s): ${constraints.slice(0, 3).join('; ')}.`);
  if (evidence.length > 0) reasoningParts.push(`Supported by ${evidence.length} evidence item(s) which ${evidence.length > 2 ? 'strongly supports' : 'partially supports'} the hypothesis.`);
  reasoningParts.push(`Impact metrics — Risk score: ${riskScore}%, Opportunity score: ${opportunityScore}%, Net outlook: ${opportunityScore > riskScore ? 'positive' : opportunityScore === riskScore ? 'neutral' : 'cautious'}.`);

  return {
    hypothesis,
    confidence,
    evidence,
    alternatives,
    reasoning: reasoningParts.join(' '),
  };
}

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

async function handleWhatIf(req, res) {
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
  const apiKey = req.headers['x-anthropic-api-key'];

  let analysis;
  try {
    if (apiKey) {
      analysis = await analyzeWhatIfWithClaude(input, domain, constraints, evidence, apiKey);
    } else {
      analysis = analyzeWhatIfRuleBased(input, domain, constraints, evidence);
    }
  } catch (err) {
    // Fall back to rule-based on Claude failure
    analysis = analyzeWhatIfRuleBased(input, domain, constraints, evidence);
  }

  const tokensUsed = Math.min(Math.ceil(input.length / 4), MAX_TOKENS);

  const signal = {
    signal_id: `sig_${Date.now().toString(16)}`,
    signal_type: 'hypothesis_signal',
    timestamp_ms: Date.now(),
    source_agent: 'what-if',
    correlation_id: correlationId,
    payload: {
      type: 'Hypothesis',
      data: {
        hypothesis: analysis.hypothesis,
        confidence: analysis.confidence,
        evidence: analysis.evidence,
        alternatives: analysis.alternatives,
        reasoning: analysis.reasoning,
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

async function handleScenario(req, res) {
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
  const apiKey = req.headers['x-anthropic-api-key'];

  let analysis;
  try {
    if (apiKey) {
      analysis = await analyzeScenarioWithClaude(scenario_id, parameters, expected_outcomes, apiKey);
    } else {
      analysis = analyzeScenarioRuleBased(scenario_id, parameters, expected_outcomes);
    }
  } catch (err) {
    // Fall back to rule-based on Claude failure
    analysis = analyzeScenarioRuleBased(scenario_id, parameters, expected_outcomes);
  }

  const tokensUsed = Math.min(
    Math.ceil(JSON.stringify(parameters).length / 4),
    MAX_TOKENS
  );

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
        outcome: analysis.outcome,
        success_probability: analysis.success_probability,
        risk_factors: analysis.risk_factors,
        recommendations: analysis.recommendations,
        evidence_items: analysis.evidence_items,
        timeline_estimate: analysis.timeline_estimate,
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
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Correlation-ID', 'X-Anthropic-Api-Key'],
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
