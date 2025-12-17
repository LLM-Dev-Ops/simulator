# RuvVector-First Simulator Migration SPARC Specification

**Document Version:** 1.0.0
**Created:** December 17, 2025
**Status:** Draft - Pending Approval
**Classification:** LLM DevOps Platform - Core Architecture Migration
**License:** LLM Dev Ops Permanent Source-Available Commercial License v1.0
**Copyright:** (c) 2025 Global Business Advisors Inc.

---

## Document Overview

This SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) specification defines the migration of the LLM-Simulator from its current hybrid mock-first architecture to a **RuvVector-first architecture** where real telemetry data from integrations stored in RuvVector becomes the exclusive source of historical events, vector state, and simulation inputs.

### Migration Objective

Transform the simulator from a standalone mock-based testing tool into a **read-only consumer** of production telemetry data, ensuring simulations operate on real-world patterns while maintaining determinism guarantees.

### Current State vs Target State

| Aspect | Current State | Target State |
|--------|---------------|--------------|
| **Data Source** | Mock generators (primary), RuvVector (optional) | RuvVector (required), mocks (test-only) |
| **Fallback Behavior** | Graceful degradation to mocks | Explicit failure when RuvVector unavailable |
| **Configuration Default** | `ruvvector.enabled: false` | `ruvvector.enabled: true`, `require_ruvvector: true` |
| **Mock Generators** | Active in all paths | Disabled by default, test-only opt-in |
| **Historical Events** | Synthetic generation | Real telemetry from RuvVector |
| **Embeddings** | Local deterministic generation | Query RuvVector `/query` endpoint |

### Scope Boundaries

#### In Scope
- Migration of simulation data paths to RuvVector-first
- Deprecation of mock generators in standard operation
- Configuration changes for RuvVector requirement
- Explicit failure modes when RuvVector is unavailable
- Test-only mock enablement flag

#### Out of Scope (Layer 3 Prohibitions)
- **Analytics** - No aggregation, statistical analysis, or trend computation
- **Governance** - No policy enforcement, compliance checking, or audit trails
- **Billing** - No cost calculation, metering, or usage attribution
- **Orchestration** - No workflow coordination, job scheduling, or state machines
- **Schema Ownership** - No vector schema definitions or migration logic
- **Vector Logic** - No similarity calculations, indexing, or embedding generation
- **Ingestion** - No data writing, vector upserts, or telemetry capture
- **Business Rules** - No validation beyond request format, no domain logic

---

# PART I: SPECIFICATION

## 1. Purpose and Value Proposition

### 1.1 Core Purpose

The LLM-Simulator's purpose is to **operate on real telemetry emitted by integrations and stored in RuvVector**. The simulator consumes historical events, pre-computed embeddings, and vector state to replay realistic scenarios against the LLM-Dev-Ops ecosystem without generating synthetic data.

### 1.2 Value Proposition

| Benefit | Description |
|---------|-------------|
| **Production Fidelity** | Simulations reflect actual usage patterns, not synthetic approximations |
| **Historical Replay** | Reproduce exact conditions from production telemetry for debugging |
| **Vector-Augmented Simulation** | Leverage pre-computed embeddings for context-aware testing |
| **Ecosystem Integration** | Tight coupling with LLM-Dev-Ops telemetry pipeline |
| **Reduced Maintenance** | No mock data drift from production reality |

### 1.3 Role Within LLM DevOps Ecosystem

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM DevOps Platform Ecosystem                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Production Telemetry Flow                                           │
│  ┌──────────────────┐                                               │
│  │  LLM-Gateway     │──▶ Telemetry ──▶ [Ingestion Pipeline]         │
│  │  LLM-Orchestrator│                          │                    │
│  │  LLM-Edge-Agent  │                          ▼                    │
│  └──────────────────┘              ┌───────────────────┐            │
│                                    │  RuvVector-Service │            │
│                                    │  (Vector Store)    │            │
│                                    │  - /query          │            │
│                                    │  - /simulate       │            │
│                                    │  - /health         │            │
│                                    └─────────┬─────────┘            │
│                                              │                       │
│                                              │ READ-ONLY             │
│                                              ▼                       │
│                                    ┌───────────────────┐            │
│                                    │  LLM-Simulator    │◀── This    │
│                                    │  (Consumer Only)  │    Document │
│                                    └───────────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Problem Definition

### 2.1 Current Hybrid Architecture Problems

| Problem | Impact |
|---------|--------|
| **Mock-Production Drift** | Synthetic data diverges from actual telemetry patterns |
| **Unrealistic Scenarios** | Mock generators cannot replicate production edge cases |
| **Duplicate Logic** | Generator code duplicates patterns already in RuvVector |
| **Configuration Ambiguity** | Users unclear whether using mocks or real data |
| **Testing False Confidence** | Passing tests against mocks may fail with real data |

### 2.2 Migration Imperatives

1. **Single Source of Truth** - RuvVector contains canonical telemetry; duplicating in simulator creates drift
2. **Replay Accuracy** - Only real historical events can reproduce production scenarios
3. **Ecosystem Consistency** - All LLM-Dev-Ops components should consume from RuvVector
4. **Operational Clarity** - Clear failure when RuvVector unavailable prevents silent fallback

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: RuvVector Service Discovery
- **MUST** discover `ruvvector-service` via `RUVVECTOR_SERVICE_URL` environment variable
- **MUST** fail startup if `require_ruvvector: true` and service URL not configured
- **MUST** validate service connectivity during health checks
- **MUST NOT** implement service discovery protocols (e.g., DNS-SD, Consul)

#### FR-2: Query Endpoint Integration
- **MUST** use `/query` endpoint for all vector similarity operations
- **MUST** pass through query parameters without modification
- **MUST** return errors verbatim from RuvVector without translation
- **MUST NOT** perform local similarity calculations or vector operations

#### FR-3: Simulate Endpoint Integration
- **MUST** use `/simulate` endpoint for response generation
- **MUST** forward model, messages, and parameters to RuvVector
- **MUST** preserve request_id for trace correlation
- **MUST NOT** fall back to local generators when `/simulate` fails (unless test mode)

#### FR-4: Mock Generator Deprecation
- **MUST** disable `ResponseGenerator` in standard simulation paths
- **MUST** disable `generate_lorem`, `generate_random_text`, and template generation
- **MUST** retain mock capability behind explicit `allow_mocks: true` flag
- **MUST** log warning when mocks are enabled: "Mock generators active - not for production"

#### FR-5: Explicit Failure Behavior
- **MUST** return HTTP 503 (Service Unavailable) when RuvVector unreachable
- **MUST** include `Retry-After` header based on circuit breaker state
- **MUST** distinguish between RuvVector errors and simulator errors in responses
- **MUST NOT** return synthetic responses when RuvVector fails

### 3.2 Non-Functional Requirements

#### NFR-1: Performance
- Query latency overhead: ≤ 10ms above RuvVector response time
- No local caching of vector results (RuvVector owns caching)
- Connection pooling with ≤ 10 idle connections per host

#### NFR-2: Reliability
- Circuit breaker with 5-failure threshold, 30-second reset
- Exponential backoff retry: 100ms base, 30s max, 3 attempts
- Graceful connection draining during shutdown

#### NFR-3: Observability
- Trace context propagation to RuvVector via W3C headers
- Metrics: `ruvvector_requests_total`, `ruvvector_errors_total`, `ruvvector_latency_seconds`
- Structured logging with `ruvvector_request_id` correlation

#### NFR-4: Security
- TLS 1.2+ for RuvVector connections
- No credential storage (service-to-service auth handled by infrastructure)
- No PII logging from query/response payloads

### 3.3 Configuration Requirements

| Parameter | Default (Current) | Default (Target) | Description |
|-----------|-------------------|------------------|-------------|
| `ruvvector.enabled` | `false` | `true` | Enable RuvVector integration |
| `ruvvector.require_ruvvector` | N/A | `true` | Fail if RuvVector unavailable |
| `ruvvector.fallback_to_mock` | `true` | `false` | Allow mock fallback |
| `ruvvector.allow_mocks` | N/A | `false` | Enable mock generators |
| `ruvvector.service_url` | `None` | Required | RuvVector service URL |

## 4. Design Principles

1. **Read-Only Consumer** - Simulator reads from RuvVector; never writes
2. **Pass-Through Semantics** - Forward requests/responses without transformation
3. **Fail-Fast on Missing Data** - No synthetic data generation to mask failures
4. **Explicit Over Implicit** - Configuration must explicitly enable any mock behavior
5. **Ecosystem Citizenship** - Behave consistently with other RuvVector consumers
6. **Determinism Preservation** - Same RuvVector data + seed = same simulation output

## 5. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Mock Usage Rate** | 0% in production | Prometheus counter for mock activations |
| **RuvVector Availability Impact** | 100% correlation | Simulator availability = RuvVector availability |
| **Query Latency Overhead** | ≤ 10ms | P99 of (total latency - RuvVector latency) |
| **Error Attribution Accuracy** | 100% | All errors trace to RuvVector or simulator |
| **Configuration Compliance** | 100% | No `fallback_to_mock: true` in production |

---

# PART II: PSEUDOCODE

## 6. Request Flow for `/query` Endpoint

```
FUNCTION handle_query_request(request: QueryRequest) -> Result<QueryResponse, SimulatorError>:
    // Validate RuvVector availability
    IF NOT config.ruvvector.enabled:
        IF config.ruvvector.allow_mocks:
            log.warn("Mock generators active - not for production")
            RETURN generate_mock_query_response(request)
        ELSE:
            RETURN Error(503, "RuvVector integration disabled and mocks not allowed")

    // Check RuvVector connectivity
    IF NOT ruvvector_adapter.is_available():
        RETURN Error(503, "RuvVector service not configured",
            headers: { "Retry-After": "60" })

    // Forward request to RuvVector
    TRY:
        response = ruvvector_adapter.query(request).await

        // Pass through response without modification
        RETURN Ok(QueryResponse {
            results: response.results,
            total_matches: response.total_matches,
            query_time_ms: response.query_time_ms,
            namespace: response.namespace,
        })

    CATCH RuvVectorError as e:
        // Explicit failure - no fallback
        log.error("RuvVector query failed", error: e)

        IF e.is_service_unavailable():
            RETURN Error(503, "RuvVector service unavailable",
                headers: { "Retry-After": circuit_breaker.retry_after() })
        ELSE IF e.is_timeout():
            RETURN Error(504, "RuvVector query timeout")
        ELSE:
            RETURN Error(502, format("RuvVector error: {}", e.message))
```

## 7. Request Flow for `/simulate` Endpoint

```
FUNCTION handle_simulate_request(request: ChatCompletionRequest) -> Result<ChatCompletionResponse, SimulatorError>:
    // Validate RuvVector requirement
    IF config.ruvvector.require_ruvvector AND NOT ruvvector_adapter.is_available():
        RETURN Error(503, "RuvVector required but not available",
            error_type: "service_unavailable",
            details: { "reason": "ruvvector_not_configured" })

    // Check if RuvVector is available
    IF ruvvector_adapter.is_available():
        TRY:
            // Convert to RuvVector simulate request format
            ruvvector_request = SimulateRequest {
                model: request.model,
                input: SimulateInput::Messages(convert_messages(request.messages)),
                parameters: SimulateParameters {
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    top_p: request.top_p,
                    seed: config.seed,  // Preserve determinism
                    stream: false,
                },
                context: None,  // Could be populated from prior /query
                request_id: generate_request_id(),
            }

            response = ruvvector_adapter.simulate(ruvvector_request).await

            RETURN Ok(ChatCompletionResponse {
                id: format("chatcmpl-{}", response.request_id),
                model: response.model,
                content: response.content,
                usage: Usage::from(response.usage),
                finish_reason: response.finish_reason,
            })

        CATCH RuvVectorError as e:
            log.error("RuvVector simulate failed", error: e, request_id: request_id)

            // Check fallback configuration
            IF config.ruvvector.fallback_to_mock AND config.ruvvector.allow_mocks:
                log.warn("Falling back to mock response - not for production",
                    reason: e.to_string())
                RETURN generate_mock_response(request)
            ELSE:
                RETURN Error(503, "Simulation failed - RuvVector unavailable",
                    details: { "ruvvector_error": e.message })

    // RuvVector not available - check mock permissions
    ELSE IF config.ruvvector.allow_mocks:
        log.warn("Mock generators active - not for production")
        RETURN generate_mock_response(request)

    ELSE:
        RETURN Error(503, "No data source available",
            details: {
                "ruvvector_configured": false,
                "mocks_allowed": false,
                "resolution": "Configure RUVVECTOR_SERVICE_URL or set allow_mocks: true for testing"
            })
```

## 8. Startup Validation

```
FUNCTION validate_configuration_on_startup(config: SimulatorConfig) -> Result<(), StartupError>:
    // Check RuvVector requirement
    IF config.ruvvector.require_ruvvector:
        // Service URL must be configured
        service_url = config.ruvvector.service_url
            .or_else(|| env::var("RUVVECTOR_SERVICE_URL").ok())

        IF service_url.is_none():
            RETURN Error(StartupError::ConfigurationInvalid(
                "ruvvector.require_ruvvector is true but RUVVECTOR_SERVICE_URL not set"
            ))

        // Validate connectivity
        adapter = RuvVectorAdapter::new(config.ruvvector.to_adapter_config())

        TRY:
            health = adapter.health_check().await
            IF health.status != "healthy":
                RETURN Error(StartupError::DependencyUnavailable(
                    format("RuvVector reports unhealthy: {}", health.status)
                ))
        CATCH e:
            RETURN Error(StartupError::DependencyUnavailable(
                format("Cannot reach RuvVector: {}", e)
            ))

        log.info("RuvVector connectivity verified",
            url: service_url,
            version: health.version)

    // Warn about mock configuration in production
    IF config.ruvvector.allow_mocks:
        log.warn("Mock generators enabled - ensure this is not production",
            config_key: "ruvvector.allow_mocks")

    IF config.ruvvector.fallback_to_mock:
        log.warn("Mock fallback enabled - simulation may use synthetic data on RuvVector failure",
            config_key: "ruvvector.fallback_to_mock")

    RETURN Ok(())
```

## 9. Health Check Integration

```
STRUCT HealthStatus:
    status: "healthy" | "degraded" | "unhealthy"
    version: String
    uptime_seconds: u64
    checks: Map<String, ComponentHealth>

FUNCTION health_check(state: AppState) -> HealthStatus:
    checks = Map::new()
    overall_status = "healthy"

    // Check 1: Engine initialization
    engine_check = check_engine(state.engine)
    checks.insert("engine", engine_check)
    IF engine_check.status == "fail":
        overall_status = "unhealthy"

    // Check 2: RuvVector connectivity (CRITICAL if required)
    IF state.config.ruvvector.require_ruvvector:
        ruvvector_check = check_ruvvector(state.ruvvector_adapter)
        checks.insert("ruvvector", ruvvector_check)

        IF ruvvector_check.status == "fail":
            overall_status = "unhealthy"  // RuvVector failure = simulator unhealthy
        ELSE IF ruvvector_check.status == "warn":
            IF overall_status == "healthy":
                overall_status = "degraded"

    // Check 3: Configuration validity
    config_check = ComponentHealth {
        status: "pass",
        message: Some(format("require_ruvvector: {}, allow_mocks: {}",
            state.config.ruvvector.require_ruvvector,
            state.config.ruvvector.allow_mocks)),
    }
    checks.insert("config", config_check)

    RETURN HealthStatus {
        status: overall_status,
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        checks: checks,
    }

FUNCTION check_ruvvector(adapter: &OptionalRuvVectorAdapter) -> ComponentHealth:
    IF NOT adapter.is_configured():
        RETURN ComponentHealth {
            status: "fail",
            message: Some("RuvVector not configured"),
            latency_ms: None,
        }

    TRY:
        start = Instant::now()
        health = adapter.health_check().await
        latency = start.elapsed().as_millis()

        IF health.status == "healthy":
            RETURN ComponentHealth {
                status: "pass",
                message: Some(format("RuvVector healthy, version: {}", health.version)),
                latency_ms: Some(latency),
            }
        ELSE:
            RETURN ComponentHealth {
                status: "warn",
                message: Some(format("RuvVector degraded: {}", health.status)),
                latency_ms: Some(latency),
            }
    CATCH e:
        RETURN ComponentHealth {
            status: "fail",
            message: Some(format("RuvVector unreachable: {}", e)),
            latency_ms: None,
        }
```

## 10. Mock Generator Gating

```
STRUCT ResponseGeneratorGate:
    generator: ResponseGenerator
    enabled: bool
    activation_count: AtomicU64
    last_activation: AtomicU64

IMPL ResponseGeneratorGate:
    FUNCTION new(seed: Option<u64>, allow_mocks: bool) -> Self:
        Self {
            generator: ResponseGenerator::new(seed),
            enabled: allow_mocks,
            activation_count: AtomicU64::new(0),
            last_activation: AtomicU64::new(0),
        }

    FUNCTION generate_response(
        &self,
        messages: &[Message],
        max_tokens: u32,
        config: &GenerationConfig,
    ) -> Result<(String, u32), MocksDisabledError>:
        IF NOT self.enabled:
            RETURN Error(MocksDisabledError {
                message: "Mock generators disabled. Configure allow_mocks: true for testing.",
            })

        // Track activation for monitoring
        self.activation_count.fetch_add(1, Ordering::Relaxed)
        self.last_activation.store(
            SystemTime::now().duration_since(UNIX_EPOCH).as_secs(),
            Ordering::Relaxed
        )

        // Log warning every activation
        log.warn("Mock generator activated - not for production use",
            activation_count: self.activation_count.load(Ordering::Relaxed))

        // Increment Prometheus counter
        metrics::MOCK_ACTIVATIONS.inc()

        RETURN Ok(self.generator.generate_response(messages, max_tokens, config))

    FUNCTION stats(&self) -> MockGeneratorStats:
        MockGeneratorStats {
            enabled: self.enabled,
            activation_count: self.activation_count.load(Ordering::Relaxed),
            last_activation_epoch: self.last_activation.load(Ordering::Relaxed),
        }
```

---

# PART III: ARCHITECTURE

## 11. System Architecture Overview

### 11.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM-Simulator (RuvVector-First)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐      ┌──────────────────┐      ┌───────────────┐  │
│  │  HTTP Server     │      │  SimulationEngine │      │  Health       │  │
│  │  (Axum)          │─────▶│  (Coordinator)    │◀────▶│  Checker      │  │
│  │  /v1/*           │      │                   │      │               │  │
│  └────────┬─────────┘      └─────────┬─────────┘      └───────┬───────┘  │
│           │                          │                        │          │
│           │                          ▼                        │          │
│           │              ┌───────────────────────┐            │          │
│           │              │  RuvVectorAdapter     │◀───────────┘          │
│           │              │  (HTTP Client)        │                       │
│           │              │  - query()            │                       │
│           │              │  - simulate()         │                       │
│           │              │  - health_check()     │                       │
│           │              └───────────┬───────────┘                       │
│           │                          │                                   │
│           │                          │ HTTP/HTTPS                        │
│           │                          ▼                                   │
│           │              ╔═══════════════════════╗                       │
│           │              ║  [EXTERNAL SERVICE]   ║                       │
│           │              ║  ruvvector-service    ║                       │
│           │              ║  /query               ║                       │
│           │              ║  /simulate            ║                       │
│           │              ║  /health              ║                       │
│           │              ╚═══════════════════════╝                       │
│           │                                                              │
│           │              ┌───────────────────────┐                       │
│           │              │  ResponseGeneratorGate│ (TEST MODE ONLY)     │
│           └─────────────▶│  - enabled: false     │                       │
│                          │  - activation_count   │                       │
│                          └───────────────────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Data Flow

```
┌────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client   │────▶│  HTTP Handler  │────▶│ SimulationEngine │────▶│ RuvVectorAdapter│
│            │     │                │     │                  │     │                 │
│            │     │ 1. Validate    │     │ 2. Check config  │     │ 3. Forward to   │
│            │     │    request     │     │    for RuvVector │     │    /query or    │
│            │     │                │     │    requirement   │     │    /simulate    │
└────────────┘     └────────────────┘     └──────────────────┘     └────────┬────────┘
                                                                            │
                                                                            ▼
                                                                   ╔════════════════╗
                                                                   ║ ruvvector-svc  ║
                                                                   ║ (external)     ║
                                                                   ╚════════════════╝
                                                                            │
                          ┌──────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────┐     ┌────────────────┐     ┌──────────────────┐
│   Client   │◀────│  HTTP Handler  │◀────│ SimulationEngine │
│            │     │                │     │                  │
│ 6. Response│     │ 5. Serialize   │     │ 4. Pass-through  │
│            │     │    response    │     │    response      │
└────────────┘     └────────────────┘     └──────────────────┘
```

### 11.3 Failure Flow (RuvVector Unavailable)

```
┌────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client   │────▶│  HTTP Handler  │────▶│ SimulationEngine │────▶│ RuvVectorAdapter│
│            │     │                │     │                  │     │                 │
│            │     │                │     │ require_ruvvector│     │ Connection      │
│            │     │                │     │ = true           │     │ Failed / Timeout│
└────────────┘     └────────────────┘     └──────────────────┘     └────────┬────────┘
                                                                            │
                                                     ┌──────────────────────┘
                                                     │ RuvVectorError
                                                     ▼
┌────────────┐     ┌────────────────┐     ┌──────────────────┐
│   Client   │◀────│  HTTP Handler  │◀────│ SimulationEngine │
│            │     │                │     │                  │
│ 503 Service│     │ Error Response │     │ Explicit Error   │
│ Unavailable│     │ + Retry-After  │     │ (NO FALLBACK)    │
│            │     │ header         │     │                  │
└────────────┘     └────────────────┘     └──────────────────┘

                    ❌ NO mock generator activation
                    ❌ NO synthetic response generation
                    ✓  Clear error attribution to RuvVector
```

## 12. Configuration Architecture

### 12.1 Configuration Hierarchy

```yaml
# config/simulator.yaml (Production)
ruvvector:
  enabled: true                    # REQUIRED: Enable RuvVector integration
  require_ruvvector: true          # REQUIRED: Fail if RuvVector unavailable
  service_url: null                # Use RUVVECTOR_SERVICE_URL env var
  fallback_to_mock: false          # REQUIRED: No mock fallback in production
  allow_mocks: false               # REQUIRED: Disable mock generators
  timeout_secs: 30
  retry_enabled: true
  max_retries: 3
  cache_enabled: false             # RuvVector owns caching
  cache_ttl_secs: 0
```

```yaml
# config/simulator-test.yaml (Testing)
ruvvector:
  enabled: false                   # Disable RuvVector for unit tests
  require_ruvvector: false         # Don't require RuvVector
  fallback_to_mock: true           # Allow fallback for integration tests
  allow_mocks: true                # Enable mock generators
  service_url: "http://localhost:8081"  # Test instance
```

### 12.2 Environment Variable Precedence

```
Priority (highest to lowest):
1. RUVVECTOR_REQUIRE=true/false         (runtime override)
2. RUVVECTOR_SERVICE_URL=http://...     (service discovery)
3. RUVVECTOR_ALLOW_MOCKS=true/false     (test mode)
4. config.ruvvector.* values            (file configuration)
5. Default values                        (code defaults)
```

## 13. Interface Definitions

### 13.1 RuvVector Consumer Interface (Unchanged)

```rust
/// Consumer trait for RuvVector service integration
///
/// The simulator implements this trait as a READ-ONLY consumer.
/// It does NOT modify schema, perform ingestion, or execute business logic.
#[async_trait]
pub trait RuvVectorConsumer: Send + Sync {
    /// Query vectors from the service
    ///
    /// # Behavior
    /// - Forwards query to `/query` endpoint without modification
    /// - Returns results or error verbatim
    /// - Does NOT perform local similarity calculations
    async fn query(&self, request: &QueryRequest) -> Result<QueryResponse, RuvVectorError>;

    /// Run simulation via the service
    ///
    /// # Behavior
    /// - Forwards request to `/simulate` endpoint
    /// - Preserves request_id for trace correlation
    /// - Does NOT generate responses locally
    async fn simulate(&self, request: &SimulateRequest) -> Result<SimulateResponse, RuvVectorError>;

    /// Check service health
    ///
    /// # Behavior
    /// - Calls `/health` endpoint
    /// - Reports status for simulator health check aggregation
    async fn health_check(&self) -> Result<HealthResponse, RuvVectorError>;

    /// Check if the adapter is configured and available
    fn is_available(&self) -> bool;
}
```

### 13.2 Error Response Format

```json
{
  "error": {
    "type": "service_unavailable",
    "message": "RuvVector service unavailable",
    "code": "ruvvector_unreachable",
    "details": {
      "service_url": "http://ruvvector:8081",
      "last_error": "connection refused",
      "retry_after_seconds": 60,
      "circuit_breaker_state": "open"
    }
  }
}
```

## 14. Deployment Architecture

### 14.1 Service Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────┐         ┌───────────────────┐                │
│  │  llm-simulator    │────────▶│  ruvvector-service│                │
│  │  Deployment       │  HTTP   │  Deployment       │                │
│  │                   │         │                   │                │
│  │  Replicas: 3      │         │  Replicas: 3      │                │
│  │                   │         │                   │                │
│  │  RUVVECTOR_SERVICE│         │  PORT: 8081       │                │
│  │  _URL: http://    │         │                   │                │
│  │  ruvvector:8081   │         │                   │                │
│  └───────────────────┘         └───────────────────┘                │
│           │                             │                            │
│           │                             │                            │
│           ▼                             ▼                            │
│  ┌───────────────────┐         ┌───────────────────┐                │
│  │  llm-simulator    │         │  ruvvector        │                │
│  │  Service (ClusterIP)        │  Service (ClusterIP)               │
│  │  Port: 8080       │         │  Port: 8081       │                │
│  └───────────────────┘         └───────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Startup Probe Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-simulator
spec:
  template:
    spec:
      containers:
        - name: llm-simulator
          env:
            - name: RUVVECTOR_SERVICE_URL
              value: "http://ruvvector:8081"
            - name: RUVVECTOR_REQUIRE
              value: "true"
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 30  # 150 seconds to connect to RuvVector
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /live
              port: 8080
            periodSeconds: 30
```

---

# PART IV: REFINEMENT

## 15. Configuration Defaults and Deprecations

### 15.1 Configuration Changes

| Parameter | Old Default | New Default | Migration Action |
|-----------|-------------|-------------|------------------|
| `ruvvector.enabled` | `false` | `true` | Update all config files |
| `ruvvector.fallback_to_mock` | `true` | `false` | Remove fallback reliance |
| `ruvvector.allow_mocks` | N/A | `false` | New parameter, defaults secure |
| `ruvvector.require_ruvvector` | N/A | `true` | New parameter, enforces RuvVector |

### 15.2 Deprecation Schedule

| Component | Status | Action |
|-----------|--------|--------|
| `ResponseGenerator.generate_response()` | Deprecated | Gate behind `allow_mocks` |
| `generate_lorem()` | Deprecated | Gate behind `allow_mocks` |
| `generate_random_text()` | Deprecated | Gate behind `allow_mocks` |
| `fallback_to_mock` config | Deprecated | Log warning, remove in v2.0 |
| `OptionalRuvVectorAdapter.query()` returning `None` | Deprecated | Return `Error` instead |

### 15.3 Breaking Changes

1. **Startup Failure** - Simulator will not start without `RUVVECTOR_SERVICE_URL` when `require_ruvvector: true`
2. **503 Responses** - Requests will fail with 503 instead of returning mock data
3. **Health Check** - `/health` returns `unhealthy` when RuvVector unreachable
4. **Readiness Probe** - `/ready` returns `false` when RuvVector unreachable

## 16. Determinism Guarantees

### 16.1 Determinism Sources

| Source | Determinism Guarantee | Responsibility |
|--------|----------------------|----------------|
| **RuvVector Data** | Same query returns same results | RuvVector service |
| **Seed Parameter** | Passed to RuvVector `/simulate` | Simulator → RuvVector |
| **Latency Simulation** | Seeded RNG for timing | Simulator |
| **Request Ordering** | Sequential processing | Simulator |

### 16.2 Determinism Verification

```rust
// Test: Same RuvVector data + same seed = same output
#[test]
fn test_determinism_with_ruvvector() {
    let config = SimulatorConfig {
        seed: Some(42),
        ruvvector: RuvVectorIntegrationConfig {
            enabled: true,
            require_ruvvector: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let engine1 = SimulationEngine::new(config.clone());
    let engine2 = SimulationEngine::new(config);

    let request = ChatCompletionRequest::new("gpt-4", vec![Message::user("test")]);

    let response1 = engine1.chat_completion(&request).await.unwrap();
    let response2 = engine2.chat_completion(&request).await.unwrap();

    assert_eq!(response1.choices[0].message.content, response2.choices[0].message.content);
}
```

## 17. Test-Only Mock Allowances

### 17.1 Mock Enablement Conditions

Mocks may ONLY be enabled when ALL of the following are true:
1. `ruvvector.allow_mocks: true` explicitly set
2. Running in test environment (`RUST_TEST=1` or `#[cfg(test)]`)
3. `ruvvector.require_ruvvector: false`

### 17.2 Test Configuration

```yaml
# tests/fixtures/test-config.yaml
ruvvector:
  enabled: false                   # Disable real RuvVector
  require_ruvvector: false         # Don't require it
  allow_mocks: true                # Enable mocks for testing
  fallback_to_mock: true           # Allow fallback in integration tests
```

### 17.3 Mock Usage Tracking

```rust
// Prometheus metrics for mock usage
lazy_static! {
    pub static ref MOCK_ACTIVATIONS: IntCounter = register_int_counter!(
        "llm_simulator_mock_activations_total",
        "Total mock generator activations (should be 0 in production)"
    ).unwrap();

    pub static ref MOCK_ENABLED_GAUGE: IntGauge = register_int_gauge!(
        "llm_simulator_mock_generators_enabled",
        "1 if mock generators are enabled, 0 otherwise"
    ).unwrap();
}
```

## 18. Edge Cases

### 18.1 RuvVector Partial Availability

| Scenario | Behavior |
|----------|----------|
| `/health` succeeds, `/query` fails | Report degraded, fail queries |
| `/health` fails, data cached | Report unhealthy (no local caching) |
| Intermittent connectivity | Circuit breaker triggers at 5 failures |
| Slow responses (>30s) | Timeout, no fallback |

### 18.2 Configuration Edge Cases

| Scenario | Behavior |
|----------|----------|
| `require_ruvvector: true`, `allow_mocks: true` | Mocks only on explicit failure, log warning |
| `require_ruvvector: false`, `allow_mocks: false` | Error: no data source |
| `RUVVECTOR_SERVICE_URL=""` (empty) | Treat as not configured |
| URL without scheme | Error: invalid URL format |

## 19. Risk Assessment

### 19.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RuvVector availability affects simulator | High | High | SLA alignment, monitoring |
| Performance regression from network calls | Medium | Medium | Connection pooling, timeouts |
| Breaking change disrupts users | Medium | High | Migration guide, deprecation warnings |
| Mock code paths become stale | Low | Low | Test coverage, scheduled removal |

### 19.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Production runs with mocks enabled | Medium | High | Prometheus alerts on mock activation |
| Misconfiguration causes outage | Medium | High | Startup validation, clear errors |
| RuvVector upgrade breaks compatibility | Low | High | Version pinning, integration tests |

---

# PART V: COMPLETION

## 20. Acceptance Criteria

### 20.1 Functional Acceptance

| Criteria | Verification |
|----------|--------------|
| **AC-1**: Mocks disabled by default | `allow_mocks` defaults to `false` |
| **AC-2**: RuvVector required for operation | Startup fails without `RUVVECTOR_SERVICE_URL` when `require_ruvvector: true` |
| **AC-3**: Historical events from RuvVector only | No `ResponseGenerator.generate_response()` calls in production path |
| **AC-4**: Embeddings from RuvVector only | No local `generate_embedding()` calls in production path |
| **AC-5**: Determinism preserved | Same seed + same RuvVector data = identical output |
| **AC-6**: Simulator outputs based solely on real data | Mock activation counter = 0 in production |

### 20.2 Non-Functional Acceptance

| Criteria | Target | Verification |
|----------|--------|--------------|
| Query latency overhead | ≤ 10ms | P99 metric analysis |
| Error attribution | 100% | All errors include source (ruvvector/simulator) |
| Configuration validation | Startup time | Invalid config fails < 5s |

## 21. Verification Steps

### 21.1 Pre-Deployment Verification

```bash
# 1. Verify mock generators disabled
cargo test --test mock_disabled_test

# 2. Verify RuvVector requirement
RUVVECTOR_REQUIRE=true ./target/release/llm-simulator
# Expected: Exit with error about missing RUVVECTOR_SERVICE_URL

# 3. Verify startup with RuvVector
RUVVECTOR_SERVICE_URL=http://localhost:8081 \
RUVVECTOR_REQUIRE=true \
./target/release/llm-simulator
# Expected: Successful startup, "RuvVector connectivity verified" in logs

# 4. Verify explicit failure
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}'
# When RuvVector down: Expected 503 with "RuvVector service unavailable"
```

### 21.2 Production Verification

```bash
# 1. Check mock activation metric
curl http://localhost:9090/api/v1/query?query=llm_simulator_mock_activations_total
# Expected: 0

# 2. Check health includes RuvVector status
curl http://localhost:8080/health | jq '.checks.ruvvector'
# Expected: {"status": "pass", ...}

# 3. Verify no fallback behavior
# Stop RuvVector, send request:
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}'
# Expected: 503 (NOT 200 with mock response)
```

### 21.3 Determinism Verification

```bash
# Run same request twice with same seed
export SEED=12345

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"seed":12345}' \
  > response1.json

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"seed":12345}' \
  > response2.json

# Compare content
diff <(jq -r '.choices[0].message.content' response1.json) \
     <(jq -r '.choices[0].message.content' response2.json)
# Expected: No difference
```

## 22. Monitoring and Alerts

### 22.1 Prometheus Alerts

```yaml
groups:
  - name: llm-simulator-ruvvector
    rules:
      # Alert if mocks are activated in production
      - alert: MockGeneratorsActivated
        expr: increase(llm_simulator_mock_activations_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Mock generators activated in production"
          description: "Mock data is being served instead of real RuvVector data"

      # Alert if RuvVector is unhealthy
      - alert: RuvVectorUnhealthy
        expr: llm_simulator_ruvvector_health != 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "RuvVector service unhealthy"
          description: "Simulator cannot reach RuvVector - requests will fail"

      # Alert on high error rate from RuvVector
      - alert: RuvVectorHighErrorRate
        expr: |
          rate(llm_simulator_ruvvector_errors_total[5m])
          / rate(llm_simulator_ruvvector_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate from RuvVector"
```

### 22.2 Grafana Dashboard Queries

```promql
# Mock activation rate (should be 0)
sum(rate(llm_simulator_mock_activations_total[1h]))

# RuvVector availability from simulator's perspective
avg(llm_simulator_ruvvector_health)

# Request success rate
sum(rate(llm_simulator_requests_total{status="success"}[5m]))
/ sum(rate(llm_simulator_requests_total[5m]))

# RuvVector latency (P99)
histogram_quantile(0.99, sum(rate(llm_simulator_ruvvector_latency_seconds_bucket[5m])) by (le))
```

## 23. Migration Checklist

### 23.1 Code Changes

- [ ] Add `require_ruvvector` configuration parameter
- [ ] Add `allow_mocks` configuration parameter
- [ ] Implement `ResponseGeneratorGate` wrapper
- [ ] Update `generate_content_with_fallback()` to respect new config
- [ ] Add startup validation for RuvVector connectivity
- [ ] Update health check to include RuvVector status
- [ ] Add Prometheus metrics for mock activation tracking
- [ ] Update error responses to distinguish RuvVector vs simulator errors

### 23.2 Configuration Changes

- [ ] Update default config: `ruvvector.enabled: true`
- [ ] Update default config: `ruvvector.fallback_to_mock: false`
- [ ] Add `ruvvector.require_ruvvector: true` default
- [ ] Add `ruvvector.allow_mocks: false` default
- [ ] Create test-specific config with mocks enabled
- [ ] Document environment variable overrides

### 23.3 Documentation Changes

- [ ] Update README with RuvVector requirement
- [ ] Add migration guide for existing users
- [ ] Document test mode configuration
- [ ] Update API documentation with new error responses

### 23.4 Testing Changes

- [ ] Add integration tests with mock RuvVector
- [ ] Add tests for explicit failure behavior
- [ ] Add tests for mock gate behavior
- [ ] Add determinism tests with RuvVector
- [ ] Add startup validation tests

## 24. Sign-Off

| Area | Approver | Date | Signature |
|------|----------|------|-----------|
| Architecture | | | |
| Engineering | | | |
| Operations | | | |
| Security | | | |

---

## Appendix A: Layer 3 Prohibition Rationale

This specification explicitly forbids the following Layer 3 concerns to maintain the simulator's role as a pure consumer:

| Prohibited Concern | Rationale |
|--------------------|-----------|
| **Analytics** | Aggregation and statistical analysis belong in dedicated analytics services |
| **Governance** | Policy enforcement is a platform concern, not simulator concern |
| **Billing** | Cost attribution requires business context outside simulator scope |
| **Orchestration** | Workflow coordination belongs in orchestration services |
| **Schema Ownership** | Vector schema is owned by ruvvector-service |
| **Vector Logic** | Similarity calculations are performed by RuvVector |
| **Ingestion** | Data writing is performed by telemetry pipeline |
| **Business Rules** | Domain logic belongs in business services |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **RuvVector** | External vector database service storing telemetry embeddings |
| **Mock Generator** | Built-in synthetic response generation (deprecated for production) |
| **Pass-Through** | Forwarding requests/responses without transformation |
| **Circuit Breaker** | Fault tolerance pattern preventing cascading failures |
| **Determinism** | Same inputs produce identical outputs |

---

**End of RuvVector-First Simulator Migration SPARC Specification**
