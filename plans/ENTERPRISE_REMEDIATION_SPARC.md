# LLM-Simulator Enterprise Remediation SPARC Specification

**Document Version:** 1.0.0
**Created:** November 26, 2025
**Status:** Approved for Implementation

---

## Executive Summary

This document provides a comprehensive SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) specification for remediating the gaps identified in the LLM-Simulator Enterprise Readiness Assessment. The remediation is organized into four phases to bring the project from its current state (6.4/10) to production-ready (9/10).

### Current State vs Target State

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| Security | 3.5/10 | 9/10 | Critical |
| Observability | 6.2/10 | 9/10 | High |
| Testing | 4.0/10 | 8/10 | Critical |
| Operations | 7.5/10 | 9/10 | Medium |
| **Overall** | **6.4/10** | **9/10** | |

### Implementation Timeline

| Phase | Focus | Timeline | Priority |
|-------|-------|----------|----------|
| Phase 1 | Security Fixes | Week 1 | 🔴 Critical |
| Phase 2 | Observability | Week 1-2 | 🟡 High |
| Phase 3 | Testing | Week 2-3 | 🟡 High |
| Phase 4 | Operations | Week 3-4 | 🟢 Medium |

### Phase Dependencies

```
Phase 1 (Security) ──┐
                     ├──▶ Phase 3 (Testing) ──┐
Phase 2 (Observability)                       ├──▶ Production Ready
                     ├──▶ Phase 4 (Operations)┘
```

---

# PHASE 1: SECURITY FIXES

## S - Specification

### Overview
Implement production-grade security controls for the LLM-Simulator to prevent unauthorized access, protect against common web vulnerabilities, and enable safe deployment in enterprise environments.

### Objectives
1. Implement API key authentication with configurable validation
2. Add role-based authorization for admin endpoints
3. Restrict CORS to explicitly allowed origins
4. Implement token bucket rate limiting
5. Add security headers to all responses

### Requirements

#### 1.1 API Key Authentication
- **MUST** validate API keys in `Authorization: Bearer <key>` header format
- **MUST** support configurable API key list via environment variable or config file
- **MUST** return 401 Unauthorized for missing/invalid keys
- **SHOULD** support key rotation without restart
- **MAY** support multiple key tiers (admin, user, readonly)

#### 1.2 Admin Endpoint Authorization
- **MUST** require special admin API key for `/admin/*` endpoints
- **MUST** return 403 Forbidden for non-admin keys accessing admin routes
- **SHOULD** log all admin endpoint access attempts
- **SHOULD** support IP allowlist for admin access

#### 1.3 CORS Restriction
- **MUST** replace wildcard `*` with explicit origin allowlist
- **MUST** make allowed origins configurable via config/env
- **MUST** validate `Origin` header against allowlist
- **SHOULD** support regex patterns for origin matching
- **MUST** set `Access-Control-Allow-Credentials: false` by default

#### 1.4 Rate Limiting
- **MUST** implement token bucket algorithm per API key
- **MUST** return 429 Too Many Requests with `Retry-After` header
- **MUST** make limits configurable (requests/minute, burst size)
- **SHOULD** support different limits per endpoint tier
- **SHOULD** expose rate limit headers (`X-RateLimit-*`)

#### 1.5 Security Headers
- **MUST** add `X-Content-Type-Options: nosniff`
- **MUST** add `X-Frame-Options: DENY`
- **MUST** add `X-XSS-Protection: 1; mode=block`
- **SHOULD** add `Content-Security-Policy` header
- **SHOULD** add `Strict-Transport-Security` when behind TLS

### Success Criteria
- All endpoints require valid API key (except health checks)
- Admin endpoints accessible only with admin keys
- CORS blocks requests from non-allowed origins
- Rate limiting enforces configured limits
- All security headers present in responses
- Zero authentication bypasses in security audit

---

## P - Pseudocode

### 1.1 API Key Validation Middleware

```
FUNCTION api_key_middleware(request, config):
    // Skip auth for health endpoints
    IF request.path IN ["/health", "/healthz", "/ready", "/readyz", "/metrics"]:
        RETURN next(request)

    // Extract API key from header
    auth_header = request.headers.get("Authorization")
    IF auth_header IS NULL:
        RETURN error_response(401, "Missing Authorization header")

    IF NOT auth_header.starts_with("Bearer "):
        RETURN error_response(401, "Invalid Authorization format")

    api_key = auth_header.strip_prefix("Bearer ")

    // Validate key against configured keys
    key_info = config.api_keys.validate(api_key)
    IF key_info IS NULL:
        log.warn("Invalid API key attempt", key_prefix=api_key[0:8])
        RETURN error_response(401, "Invalid API key")

    // Attach key info to request context
    request.extensions.insert(ApiKeyInfo(key_info))

    RETURN next(request)
```

### 1.2 Admin Authorization Middleware

```
FUNCTION admin_auth_middleware(request):
    IF NOT request.path.starts_with("/admin"):
        RETURN next(request)

    key_info = request.extensions.get(ApiKeyInfo)
    IF key_info IS NULL:
        RETURN error_response(401, "Authentication required")

    IF key_info.role != "admin":
        log.warn("Non-admin access attempt to admin endpoint",
                 key_id=key_info.id, path=request.path)
        RETURN error_response(403, "Admin access required")

    // Optional: Check IP allowlist
    IF config.admin_ip_allowlist IS NOT EMPTY:
        client_ip = request.client_ip()
        IF client_ip NOT IN config.admin_ip_allowlist:
            log.warn("Admin access from non-allowed IP", ip=client_ip)
            RETURN error_response(403, "IP not allowed for admin access")

    RETURN next(request)
```

### 1.3 CORS Configuration

```
STRUCT CorsConfig:
    enabled: bool
    allowed_origins: Vec<String>
    allowed_methods: Vec<Method>
    allowed_headers: Vec<String>
    allow_credentials: bool
    max_age: Duration

FUNCTION build_cors_layer(config: CorsConfig) -> CorsLayer:
    IF NOT config.enabled:
        RETURN CorsLayer::very_permissive()  // For development only

    layer = CorsLayer::new()

    // Build origin validator
    IF config.allowed_origins.contains("*"):
        log.warn("Wildcard CORS origin detected - not recommended for production")
        layer = layer.allow_origin(Any)
    ELSE:
        origins = config.allowed_origins
            .map(|o| o.parse::<HeaderValue>())
            .collect()
        layer = layer.allow_origin(origins)

    layer = layer
        .allow_methods(config.allowed_methods)
        .allow_headers(config.allowed_headers)
        .allow_credentials(config.allow_credentials)
        .max_age(config.max_age)

    RETURN layer
```

### 1.4 Token Bucket Rate Limiter

```
STRUCT TokenBucket:
    capacity: u32           // Max tokens
    tokens: AtomicU32       // Current tokens
    refill_rate: u32        // Tokens per second
    last_refill: AtomicU64  // Last refill timestamp

STRUCT RateLimiter:
    buckets: DashMap<String, TokenBucket>  // Key -> Bucket
    config: RateLimitConfig

FUNCTION rate_limit_middleware(request, limiter):
    key_info = request.extensions.get(ApiKeyInfo)
    IF key_info IS NULL:
        RETURN next(request)  // Will be caught by auth middleware

    bucket_key = key_info.id
    limit_config = limiter.config.get_limit_for_tier(key_info.tier)

    bucket = limiter.buckets.entry(bucket_key)
        .or_insert(TokenBucket::new(limit_config))

    // Refill tokens based on elapsed time
    bucket.refill()

    // Try to consume a token
    IF NOT bucket.try_consume(1):
        retry_after = bucket.time_until_token()
        RETURN error_response(429, "Rate limit exceeded")
            .header("Retry-After", retry_after.as_secs())
            .header("X-RateLimit-Limit", limit_config.requests_per_minute)
            .header("X-RateLimit-Remaining", bucket.tokens())
            .header("X-RateLimit-Reset", bucket.next_reset_time())

    // Add rate limit headers to response
    response = next(request)
    response.headers.insert("X-RateLimit-Limit", limit_config.requests_per_minute)
    response.headers.insert("X-RateLimit-Remaining", bucket.tokens())

    RETURN response
```

### 1.5 Security Headers Middleware

```
FUNCTION security_headers_middleware(request, next):
    response = next(request)

    headers = response.headers_mut()

    // Prevent MIME sniffing
    headers.insert("X-Content-Type-Options", "nosniff")

    // Prevent clickjacking
    headers.insert("X-Frame-Options", "DENY")

    // XSS protection (legacy browsers)
    headers.insert("X-XSS-Protection", "1; mode=block")

    // Referrer policy
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin")

    // Content Security Policy (API-focused)
    headers.insert("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")

    // Permissions policy
    headers.insert("Permissions-Policy", "geolocation=(), microphone=(), camera=()")

    RETURN response
```

---

## A - Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Request Flow                                   │
└─────────────────────────────────────────────────────────────────────┘

    Client Request
          │
          ▼
┌─────────────────────┐
│  Security Headers   │  ← Adds headers to all responses
│     Middleware      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CORS Middleware   │  ← Validates Origin header
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   API Key Auth      │  ← Validates Bearer token
│     Middleware      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Rate Limiting      │  ← Token bucket per API key
│     Middleware      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Admin Auth         │  ← Checks admin role for /admin/*
│     Middleware      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Route Handlers    │
└─────────────────────┘
```

### File Structure

```
src/
├── security/
│   ├── mod.rs              # Module exports
│   ├── api_key.rs          # API key validation & storage
│   ├── rate_limit.rs       # Token bucket implementation
│   ├── cors.rs             # CORS configuration builder
│   └── headers.rs          # Security headers middleware
├── server/
│   ├── middleware.rs       # Updated middleware stack
│   └── mod.rs              # Updated server builder
└── config/
    └── security.rs         # Security configuration structs
```

### Configuration Schema

```yaml
# config/default.yaml additions
security:
  # API Key Authentication
  api_keys:
    enabled: true
    keys:
      - id: "key-001"
        key: "${API_KEY_001}"  # From environment
        role: "admin"
        rate_limit_tier: "premium"
      - id: "key-002"
        key: "${API_KEY_002}"
        role: "user"
        rate_limit_tier: "standard"

  # Admin Authorization
  admin:
    require_admin_key: true
    ip_allowlist: []  # Empty = allow all IPs with admin key

  # CORS Settings
  cors:
    enabled: true
    allowed_origins:
      - "https://app.example.com"
      - "https://staging.example.com"
    allowed_methods: ["GET", "POST", "OPTIONS"]
    allowed_headers: ["Content-Type", "Authorization", "X-Request-ID"]
    allow_credentials: false
    max_age_seconds: 3600

  # Rate Limiting
  rate_limiting:
    enabled: true
    default_tier: "standard"
    tiers:
      standard:
        requests_per_minute: 60
        burst_size: 10
      premium:
        requests_per_minute: 600
        burst_size: 100
      admin:
        requests_per_minute: 1000
        burst_size: 200

  # Security Headers
  headers:
    enabled: true
    hsts_enabled: false  # Enable when behind TLS
    hsts_max_age: 31536000
```

---

## R - Refinement

### Edge Cases

1. **Key Rotation During Request**
   - Use versioned keys with grace period
   - Accept old key for 5 minutes after rotation

2. **Rate Limit Bucket Overflow**
   - Cap refill at bucket capacity
   - Use atomic operations for thread safety

3. **CORS Preflight Caching**
   - Set appropriate `Access-Control-Max-Age`
   - Cache preflight responses client-side

4. **Clock Skew in Rate Limiting**
   - Use monotonic clock for timing
   - Don't rely on wall clock for token refill

5. **Memory Growth from Rate Limit Buckets**
   - Implement bucket expiration (TTL)
   - Clean up inactive buckets periodically

### Error Handling

```rust
// Security-specific error types
pub enum SecurityError {
    MissingApiKey,
    InvalidApiKey { key_prefix: String },
    ExpiredApiKey { key_id: String },
    InsufficientPermissions { required: String, actual: String },
    RateLimitExceeded { retry_after: Duration },
    CorsOriginNotAllowed { origin: String },
    IpNotAllowed { ip: IpAddr },
}

impl IntoResponse for SecurityError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            Self::MissingApiKey => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "Missing API key in Authorization header"
            ),
            Self::InvalidApiKey { .. } => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "Invalid API key"
            ),
            Self::InsufficientPermissions { required, .. } => (
                StatusCode::FORBIDDEN,
                "permission_error",
                format!("Requires {} permission", required)
            ),
            Self::RateLimitExceeded { retry_after } => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "Rate limit exceeded"
            ),
            // ... other variants
        };

        // Return OpenAI-compatible error response
        (status, Json(ErrorResponse::new(error_type, &message))).into_response()
    }
}
```

### Performance Considerations

1. **API Key Lookup**: O(1) with HashMap, consider bloom filter for negative lookups
2. **Rate Limit Buckets**: Use `DashMap` for concurrent access without global lock
3. **CORS Origin Matching**: Pre-compile regex patterns, cache results
4. **Header Injection**: Use static `HeaderName` constants to avoid parsing

---

## C - Completion

### Definition of Done

- [ ] API key validation middleware implemented and tested
- [ ] Admin authorization middleware implemented and tested
- [ ] CORS restriction with configurable origins
- [ ] Token bucket rate limiter with per-key tracking
- [ ] Security headers added to all responses
- [ ] Configuration schema documented and validated
- [ ] All existing tests still pass
- [ ] New security tests achieve >90% coverage of security module
- [ ] Security audit passes with no critical findings
- [ ] Documentation updated with security configuration guide

### Verification Checklist

```bash
# 1. Test missing API key
curl -i http://localhost:8080/v1/chat/completions
# Expected: 401 Unauthorized

# 2. Test invalid API key
curl -i -H "Authorization: Bearer invalid-key" http://localhost:8080/v1/chat/completions
# Expected: 401 Unauthorized

# 3. Test valid API key
curl -i -H "Authorization: Bearer $VALID_KEY" http://localhost:8080/v1/chat/completions -d '{...}'
# Expected: 200 OK with response

# 4. Test admin endpoint with user key
curl -i -H "Authorization: Bearer $USER_KEY" http://localhost:8080/admin/stats
# Expected: 403 Forbidden

# 5. Test admin endpoint with admin key
curl -i -H "Authorization: Bearer $ADMIN_KEY" http://localhost:8080/admin/stats
# Expected: 200 OK

# 6. Test CORS from disallowed origin
curl -i -H "Origin: https://evil.com" http://localhost:8080/v1/models
# Expected: No CORS headers / blocked

# 7. Test rate limiting
for i in {1..100}; do curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $KEY" http://localhost:8080/v1/models; done
# Expected: 429 responses after limit exceeded

# 8. Verify security headers
curl -i http://localhost:8080/health
# Expected: X-Content-Type-Options, X-Frame-Options, etc.
```

### Monitoring

- Metric: `llm_simulator_auth_failures_total{reason="..."}`
- Metric: `llm_simulator_rate_limit_hits_total{tier="..."}`
- Metric: `llm_simulator_cors_blocked_total{origin="..."}`
- Alert: Auth failure rate > 10/min from single IP
- Alert: Rate limit exhaustion for any key

---

# PHASE 2: OBSERVABILITY

## S - Specification

### Overview
Enable comprehensive observability for the LLM-Simulator by activating distributed tracing, implementing meaningful health checks, and adding missing metrics to support production monitoring and alerting.

### Objectives
1. Enable OpenTelemetry distributed tracing
2. Implement comprehensive health check logic
3. Add missing metrics (queue_depth, provider/model labels)
4. Ensure alert rules match available metrics
5. Add trace correlation to logs

### Requirements

#### 2.1 Distributed Tracing
- **MUST** call `init_otel()` when OTLP endpoint is configured
- **MUST** create spans for each HTTP request
- **MUST** propagate trace context via W3C headers
- **MUST** add span attributes for model, provider, token counts
- **SHOULD** sample traces based on configuration (default 100%)
- **SHOULD** export to OTLP gRPC endpoint

#### 2.2 Health Check Logic
- **MUST** verify engine initialization state
- **MUST** check configuration validity
- **SHOULD** verify telemetry pipeline connectivity
- **SHOULD** report degraded state (not just healthy/unhealthy)
- **MUST** differentiate liveness vs readiness checks
- **MUST** return detailed component status in response

#### 2.3 Missing Metrics
- **MUST** add `llm_simulator_queue_depth` gauge
- **MUST** add `llm_simulator_queue_capacity` gauge
- **MUST** add provider label to all request metrics
- **MUST** add model label to all request metrics
- **SHOULD** add `llm_simulator_cost_dollars` counter
- **SHOULD** add `llm_simulator_cache_hits_total` counter

#### 2.4 Log-Trace Correlation
- **MUST** include `trace_id` in all log entries when tracing enabled
- **MUST** include `span_id` in all log entries when tracing enabled
- **SHOULD** use structured logging format for trace context
- **SHOULD** support log sampling based on trace sampling

### Success Criteria
- Traces visible in Jaeger/OTLP backend
- Health endpoint returns actual system state
- All alert rules reference existing metrics
- Logs correlate with traces via trace_id
- 100% of requests have trace context

---

## P - Pseudocode

### 2.1 Enable Distributed Tracing

```
FUNCTION run_server(config):
    // Initialize telemetry first
    init_telemetry(config.telemetry)

    // NEW: Initialize OpenTelemetry if endpoint configured
    IF config.telemetry.otlp_endpoint IS NOT NULL:
        init_otel(config.telemetry)
        log.info("OpenTelemetry tracing enabled", endpoint=config.telemetry.otlp_endpoint)

    // Build router with tracing middleware
    router = build_router(config)
        .layer(TraceLayer::new_for_http()
            .make_span_with(|request| {
                tracing::info_span!(
                    "http_request",
                    method = %request.method(),
                    uri = %request.uri(),
                    trace_id = tracing::field::Empty,
                    span_id = tracing::field::Empty,
                )
            })
            .on_response(|response, latency, span| {
                span.record("status", response.status().as_u16());
                span.record("latency_ms", latency.as_millis());
            })
        )

    // Start server
    serve(router, config.server.socket_addr())
```

### 2.2 Health Check Implementation

```
STRUCT HealthStatus:
    status: "healthy" | "degraded" | "unhealthy"
    version: String
    uptime_seconds: u64
    checks: Map<String, ComponentHealth>

STRUCT ComponentHealth:
    status: "pass" | "warn" | "fail"
    message: Option<String>
    latency_ms: Option<u64>

FUNCTION health_check(state) -> HealthStatus:
    checks = Map::new()
    overall_status = "healthy"

    // Check 1: Engine initialization
    engine_check = check_engine(state.engine)
    checks.insert("engine", engine_check)
    IF engine_check.status == "fail":
        overall_status = "unhealthy"

    // Check 2: Configuration validity
    config_check = check_config(state.config)
    checks.insert("config", config_check)
    IF config_check.status == "fail":
        overall_status = "unhealthy"

    // Check 3: Metrics subsystem
    metrics_check = check_metrics(state.metrics)
    checks.insert("metrics", metrics_check)
    IF metrics_check.status == "warn":
        IF overall_status == "healthy":
            overall_status = "degraded"

    // Check 4: Memory usage
    memory_check = check_memory()
    checks.insert("memory", memory_check)
    IF memory_check.status == "warn":
        IF overall_status == "healthy":
            overall_status = "degraded"

    RETURN HealthStatus {
        status: overall_status,
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        checks: checks,
    }

FUNCTION readiness_check(state) -> bool:
    health = health_check(state)
    // Ready only if healthy or degraded (not unhealthy)
    RETURN health.status != "unhealthy"
```

### 2.3 Missing Metrics Implementation

```
STRUCT EnhancedMetrics:
    // Existing metrics
    requests_total: Counter
    request_duration: Histogram
    tokens_input: Counter
    tokens_output: Counter
    errors_total: Counter
    active_requests: Gauge

    // NEW: Queue metrics
    queue_depth: Gauge
    queue_capacity: Gauge

    // NEW: Cost tracking
    cost_dollars: Counter

    // NEW: Cache metrics
    cache_hits: Counter
    cache_misses: Counter

FUNCTION record_request(metrics, request, response, duration):
    // Get labels
    provider = determine_provider(request.model)
    model = request.model

    // Record with labels
    metrics.requests_total
        .with_label("provider", provider)
        .with_label("model", model)
        .with_label("status", response.status)
        .increment()

    metrics.request_duration
        .with_label("provider", provider)
        .with_label("model", model)
        .record(duration.as_secs_f64())

    metrics.tokens_input
        .with_label("provider", provider)
        .with_label("model", model)
        .increment_by(response.usage.prompt_tokens)

    metrics.tokens_output
        .with_label("provider", provider)
        .with_label("model", model)
        .increment_by(response.usage.completion_tokens)

    // Calculate and record cost
    cost = calculate_cost(provider, model, response.usage)
    metrics.cost_dollars
        .with_label("provider", provider)
        .with_label("model", model)
        .increment_by(cost)
```

---

## A - Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Observability Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   OTLP Collector │
                              │   (Jaeger/Tempo) │
                              └────────▲────────┘
                                       │ gRPC/HTTP
                                       │
┌─────────────────────────────────────┴─────────────────────────────────┐
│                           LLM-Simulator                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Tracing   │  │   Metrics   │  │   Logging   │  │   Health    │  │
│  │   Layer     │  │   Registry  │  │  Subscriber │  │   Checker   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │                │          │
│         │    ┌───────────┴────────────────┴───────────┐   │          │
│         │    │         Request Handler                 │   │          │
│         └────┤  - Creates spans per request            │───┘          │
│              │  - Records metrics with labels          │              │
│              │  - Logs with trace context              │              │
│              │  - Reports component health             │              │
│              └─────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Jaeger UI     │   │   Prometheus    │   │   Log Aggregator │
│   (Traces)      │   │   (Metrics)     │   │   (ELK/Loki)    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Health Check Response Schema

```json
{
  "status": "healthy|degraded|unhealthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "timestamp": "2025-11-26T12:00:00Z",
  "checks": {
    "engine": {
      "status": "pass",
      "latency_ms": 2
    },
    "config": {
      "status": "pass"
    },
    "metrics": {
      "status": "pass"
    },
    "memory": {
      "status": "warn",
      "message": "Memory usage at 75%",
      "value": 75
    },
    "tracing": {
      "status": "pass",
      "message": "Connected to OTLP endpoint"
    }
  }
}
```

### Metrics Schema

```prometheus
# Existing metrics - now with labels
llm_simulator_requests_total{provider="openai",model="gpt-4",status="success"} 1234
llm_simulator_request_duration_seconds{provider="openai",model="gpt-4",quantile="0.99"} 2.5
llm_simulator_tokens_input_total{provider="openai",model="gpt-4"} 50000
llm_simulator_tokens_output_total{provider="openai",model="gpt-4"} 25000
llm_simulator_errors_total{provider="openai",model="gpt-4",error_type="rate_limit"} 10

# NEW: Queue metrics (referenced in alerts)
llm_simulator_queue_depth 42
llm_simulator_queue_capacity 1000

# NEW: Cost tracking
llm_simulator_cost_dollars_total{provider="openai",model="gpt-4"} 12.50

# NEW: Cache metrics
llm_simulator_cache_hits_total{cache="response"} 500
llm_simulator_cache_misses_total{cache="response"} 100
```

---

## R - Refinement

### Edge Cases

1. **OTLP Endpoint Unavailable**
   - Continue operation without tracing (degraded mode)
   - Log warning, don't fail startup
   - Report in health check as degraded

2. **High Cardinality Labels**
   - Limit model label to known models
   - Use "unknown" for unrecognized models
   - Monitor label cardinality

3. **Memory Pressure from Trace Buffering**
   - Configure batch export with limits
   - Drop traces under memory pressure
   - Alert on trace drop rate

4. **Health Check During Startup**
   - Return 503 during initialization
   - Startup probe should tolerate initial failures
   - Track initialization progress

---

## C - Completion

### Definition of Done

- [ ] `init_otel()` called in `run_server()` when endpoint configured
- [ ] Spans created for all HTTP request types
- [ ] W3C trace context propagation working
- [ ] Health check verifies engine, config, metrics status
- [ ] Readiness check returns false when unhealthy
- [ ] `queue_depth` and `queue_capacity` metrics implemented
- [ ] Provider and model labels on all request metrics
- [ ] Log entries include trace_id when tracing enabled
- [ ] All alert rules reference existing metrics
- [ ] Documentation updated with observability guide

### Verification Checklist

```bash
# 1. Verify tracing is enabled
curl -s http://localhost:8080/health | jq '.checks.tracing'
# Expected: {"status": "pass", "message": "Connected to OTLP endpoint"}

# 2. Verify health check logic
curl -s http://localhost:8080/health | jq '.status'
# Expected: "healthy", "degraded", or "unhealthy"

# 3. Verify new metrics
curl -s http://localhost:8080/metrics | grep queue_depth
# Expected: llm_simulator_queue_depth 0

# 4. Verify log-trace correlation
docker logs llm-simulator 2>&1 | grep trace_id
# Expected: Logs with trace_id="..." field
```

---

# PHASE 3: TESTING

## S - Specification

### Overview
Implement comprehensive testing infrastructure to achieve production-grade reliability through integration tests, property-based testing, and improved coverage across all modules.

### Objectives
1. Create integration test suite (minimum 30 tests)
2. Implement property-based tests with proptest
3. Add mock-based failure scenario tests
4. Complete streaming edge case coverage
5. Achieve 70% code coverage minimum

### Requirements

#### 3.1 Integration Tests
- **MUST** test full HTTP request/response cycles
- **MUST** cover all three providers (OpenAI, Anthropic, Google)
- **MUST** test streaming endpoints end-to-end
- **MUST** test error responses match API specs
- **SHOULD** test concurrent request handling
- **SHOULD** test configuration hot-reload scenarios

#### 3.2 Property-Based Tests
- **MUST** test latency distribution statistical properties
- **MUST** test request/response serialization roundtrips
- **MUST** test token estimation accuracy
- **SHOULD** test configuration validation exhaustively
- **SHOULD** test deterministic behavior with seeds

#### 3.3 Failure Scenario Tests
- **MUST** test rate limiting behavior
- **MUST** test circuit breaker state transitions
- **MUST** test chaos injection scenarios
- **SHOULD** test graceful degradation under load
- **SHOULD** test recovery after failures

#### 3.4 Streaming Tests
- **MUST** test stream completion with all providers
- **MUST** test stream interruption handling
- **MUST** test SSE format correctness
- **SHOULD** test large token count streaming
- **SHOULD** test keep-alive behavior

#### 3.5 Coverage Requirements
- **MUST** achieve 70% overall line coverage
- **MUST** achieve 80% coverage on critical paths (engine, handlers)
- **SHOULD** achieve 90% coverage on security module
- **MUST** fail CI if coverage drops below threshold

### Success Criteria
- Integration test directory populated with 30+ tests
- Property tests validate statistical properties
- All failure scenarios have test coverage
- Coverage gate enforced in CI
- No regressions in existing tests

---

## P - Pseudocode

### 3.1 Integration Test Framework

```
// tests/integration/common/mod.rs
MODULE test_common:
    STRUCT TestServer:
        addr: SocketAddr
        client: reqwest::Client
        shutdown_tx: oneshot::Sender<()>

    FUNCTION spawn_test_server(config: Option<SimulatorConfig>) -> TestServer:
        config = config.unwrap_or(SimulatorConfig::default())

        // Find available port
        port = find_available_port()
        config.server.port = port

        // Create shutdown channel
        (shutdown_tx, shutdown_rx) = oneshot::channel()

        // Spawn server in background
        tokio::spawn(async move {
            run_server_with_shutdown(config, shutdown_rx).await
        })

        // Wait for server to be ready
        wait_for_health(format!("http://127.0.0.1:{}/health", port))

        RETURN TestServer {
            addr: format!("127.0.0.1:{}", port).parse(),
            client: reqwest::Client::new(),
            shutdown_tx,
        }
```

### 3.2 Property-Based Tests

```
// tests/property/latency_test.rs
use proptest::prelude::*;

proptest! {
    // Test that normal distribution produces values within expected range
    #[test]
    fn test_normal_distribution_bounds(
        mean in 10.0..1000.0f64,
        std_dev in 1.0..100.0f64,
        samples in 100..1000usize,
    ) {
        let config = LatencyDistribution::Normal { mean_ms: mean, std_dev_ms: std_dev };
        let sampler = DistributionSampler::new(&config, None);

        let values: Vec<f64> = (0..samples).map(|_| sampler.sample()).collect();

        // Statistical validation
        let actual_mean = values.iter().sum::<f64>() / values.len() as f64;

        // Mean should be within 20% of configured (with enough samples)
        prop_assert!((actual_mean - mean).abs() / mean < 0.2);

        // All values should be positive
        prop_assert!(values.iter().all(|&v| v >= 0.0));
    }

    // Test serialization roundtrip
    #[test]
    fn test_chat_request_roundtrip(
        model in "[a-z]{3,10}-[0-9]",
        message_count in 1..10usize,
        temperature in prop::option::of(0.0..2.0f32),
    ) {
        let messages: Vec<Message> = (0..message_count)
            .map(|i| Message::user(format!("Message {}", i)))
            .collect();

        let request = ChatCompletionRequest {
            model,
            messages,
            temperature,
            ..Default::default()
        };

        // Serialize to JSON
        let json = serde_json::to_string(&request).unwrap();

        // Deserialize back
        let parsed: ChatCompletionRequest = serde_json::from_str(&json).unwrap();

        // Verify equality
        prop_assert_eq!(request.model, parsed.model);
        prop_assert_eq!(request.messages.len(), parsed.messages.len());
        prop_assert_eq!(request.temperature, parsed.temperature);
    }
}
```

---

## A - Architecture

### Test Directory Structure

```
tests/
├── integration/
│   ├── mod.rs                  # Test module declarations
│   ├── common/
│   │   ├── mod.rs              # Shared test utilities
│   │   ├── server.rs           # Test server spawning
│   │   ├── assertions.rs       # Custom assertions
│   │   └── fixtures.rs         # Request/response fixtures
│   ├── openai_test.rs          # OpenAI endpoint tests
│   ├── anthropic_test.rs       # Anthropic endpoint tests
│   ├── google_test.rs          # Google/Gemini endpoint tests
│   ├── streaming_test.rs       # SSE streaming tests
│   ├── failure_test.rs         # Error and chaos tests
│   ├── concurrent_test.rs      # Concurrency tests
│   └── config_test.rs          # Configuration tests
├── property/
│   ├── mod.rs
│   ├── latency_test.rs         # Latency distribution properties
│   ├── serialization_test.rs   # Roundtrip properties
│   └── determinism_test.rs     # Seed-based properties
└── fixtures/
    ├── requests/               # Sample request JSON files
    └── responses/              # Expected response JSON files
```

### Test Count Targets

| Category | Target | Minimum |
|----------|--------|---------|
| OpenAI endpoint tests | 10 | 8 |
| Anthropic endpoint tests | 6 | 4 |
| Google endpoint tests | 6 | 4 |
| Streaming tests | 8 | 6 |
| Failure scenario tests | 6 | 4 |
| Property tests | 10 | 6 |
| Configuration tests | 4 | 2 |
| **Total** | **50** | **34** |

---

## C - Completion

### Definition of Done

- [ ] Integration test directory created with 30+ tests
- [ ] All three providers have endpoint tests
- [ ] Streaming tests cover all providers
- [ ] Property tests implemented with proptest
- [ ] Failure scenario tests for chaos/rate limiting
- [ ] Coverage report generated and uploaded
- [ ] Coverage gate of 70% enforced in CI
- [ ] All tests pass in CI environment
- [ ] Test documentation added to CONTRIBUTING.md

### Verification Checklist

```bash
# 1. Run all tests
cargo test --all

# 2. Run integration tests only
cargo test --test '*'

# 3. Run property tests with verbose output
cargo test --test property_* -- --nocapture

# 4. Generate coverage report
cargo tarpaulin --out Html --open

# 5. Verify coverage threshold
cargo tarpaulin --fail-under 70
```

---

# PHASE 4: OPERATIONS

## S - Specification

### Overview
Implement production-grade operational capabilities including backup/recovery, secrets management, disaster recovery planning, and operational documentation to ensure reliable and maintainable production deployments.

### Objectives
1. Implement automated backup strategy with Velero
2. Integrate secrets management (HashiCorp Vault / Cloud KMS)
3. Define and document RTO/RPO objectives
4. Create disaster recovery procedures
5. Develop operational runbooks

### Requirements

#### 4.1 Backup Strategy
- **MUST** implement Velero for Kubernetes backup
- **MUST** backup PersistentVolumeClaims for StatefulSet deployments
- **MUST** schedule daily backups with 30-day retention
- **SHOULD** support cross-region backup replication
- **SHOULD** automate backup verification/restore testing
- **MUST** document backup recovery procedures

#### 4.2 Secrets Management
- **MUST** integrate with at least one secrets provider (Vault/AWS/Azure/GCP)
- **MUST** remove all hardcoded secrets from configuration
- **MUST** support automatic secret rotation
- **SHOULD** use Kubernetes External Secrets Operator
- **SHOULD** encrypt secrets at rest and in transit
- **MUST** audit secret access

#### 4.3 Disaster Recovery
- **MUST** define RTO (Recovery Time Objective) ≤ 1 hour
- **MUST** define RPO (Recovery Point Objective) ≤ 15 minutes
- **MUST** document failover procedures
- **SHOULD** implement automated failover for critical components
- **SHOULD** test DR procedures quarterly
- **MUST** maintain DR runbook

#### 4.4 Connection Draining
- **MUST** implement graceful shutdown with request completion
- **MUST** configure preStop hooks for Kubernetes
- **SHOULD** track in-flight requests during shutdown
- **SHOULD** support configurable drain timeout
- **MUST** disable readiness before draining

#### 4.5 Operational Documentation
- **MUST** create runbooks for common operations
- **MUST** document incident response procedures
- **MUST** create troubleshooting guides
- **SHOULD** implement operational dashboards
- **SHOULD** create capacity planning documentation

### Success Criteria
- Automated daily backups running and verified
- Secrets injected from external provider (no hardcoded values)
- DR procedures documented and tested
- Graceful shutdown completes all in-flight requests
- Runbooks cover 90% of operational scenarios

---

## P - Pseudocode

### 4.1 Connection Draining Implementation

```
// src/server/shutdown.rs

STRUCT ShutdownState:
    in_flight_requests: AtomicU64
    draining: AtomicBool
    drain_timeout: Duration

IMPL ShutdownState:
    FUNCTION new(drain_timeout: Duration) -> Self:
        Self {
            in_flight_requests: AtomicU64::new(0),
            draining: AtomicBool::new(false),
            drain_timeout,
        }

    FUNCTION request_started(&self):
        self.in_flight_requests.fetch_add(1, Ordering::SeqCst)

    FUNCTION request_completed(&self):
        self.in_flight_requests.fetch_sub(1, Ordering::SeqCst)

    FUNCTION is_draining(&self) -> bool:
        self.draining.load(Ordering::SeqCst)

    FUNCTION start_drain(&self):
        self.draining.store(true, Ordering::SeqCst)

// Request tracking middleware
FUNCTION request_tracking_middleware(state: ShutdownState, request, next):
    // Reject new requests if draining
    IF state.is_draining():
        RETURN error_response(503, "Service is shutting down")

    // Track request start
    state.request_started()

    // Execute request
    response = next(request).await

    // Track request completion
    state.request_completed()

    RETURN response

// Enhanced shutdown handler
FUNCTION graceful_shutdown(state: ShutdownState, server_handle):
    // Step 1: Mark as draining (stop accepting new requests)
    log.info("Starting graceful shutdown, marking as draining")
    state.start_drain()

    // Step 2: Disable readiness check
    set_not_ready()

    // Step 3: Wait for in-flight requests with timeout
    drain_start = Instant::now()

    WHILE state.in_flight_count() > 0:
        IF drain_start.elapsed() > state.drain_timeout:
            log.warn("Drain timeout exceeded, forcing shutdown",
                remaining_requests=state.in_flight_count())
            BREAK

        log.info("Waiting for in-flight requests",
            count=state.in_flight_count(),
            elapsed=drain_start.elapsed())

        sleep(Duration::from_millis(100))

    // Step 4: Shutdown server
    log.info("All requests drained, shutting down server")
    server_handle.graceful_shutdown(None)
```

---

## A - Architecture

### Backup Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Backup Architecture                             │
└─────────────────────────────────────────────────────────────────────┘

  Primary Region (us-west-2)              Backup Region (us-east-1)
  ┌─────────────────────────┐            ┌─────────────────────────┐
  │   LLM-Simulator         │            │   LLM-Simulator         │
  │   Namespace             │            │   (Standby)             │
  │                         │            │                         │
  │  ┌─────────────────┐   │            │                         │
  │  │  StatefulSet    │   │            │                         │
  │  │  + PVC (50Gi)   │   │            │                         │
  │  └────────┬────────┘   │            │                         │
  │           │             │            │                         │
  │  ┌────────▼────────┐   │            │                         │
  │  │   Velero Agent  │   │            │                         │
  │  └────────┬────────┘   │            │                         │
  └───────────┼─────────────┘            └─────────────────────────┘
              │
              │ Daily backup (2 AM UTC)
              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    S3 / Blob Storage                             │
  │  Cross-Region Replication ─────────────────────────────────────▶ │
  └─────────────────────────────────────────────────────────────────┘
```

### Connection Draining Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Graceful Shutdown Sequence                        │
└─────────────────────────────────────────────────────────────────────┘

  T+0: SIGTERM received
  │
  ├── preStop hook executes (POST /admin/drain)
  │
  ├── Server marks draining=true
  │   └── Readiness probe returns false
  │
  ├── Kubernetes removes pod from Service endpoints
  │   └── No new traffic routed to pod
  │
  ├── In-flight request tracking
  │   └── In-flight: 50 → 30 → 15 → 5 → 0
  │
  ├── All requests drained (or timeout at T+55s)
  │
  └── T+60s: terminationGracePeriodSeconds expires

  Timeline:
  0s        5s         30s        55s        60s
  │         │          │          │          │
  SIGTERM   preStop    Draining   Drain      SIGKILL
            complete   in-flight  complete   (forced)
```

### RTO/RPO Documentation

| Metric | Target | Measurement |
|--------|--------|-------------|
| **RTO** | ≤ 1 hour | Time from incident to service restoration |
| **RPO** | ≤ 15 minutes | Maximum data loss (backup frequency) |
| **Backup Frequency** | Daily + hourly incremental | Full daily, incremental hourly |
| **Backup Retention** | 30 days | Oldest recoverable point |
| **Failover Time** | ≤ 15 minutes | Automated failover duration |
| **Failback Time** | ≤ 2 hours | Manual with gradual traffic shift |

---

## C - Completion

### Definition of Done

- [ ] Velero installed and configured in cluster
- [ ] Daily backup schedule running
- [ ] Backup verification automated (weekly)
- [ ] External Secrets Operator deployed
- [ ] Secrets syncing from Vault/cloud provider
- [ ] Secret rotation tested and documented
- [ ] RTO/RPO defined and documented
- [ ] Failover procedure documented and tested
- [ ] Connection draining implemented
- [ ] preStop hook configured in deployment
- [ ] Runbooks created for common operations
- [ ] Incident response procedure documented
- [ ] DR test completed successfully

### Verification Checklist

```bash
# 1. Verify Velero installation
velero version
velero backup-location get

# 2. Verify backup schedule
velero schedule get
velero backup get | head -5

# 3. Verify External Secrets
kubectl get externalsecrets -n llm-simulator
kubectl get secrets llm-simulator-api-keys -n llm-simulator

# 4. Test graceful shutdown
kubectl exec -it <pod> -- curl -X POST localhost:8080/admin/drain
kubectl logs <pod> | grep -i drain

# 5. Test failover (DR drill)
./deploy/scripts/failover.sh --dry-run
```

---

# APPENDIX

## A. File Modification Summary

### Phase 1: Security
| File | Action | Description |
|------|--------|-------------|
| `src/security/mod.rs` | Create | Module exports |
| `src/security/api_key.rs` | Create | API key validation |
| `src/security/rate_limit.rs` | Create | Token bucket |
| `src/security/cors.rs` | Create | CORS builder |
| `src/security/headers.rs` | Create | Security headers |
| `src/server/middleware.rs` | Modify | Add security middleware |
| `src/config/security.rs` | Create | Security config |

### Phase 2: Observability
| File | Action | Description |
|------|--------|-------------|
| `src/telemetry/mod.rs` | Modify | Call init_otel() |
| `src/telemetry/metrics.rs` | Modify | Add queue/cost metrics |
| `src/telemetry/tracing.rs` | Create | Span utilities |
| `src/server/handlers.rs` | Modify | Real health checks |

### Phase 3: Testing
| File | Action | Description |
|------|--------|-------------|
| `tests/integration/*.rs` | Create | Integration tests |
| `tests/property/*.rs` | Create | Property tests |
| `codecov.yml` | Create | Coverage config |

### Phase 4: Operations
| File | Action | Description |
|------|--------|-------------|
| `deploy/velero/*.yaml` | Create | Backup config |
| `deploy/external-secrets/*.yaml` | Create | Secrets config |
| `src/server/shutdown.rs` | Create | Connection draining |
| `docs/runbooks/*.md` | Create | Runbooks |

## B. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Security breach via unauthenticated access | HIGH | CRITICAL | Phase 1 implementation |
| Silent failures due to broken tracing | MEDIUM | MEDIUM | Phase 2 implementation |
| Integration failures from untested paths | MEDIUM | MEDIUM | Phase 3 implementation |
| Data loss from missing backups | LOW | HIGH | Phase 4 implementation |

## C. Success Metrics

| Metric | Current | Target | Phase |
|--------|---------|--------|-------|
| Security Score | 3.5/10 | 9/10 | 1 |
| Auth bypass vulnerabilities | 5 | 0 | 1 |
| Trace coverage | 0% | 100% | 2 |
| Health check accuracy | 0% | 100% | 2 |
| Test coverage | ~40% | 70%+ | 3 |
| Integration tests | 0 | 30+ | 3 |
| Backup success rate | 0% | 99%+ | 4 |
| RTO achievement | N/A | ≤1 hour | 4 |

---

**Document Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Technical Lead | | | |
| Security Reviewer | | | |
| Operations Lead | | | |
| Project Manager | | | |
