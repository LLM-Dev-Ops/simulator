# LLM-Simulator: Configuration Management System Design

## Executive Summary

This document describes the enterprise-grade configuration management system for LLM-Simulator, providing flexible, type-safe configuration with multi-format support, hot-reload capabilities, comprehensive validation, and automatic migration.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Sources                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   CLI    │  │   ENV    │  │  Local   │  │   Main   │   │
│  │   Args   │  │   Vars   │  │  Config  │  │  Config  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │             │          │
│       └─────────────┴──────────────┴─────────────┘          │
│                          │                                   │
│                    ┌─────▼──────┐                           │
│                    │   Merger   │                           │
│                    └─────┬──────┘                           │
│                          │                                   │
│                    ┌─────▼──────┐                           │
│                    │ Validator  │                           │
│                    └─────┬──────┘                           │
│                          │                                   │
│                    ┌─────▼──────┐                           │
│                    │  Migrator  │                           │
│                    └─────┬──────┘                           │
│                          │                                   │
│                  ┌───────▼────────┐                         │
│                  │ Final Config   │                         │
│                  └────────────────┘                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Hot-Reload System                         │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │ File Watcher │───────▶│  Validator   │                  │
│  └──────────────┘        └──────┬───────┘                  │
│                                  │                           │
│                          ┌───────▼────────┐                 │
│                          │ Atomic Update  │                 │
│                          └───────┬────────┘                 │
│                                  │                           │
│                          ┌───────▼────────┐                 │
│                          │   Callbacks    │                 │
│                          └────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Multi-Format Support

**Supported Formats:**
- YAML (primary, human-friendly)
- JSON (API-friendly, strict)
- TOML (Rust-native alternative)

**Automatic Format Detection:**
```rust
ConfigFormat::from_path(path)  // Detects from extension
```

### 2. Hierarchical Configuration

**Precedence Order (Highest → Lowest):**
1. CLI Arguments
2. Environment Variables
3. Local Config File (`simulator.local.yaml`)
4. Main Config File (`simulator.yaml`)
5. Built-in Defaults

**Example:**
```rust
let config = loader.load_with_overrides(cli_args)?;
// Automatically merges all sources in correct order
```

### 3. Comprehensive Schema

**Configuration Sections:**
- **Server:** Host, port, TLS, CORS, rate limiting
- **Providers:** Latency profiles, error injection, response behavior
- **Simulation:** Concurrency, sessions, resource limits
- **Telemetry:** Logging, metrics, distributed tracing
- **Scenarios:** Load testing patterns, assertions
- **Security:** Authentication, API keys, IP filtering
- **Features:** Feature flags for capabilities

### 4. Type-Safe Validation

**Validation Features:**
- Field-level constraints
- Cross-field validation
- Detailed error messages
- Batch error reporting
- Constraint documentation

**Example Error Output:**
```
❌ Configuration validation failed with 3 error(s):

1. Field: server.port
   Message: Port must be greater than 0
   Constraint: 1-65535
   Actual: 0

2. Field: providers.gpt-4.latency.ttft.p99_ms
   Message: p99 must be greater than p50
   Constraint: > 800.0
   Actual: 500.0

3. Field: telemetry.logging.level
   Message: Invalid log level
   Constraint: One of: trace, debug, info, warn, error
   Actual: invalid
```

### 5. Environment Variable Mapping

**Automatic Mapping with Prefix:**
```bash
# Environment variables
export LLM_SIM_HOST=0.0.0.0
export LLM_SIM_PORT=9090
export LLM_SIM_LOG_LEVEL=debug
export LLM_SIM_METRICS_ENABLED=true

# Automatically mapped to config structure
```

**Type-Safe Parsing:**
- Strings, integers, floats, booleans
- Paths, URLs, durations
- Helpful error messages on parse failures

### 6. Hot-Reload System

**Features:**
- File system polling
- Thread-safe atomic updates
- Validation before applying
- Callback notifications
- Configurable poll interval

**Usage:**
```rust
let watcher = ConfigWatcher::new(config_path, initial_config, loader)?
    .with_poll_interval(Duration::from_secs(5));

let handle = watcher.watch(|new_config| {
    println!("Configuration reloaded!");
    // Apply new config to running system
});
```

### 7. Configuration Migration

**Version Tracking:**
```yaml
version: "1.0"  # Schema version
```

**Automatic Migration:**
```rust
config = ConfigMigrator::migrate(config)?;
// Automatically upgrades from older versions
```

**Migration Chain:**
```
0.9 → 1.0 → 1.1 → 2.0 (current)
```

## Configuration Schema Reference

### Server Configuration

```yaml
server:
  host: "127.0.0.1"           # Bind address
  port: 8080                   # Listen port
  max_connections: 10000       # Concurrent connection limit
  request_timeout_secs: 300    # Request timeout
  keepalive_timeout_secs: 75   # Keep-alive timeout

  tls:                         # Optional TLS
    cert_path: "/path/to/cert.pem"
    key_path: "/path/to/key.pem"
    min_version: "1.2"         # 1.2 or 1.3
    client_auth: "none"        # none, optional, required

  cors:
    enabled: true
    allowed_origins: ["*"]
    allowed_methods: ["GET", "POST", "OPTIONS"]
    allowed_headers: ["Content-Type", "Authorization"]
    max_age_secs: 3600

  rate_limit:
    enabled: true
    requests_per_second: 100
    burst_capacity: 200
    rate_limit_header: "X-API-Key"  # Optional
```

### Provider Profile Configuration

```yaml
providers:
  gpt-4-turbo:
    name: "GPT-4 Turbo"
    provider: "openai"
    model: "gpt-4-turbo"
    enabled: true

    latency:
      ttft:                    # Time to First Token
        type: "log_normal"
        p50_ms: 800.0
        p99_ms: 2500.0

      itl:                     # Inter-Token Latency
        type: "normal"
        mean_ms: 20.0
        std_dev_ms: 5.0
        min_clamp_ms: 0.0

      network_jitter:          # Optional jitter
        type: "normal"
        mean_ms: 0.0
        std_dev_ms: 2.0

      degradation:             # Load degradation
        type: "exponential"
        alpha: 0.5
        baseline_qps: 10.0

      validation_metrics:      # Optional validation
        ttft_p50_ms: 800.0
        ttft_p99_ms: 2500.0
        itl_mean_ms: 20.0
        itl_p99_ms: 35.0
        tokens_per_sec: 50.0
        measurement_date: "2024-01"
        sample_size: 1000

    errors:
      enabled: false
      base_rate: 0.01          # 1% error rate
      error_types:
        rate_limit:
          probability: 0.6     # 60% of errors
          status_code: 429
          message: "Rate limit exceeded"
          retry_after: 60
        server_error:
          probability: 0.4     # 40% of errors
          status_code: 500
          message: "Internal server error"

      chaos_scenarios:
        - name: "peak_load_failure"
          trigger:
            type: "load"
            qps_threshold: 50.0
          duration_secs: 300
          error_rate: 0.2
          latency_multiplier: 3.0

    response:
      content_strategy: "random"  # random, pattern, markov, echo, template
      max_tokens: 4096
      vocab_size: 50000
      streaming_chunk_size: 1
      include_usage: true

    cost:
      input_tokens_per_million: 10.0
      output_tokens_per_million: 30.0
      cached_tokens_per_million: 5.0
      include_in_response: true
```

### Distribution Types

**Normal Distribution:**
```yaml
type: "normal"
mean_ms: 100.0
std_dev_ms: 20.0
min_clamp_ms: 0.0  # Optional
```

**Log-Normal Distribution:**
```yaml
type: "log_normal"
p50_ms: 100.0     # Median
p99_ms: 500.0     # 99th percentile
```

**Exponential Distribution:**
```yaml
type: "exponential"
mean_ms: 50.0
```

**Bimodal Distribution (Cache Hit/Miss):**
```yaml
type: "bimodal"
fast_mean_ms: 10.0      # Cache hit
fast_std_ms: 2.0
slow_mean_ms: 100.0     # Cache miss
slow_std_ms: 20.0
fast_probability: 0.9   # 90% hit rate
```

**Empirical Distribution (Real Data):**
```yaml
type: "empirical"
samples_ms: [100.0, 150.0, 200.0, 180.0, 220.0]
interpolation: "linear"  # linear or nearest
```

**Constant Distribution:**
```yaml
type: "constant"
value_ms: 100.0
```

### Degradation Models

**Linear Degradation:**
```yaml
degradation:
  type: "linear"
  slope: 0.3
  baseline_qps: 10.0
# latency_multiplier = 1 + 0.3 * (qps / baseline_qps)
```

**Exponential Degradation:**
```yaml
degradation:
  type: "exponential"
  alpha: 0.5
  baseline_qps: 10.0
# latency_multiplier = exp(0.5 * qps / baseline_qps)
```

**Queueing Theory (M/M/1):**
```yaml
degradation:
  type: "mm_one"
  service_rate: 100.0  # Max requests/sec
  baseline_qps: 10.0
# latency_multiplier = 1 / (1 - utilization)
```

**Piecewise Linear:**
```yaml
degradation:
  type: "piecewise"
  baseline_qps: 10.0
  breakpoints:
    - qps: 10.0
      multiplier: 1.0
    - qps: 50.0
      multiplier: 2.0
    - qps: 100.0
      multiplier: 5.0
```

### Simulation Configuration

```yaml
simulation:
  seed: 42                        # Random seed (optional)
  max_concurrent_sessions: 1000
  session_timeout_secs: 3600
  persist_sessions: false
  session_storage_path: "/var/lib/llm-simulator/sessions"

  concurrency:
    worker_threads: 0             # 0 = auto-detect
    blocking_threads: 512
    task_queue_size: 10000
    backpressure: "reject"        # drop, block, reject

  limits:
    max_memory_mb: 4096
    max_cpu_percent: 80
    max_disk_mb: 10240
    max_body_size_mb: 10
    max_response_size_mb: 100

  warmup:
    enabled: true
    requests_per_profile: 100
    duration_secs: 30
```

### Telemetry Configuration

```yaml
telemetry:
  enabled: true
  service_name: "llm-simulator"

  logging:
    level: "info"                 # trace, debug, info, warn, error
    format: "json"                # json, text, pretty
    output: "stdout"              # stdout, stderr, file
    file_path: "/var/log/llm-simulator.log"

    rotation:
      max_size_mb: 100
      max_backups: 10
      compress: true

    include_location: false
    include_thread_id: false

  metrics:
    enabled: true
    exporter: "prometheus"        # prometheus, otlp, statsd
    endpoint: "/metrics"
    interval_secs: 60

    histogram_buckets:
      - 0.001
      - 0.005
      - 0.01
      - 0.025
      - 0.05
      - 0.1
      - 0.25
      - 0.5
      - 1.0
      - 2.5
      - 5.0
      - 10.0

    custom_metrics:
      - name: "custom_counter"
        metric_type: "counter"
        description: "Custom counter metric"
        labels: ["provider", "model"]

  tracing:
    enabled: true
    exporter: "otlp"              # otlp, jaeger, zipkin
    otlp_endpoint: "http://localhost:4317"
    sampling_rate: 1.0            # 0.0 - 1.0
    trace_id_strategy: "random"   # random, deterministic
    propagation_format: "w3c"     # w3c, b3, jaeger
```

### Scenario Configuration

```yaml
scenarios:
  load_test:
    name: "Load Test"
    description: "Constant load test"
    enabled: true
    profile: "gpt-4-turbo"

    pattern:
      type: "constant"
      qps: 10.0
      duration_secs: 300

    assertions:
      - metric: "latency_p99"
        operator: "less_than"
        threshold: 3000.0         # ms
      - metric: "error_rate"
        operator: "less_than"
        threshold: 0.01           # 1%

    tags: ["load-test", "regression"]

  ramp_test:
    name: "Ramp Test"
    enabled: true
    profile: "gpt-4-turbo"

    pattern:
      type: "ramp"
      start_qps: 1.0
      end_qps: 100.0
      duration_secs: 600

  spike_test:
    name: "Spike Test"
    enabled: true
    profile: "gpt-4-turbo"

    pattern:
      type: "spike"
      baseline_qps: 10.0
      spike_qps: 100.0
      spike_duration_secs: 60
      total_duration_secs: 300

  poisson_test:
    name: "Poisson Traffic"
    enabled: true
    profile: "gpt-4-turbo"

    pattern:
      type: "poisson"
      mean_qps: 20.0
      duration_secs: 300
```

### Security Configuration

```yaml
security:
  authentication:
    method: "api_key"             # api_key, jwt, oauth2

    jwt:
      secret: "your-secret-key"
      algorithm: "HS256"          # HS256, RS256, etc.
      expiration_secs: 3600
      issuer: "llm-simulator"
      audience: "api-clients"

    oauth2:
      provider: "custom"
      client_id: "your-client-id"
      client_secret: "your-secret"
      auth_url: "https://auth.example.com/authorize"
      token_url: "https://auth.example.com/token"
      scopes: ["read", "write"]

  api_keys:
    - key: "sk-test-key-1"
      name: "Test Key 1"
      permissions: ["read", "write"]
      expires_at: "2025-12-31T23:59:59Z"
      rate_limit:
        requests_per_second: 50
        burst_capacity: 100

  ip_allowlist:
    - "192.168.1.0/24"
    - "10.0.0.0/8"

  ip_blocklist:
    - "1.2.3.4"

  request_signing: true
  signing_secret: "your-signing-secret"
```

### Feature Flags

```yaml
features:
  streaming: true
  function_calling: true
  vision: false
  embeddings: true
  fine_tuning: false
  batch_processing: true
  caching: true
  prompt_caching: true

  custom:
    experimental_features: false
    debug_mode: false
```

## Usage Examples

### Basic Configuration Loading

```rust
use config_system::*;

// Load configuration from default locations
let loader = ConfigLoader::new();
let config = loader.load()?;

// Validate
config.validate()?;

println!("Server: {}:{}", config.server.host, config.server.port);
```

### Loading with Overrides

```rust
// Load with CLI arguments and env vars
let cli_args = CliArgs {
    port: Some(9090),
    log_level: Some("debug".to_string()),
    ..Default::default()
};

let config = loader.load_with_overrides(Some(&cli_args))?;
```

### Hot-Reload Setup

```rust
let watcher = ConfigWatcher::new(
    PathBuf::from("config/simulator.yaml"),
    initial_config,
    loader,
)?;

let config_ref = watcher.get_config_ref();

// Spawn background watcher
let handle = watcher.watch(|new_config| {
    println!("Config reloaded: {} providers", new_config.providers.len());
    // Trigger application reconfiguration
});

// Access current config from any thread
let current = config_ref.read().unwrap();
println!("Current port: {}", current.server.port);
```

### Validation Error Handling

```rust
match config.validate() {
    Ok(_) => {
        println!("Configuration is valid");
    }
    Err(errors) => {
        print_validation_errors(&errors);
        std::process::exit(1);
    }
}
```

### Programmatic Configuration

```rust
let mut config = SimulatorConfig::default();
config.server.port = 8080;
config.server.host = "0.0.0.0".to_string();

// Add provider profile
config.providers.insert(
    "gpt-4-turbo".to_string(),
    ProviderProfile {
        name: "GPT-4 Turbo".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4-turbo".to_string(),
        enabled: true,
        latency: LatencyProfileConfig {
            ttft: DistributionConfig::LogNormal {
                p50_ms: 800.0,
                p99_ms: 2500.0,
            },
            itl: DistributionConfig::Normal {
                mean_ms: 20.0,
                std_dev_ms: 5.0,
                min_clamp_ms: Some(0.0),
            },
            network_jitter: None,
            degradation: DegradationConfig::default(),
            validation_metrics: None,
        },
        errors: ErrorConfig::default(),
        response: ResponseConfig::default(),
        cost: Some(CostConfig {
            input_tokens_per_million: 10.0,
            output_tokens_per_million: 30.0,
            cached_tokens_per_million: None,
            include_in_response: true,
        }),
        metadata: HashMap::new(),
    },
);

config.validate()?;
```

## Environment Variable Reference

### Server Variables
```bash
LLM_SIM_HOST=0.0.0.0
LLM_SIM_PORT=9090
LLM_SIM_MAX_CONNECTIONS=5000
LLM_SIM_REQUEST_TIMEOUT=300
```

### TLS Variables
```bash
LLM_SIM_TLS_CERT=/path/to/cert.pem
LLM_SIM_TLS_KEY=/path/to/key.pem
```

### Simulation Variables
```bash
LLM_SIM_SEED=42
LLM_SIM_MAX_SESSIONS=1000
```

### Telemetry Variables
```bash
LLM_SIM_LOG_LEVEL=debug
LLM_SIM_LOG_FORMAT=json
LLM_SIM_METRICS_ENABLED=true
LLM_SIM_METRICS_ENDPOINT=/metrics
```

## CLI Arguments

```bash
./llm-simulator \
  --config /path/to/config.yaml \
  --host 0.0.0.0 \
  --port 9090 \
  --log-level debug \
  --enable-metrics \
  --enable-tracing \
  --seed 42
```

## Migration Guide

### Migrating from 0.9 to 1.0

**Automatic Migration:**
```rust
let config = ConfigMigrator::migrate(old_config)?;
```

**Breaking Changes:**
- `server.tls` structure changed (now requires `client_auth` field)
- `features` section added (defaults to backward-compatible values)
- `telemetry.tracing` added (opt-in)

**Manual Steps:**
1. Update `version` field to `"1.0"`
2. Add `client_auth: "none"` to TLS config if using TLS
3. Review new feature flags in `features` section

## Best Practices

### 1. Configuration Organization

**Recommended Structure:**
```
config/
├── simulator.yaml              # Main config (committed to git)
├── simulator.local.yaml        # Local overrides (gitignored)
├── profiles/
│   ├── gpt-4-turbo.yaml
│   ├── claude-3-opus.yaml
│   └── gemini-1.5-pro.yaml
└── scenarios/
    ├── load-test.yaml
    ├── stress-test.yaml
    └── chaos-test.yaml
```

### 2. Sensitive Data

**DO NOT commit:**
- API keys
- TLS private keys
- JWT secrets
- OAuth2 credentials

**Use instead:**
- Environment variables
- Secret management service (Vault, AWS Secrets Manager)
- Local config file (gitignored)

### 3. Configuration Profiles

**Development:**
```yaml
server:
  host: "127.0.0.1"
  port: 8080

telemetry:
  logging:
    level: "debug"
    format: "pretty"
```

**Production:**
```yaml
server:
  host: "0.0.0.0"
  port: 8080
  tls:
    cert_path: "/etc/ssl/certs/simulator.pem"
    key_path: "/etc/ssl/private/simulator.key"

telemetry:
  logging:
    level: "info"
    format: "json"
    output: "file"
    file_path: "/var/log/llm-simulator.log"
```

### 4. Validation Before Deployment

```bash
# Dry-run validation
./llm-simulator --config config/simulator.yaml --validate-only

# Dump effective configuration
./llm-simulator --config config/simulator.yaml --dump-config
```

## Performance Considerations

### Configuration Loading
- **Hot Path:** Config access (via `Arc<RwLock<>>`)
- **Cold Path:** Config loading and validation
- **Optimization:** Cache parsed configs, minimize lock contention

### Hot-Reload
- **Poll Interval:** 5 seconds (configurable)
- **Overhead:** Minimal (file stat check)
- **Validation:** Full validation before applying
- **Atomicity:** RwLock ensures thread-safe updates

### Memory Usage
- **Per Config:** ~10-50 KB (depends on providers)
- **Shared:** Single config instance via Arc
- **Cloning:** Cheap (Arc clone is pointer copy)

## Troubleshooting

### Configuration Not Found

**Error:**
```
Configuration file not found: ./config/simulator.yaml
```

**Solution:**
```bash
# Specify explicit path
./llm-simulator --config /path/to/simulator.yaml

# Or set search paths in code
let loader = ConfigLoader::new()
    .with_search_paths(vec![PathBuf::from("/custom/path")]);
```

### Validation Errors

**Error:**
```
❌ Configuration validation failed with 2 error(s):

1. Field: server.port
   Message: Port must be greater than 0
   Constraint: 1-65535
   Actual: 0
```

**Solution:**
- Check field value in config file
- Refer to constraint documentation
- Fix value to meet constraint

### Hot-Reload Not Working

**Checklist:**
1. File permissions (readable by process)
2. File watcher started (`watcher.watch()`)
3. Validation passing (check logs)
4. Poll interval not too long

### Environment Variables Not Applied

**Checklist:**
1. Correct prefix (`LLM_SIM_*`)
2. Correct variable name (check reference)
3. Valid value for type (e.g., integer for port)
4. Environment loaded before config

## Security Considerations

### Secrets Management
- Never commit secrets to version control
- Use environment variables or secret management service
- Rotate secrets regularly
- Encrypt config files at rest if containing sensitive data

### Access Control
- Limit file permissions (600 for configs with secrets)
- Use TLS for remote config loading
- Implement RBAC for config management APIs
- Audit config changes

### Validation
- Always validate before applying
- Sanitize user inputs
- Check file paths for directory traversal
- Validate URLs and network addresses

## Future Enhancements

### Planned Features
- [ ] Remote configuration (etcd, Consul, AWS Systems Manager)
- [ ] Configuration as Code (Terraform provider)
- [ ] GraphQL API for configuration management
- [ ] Configuration diff and rollback
- [ ] Encrypted configuration support
- [ ] Configuration testing framework
- [ ] Auto-generate OpenAPI spec from config schema

### Under Consideration
- [ ] Dynamic configuration without restart (beyond hot-reload)
- [ ] Configuration inheritance and templates
- [ ] Configuration A/B testing
- [ ] Configuration gradual rollout
- [ ] Configuration compliance checking

## Conclusion

The LLM-Simulator configuration management system provides:

✅ **Flexibility:** Multi-format support (YAML/JSON/TOML)
✅ **Hierarchy:** CLI > Env > Local > Main > Defaults
✅ **Validation:** Comprehensive, helpful error messages
✅ **Hot-Reload:** Zero-downtime configuration updates
✅ **Migration:** Automatic version upgrades
✅ **Type-Safety:** Compile-time guarantees
✅ **Enterprise-Ready:** Production-grade reliability

For questions or issues, refer to the inline documentation in `config_system_design.rs`.
