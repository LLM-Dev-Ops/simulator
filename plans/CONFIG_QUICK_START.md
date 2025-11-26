# LLM-Simulator Configuration Quick Start

## 5-Minute Setup Guide

### 1. Create Basic Configuration

Create `config/simulator.yaml`:

```yaml
version: "1.0"

server:
  host: "127.0.0.1"
  port: 8080

providers:
  gpt-4-turbo:
    name: "GPT-4 Turbo"
    provider: "openai"
    model: "gpt-4-turbo"
    enabled: true

    latency:
      ttft:
        type: "log_normal"
        p50_ms: 800.0
        p99_ms: 2500.0

      itl:
        type: "normal"
        mean_ms: 20.0
        std_dev_ms: 5.0

      degradation:
        type: "none"
        baseline_qps: 10.0

    errors:
      enabled: false

    response:
      content_strategy: "random"
      max_tokens: 4096

telemetry:
  logging:
    level: "info"
    format: "json"

  metrics:
    enabled: true
    endpoint: "/metrics"
```

### 2. Load Configuration in Code

```rust
use config_system::*;

fn main() -> Result<(), ConfigError> {
    // Simple load
    let config = ConfigLoader::new().load()?;

    // Validate
    config.validate()?;

    // Use config
    println!("Starting server on {}:{}",
             config.server.host,
             config.server.port);

    Ok(())
}
```

### 3. Environment Variable Overrides

```bash
export LLM_SIM_PORT=9090
export LLM_SIM_LOG_LEVEL=debug
./llm-simulator
```

### 4. CLI Arguments

```bash
./llm-simulator --port 9090 --log-level debug
```

## Common Configuration Patterns

### Pattern 1: Multiple Providers

```yaml
providers:
  gpt-4-turbo:
    name: "GPT-4 Turbo"
    provider: "openai"
    model: "gpt-4-turbo"
    enabled: true
    latency:
      ttft:
        type: "log_normal"
        p50_ms: 800.0
        p99_ms: 2500.0
      itl:
        type: "normal"
        mean_ms: 20.0
        std_dev_ms: 5.0
      degradation:
        type: "exponential"
        alpha: 0.5
        baseline_qps: 10.0
    response:
      content_strategy: "random"
      max_tokens: 4096

  claude-3-opus:
    name: "Claude 3 Opus"
    provider: "anthropic"
    model: "claude-3-opus"
    enabled: true
    latency:
      ttft:
        type: "log_normal"
        p50_ms: 1200.0
        p99_ms: 3000.0
      itl:
        type: "normal"
        mean_ms: 25.0
        std_dev_ms: 6.0
      degradation:
        type: "mm_one"
        service_rate: 8.0
        baseline_qps: 5.0
    response:
      content_strategy: "random"
      max_tokens: 4096
```

### Pattern 2: Error Injection

```yaml
providers:
  gpt-4-turbo:
    name: "GPT-4 Turbo"
    # ... other config ...

    errors:
      enabled: true
      base_rate: 0.02  # 2% error rate

      error_types:
        rate_limit:
          probability: 0.7  # 70% of errors are rate limits
          status_code: 429
          message: "Rate limit exceeded. Please retry after 60 seconds."
          retry_after: 60

        server_error:
          probability: 0.3  # 30% of errors are server errors
          status_code: 500
          message: "Internal server error"
```

### Pattern 3: Chaos Engineering

```yaml
providers:
  gpt-4-turbo:
    # ... other config ...

    errors:
      enabled: true
      base_rate: 0.0  # No base errors

      chaos_scenarios:
        - name: "peak_load_degradation"
          trigger:
            type: "load"
            qps_threshold: 50.0  # Trigger at 50 QPS
          duration_secs: 300     # Last for 5 minutes
          error_rate: 0.2        # 20% errors during scenario
          latency_multiplier: 3.0  # 3x slower

        - name: "random_outage"
          trigger:
            type: "random"
            probability: 0.001   # 0.1% chance per request
          duration_secs: 60      # 1 minute outage
          error_rate: 1.0        # 100% errors
          latency_multiplier: 1.0
```

### Pattern 4: Load Testing Scenarios

```yaml
scenarios:
  baseline_test:
    name: "Baseline Performance Test"
    enabled: true
    profile: "gpt-4-turbo"

    pattern:
      type: "constant"
      qps: 10.0
      duration_secs: 300

    assertions:
      - metric: "latency_p99"
        operator: "less_than"
        threshold: 3000.0  # ms
      - metric: "error_rate"
        operator: "less_than"
        threshold: 0.01    # 1%

  stress_test:
    name: "Stress Test"
    enabled: true
    profile: "gpt-4-turbo"

    pattern:
      type: "ramp"
      start_qps: 1.0
      end_qps: 100.0
      duration_secs: 600

    assertions:
      - metric: "latency_p95"
        operator: "less_than"
        threshold: 5000.0
```

### Pattern 5: TLS Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8443

  tls:
    cert_path: "/etc/ssl/certs/simulator.pem"
    key_path: "/etc/ssl/private/simulator.key"
    min_version: "1.2"
    client_auth: "none"
```

### Pattern 6: Authentication

```yaml
security:
  authentication:
    method: "api_key"

  api_keys:
    - key: "sk-prod-key-1"
      name: "Production Key 1"
      permissions: ["read", "write"]
      rate_limit:
        requests_per_second: 100
        burst_capacity: 200

    - key: "sk-dev-key-1"
      name: "Development Key 1"
      permissions: ["read"]
      rate_limit:
        requests_per_second: 10
        burst_capacity: 20
```

### Pattern 7: Metrics and Tracing

```yaml
telemetry:
  enabled: true
  service_name: "llm-simulator"

  logging:
    level: "info"
    format: "json"
    output: "file"
    file_path: "/var/log/llm-simulator/app.log"
    rotation:
      max_size_mb: 100
      max_backups: 10
      compress: true

  metrics:
    enabled: true
    exporter: "prometheus"
    endpoint: "/metrics"
    interval_secs: 60

    histogram_buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

  tracing:
    enabled: true
    exporter: "otlp"
    otlp_endpoint: "http://localhost:4317"
    sampling_rate: 0.1  # Sample 10% of requests
    propagation_format: "w3c"
```

## Distribution Examples

### Realistic LLM Latency (Log-Normal)

```yaml
latency:
  ttft:
    type: "log_normal"
    p50_ms: 800.0   # Median TTFT
    p99_ms: 2500.0  # Tail latency
```

**Why Log-Normal?**
- Models right-skewed distributions
- Realistic for network/compute latencies
- Captures occasional tail spikes

### Stable Latency (Normal)

```yaml
latency:
  itl:
    type: "normal"
    mean_ms: 20.0
    std_dev_ms: 5.0
    min_clamp_ms: 0.0
```

**Why Normal?**
- Stable, predictable behavior
- Good for inter-token latency
- Easy to reason about

### Cache Simulation (Bimodal)

```yaml
latency:
  ttft:
    type: "bimodal"
    fast_mean_ms: 50.0      # Cache hit
    fast_std_ms: 10.0
    slow_mean_ms: 800.0     # Cache miss
    slow_std_ms: 200.0
    fast_probability: 0.9   # 90% hit rate
```

**Why Bimodal?**
- Two distinct performance modes
- Models cache hit/miss scenarios
- Realistic for prefill caching

### Real Data (Empirical)

```yaml
latency:
  ttft:
    type: "empirical"
    samples_ms: [750, 820, 790, 850, 1200, 780, 900, 850, 780, 810]
    interpolation: "linear"
```

**Why Empirical?**
- Use actual production measurements
- No assumptions about distribution
- Most accurate for known workloads

## Hot-Reload Example

```rust
use config_system::*;
use std::path::PathBuf;
use std::time::Duration;

fn main() -> Result<(), ConfigError> {
    // Initial load
    let config_path = PathBuf::from("config/simulator.yaml");
    let loader = ConfigLoader::new();
    let initial_config = loader.load_from_file(&config_path)?;

    // Setup hot-reload
    let watcher = ConfigWatcher::new(
        config_path,
        initial_config,
        loader,
    )?
    .with_poll_interval(Duration::from_secs(5));

    // Get shared config reference
    let config_ref = watcher.get_config_ref();

    // Start watching in background
    let _handle = watcher.watch(|new_config| {
        println!("🔄 Configuration reloaded!");
        println!("  Active providers: {}", new_config.providers.len());
        println!("  Server port: {}", new_config.server.port);

        // Trigger application reconfiguration here
        // e.g., update routing, reload profiles, etc.
    });

    // Use config in application
    loop {
        let config = config_ref.read().unwrap();

        // Access current config
        println!("Current port: {}", config.server.port);

        // ... application logic ...

        std::thread::sleep(Duration::from_secs(10));
    }
}
```

## Validation Example

```rust
use config_system::*;

fn validate_config_file(path: &str) -> Result<(), ConfigError> {
    let loader = ConfigLoader::new();
    let config = loader.load_from_file(&PathBuf::from(path))?;

    match config.validate() {
        Ok(_) => {
            println!("✅ Configuration is valid");

            // Print summary
            println!("\nConfiguration Summary:");
            println!("  Version: {}", config.version);
            println!("  Server: {}:{}", config.server.host, config.server.port);
            println!("  Providers: {}", config.providers.len());
            println!("  Scenarios: {}", config.scenarios.len());
            println!("  Metrics enabled: {}", config.telemetry.metrics.enabled);
            println!("  Tracing enabled: {}", config.telemetry.tracing.enabled);

            Ok(())
        }
        Err(errors) => {
            println!("❌ Configuration validation failed\n");
            print_validation_errors(&errors);
            Err(ConfigError::ValidationError(errors))
        }
    }
}

// Usage:
// validate_config_file("config/simulator.yaml")
```

## Common Validation Errors and Fixes

### Error: Port must be greater than 0

```yaml
# ❌ Wrong
server:
  port: 0

# ✅ Correct
server:
  port: 8080
```

### Error: p99 must be greater than p50

```yaml
# ❌ Wrong
latency:
  ttft:
    type: "log_normal"
    p50_ms: 1000.0
    p99_ms: 500.0  # Less than p50!

# ✅ Correct
latency:
  ttft:
    type: "log_normal"
    p50_ms: 800.0
    p99_ms: 2500.0
```

### Error: Invalid log level

```yaml
# ❌ Wrong
telemetry:
  logging:
    level: "verbose"  # Not valid

# ✅ Correct
telemetry:
  logging:
    level: "debug"  # One of: trace, debug, info, warn, error
```

### Error: Error type probabilities must sum to 1.0

```yaml
# ❌ Wrong
errors:
  error_types:
    rate_limit:
      probability: 0.6
    server_error:
      probability: 0.5  # 0.6 + 0.5 = 1.1 (wrong!)

# ✅ Correct
errors:
  error_types:
    rate_limit:
      probability: 0.6
    server_error:
      probability: 0.4  # 0.6 + 0.4 = 1.0 ✓
```

## Environment Variable Examples

### Development

```bash
export LLM_SIM_HOST=127.0.0.1
export LLM_SIM_PORT=8080
export LLM_SIM_LOG_LEVEL=debug
export LLM_SIM_LOG_FORMAT=pretty
export LLM_SIM_METRICS_ENABLED=true
```

### Production

```bash
export LLM_SIM_HOST=0.0.0.0
export LLM_SIM_PORT=8080
export LLM_SIM_LOG_LEVEL=info
export LLM_SIM_LOG_FORMAT=json
export LLM_SIM_METRICS_ENABLED=true
export LLM_SIM_TLS_CERT=/etc/ssl/certs/simulator.pem
export LLM_SIM_TLS_KEY=/etc/ssl/private/simulator.key
```

### Docker

```dockerfile
ENV LLM_SIM_HOST=0.0.0.0
ENV LLM_SIM_PORT=8080
ENV LLM_SIM_LOG_LEVEL=info
ENV LLM_SIM_LOG_FORMAT=json
```

## Testing Your Configuration

### 1. Syntax Check

```bash
# YAML syntax
yamllint config/simulator.yaml

# Or use a YAML parser
python3 -c "import yaml; yaml.safe_load(open('config/simulator.yaml'))"
```

### 2. Validation Check

```rust
// In your code
let config = loader.load()?;
config.validate()?;  // Will return detailed errors
```

### 3. Dry Run

```bash
./llm-simulator --config config/simulator.yaml --validate-only
```

### 4. Dump Effective Config

```bash
./llm-simulator --config config/simulator.yaml --dump-config
```

## Next Steps

1. **Read Full Documentation:** See `CONFIG_SYSTEM_DESIGN.md`
2. **Review Code:** See `config_system_design.rs`
3. **Create Your Config:** Start with examples above
4. **Validate:** Run validation checks
5. **Test:** Use dry-run mode
6. **Deploy:** Apply configuration

## Troubleshooting

### Config file not found
```bash
# Check file exists
ls -l config/simulator.yaml

# Check permissions
chmod 644 config/simulator.yaml

# Use absolute path
./llm-simulator --config /full/path/to/simulator.yaml
```

### Hot-reload not working
```rust
// Ensure watcher is started
let handle = watcher.watch(|config| {
    println!("Reloaded!");
});

// Don't drop the handle
std::mem::forget(handle);
```

### Environment variables not applied
```bash
# Check variable is set
echo $LLM_SIM_PORT

# Check prefix matches
export LLM_SIM_PORT=9090  # ✓ Correct prefix
export SIM_PORT=9090       # ✗ Wrong prefix
```

## Additional Resources

- **Full Design Doc:** `CONFIG_SYSTEM_DESIGN.md`
- **Implementation:** `config_system_design.rs`
- **Schema Reference:** See "Configuration Schema Reference" in design doc
- **Migration Guide:** See "Migration Guide" in design doc

---

**Quick Reference Card:**

| Action | Command/Code |
|--------|--------------|
| Load config | `ConfigLoader::new().load()?` |
| Validate | `config.validate()?` |
| Hot-reload | `ConfigWatcher::new(...).watch(\|cfg\| {...})` |
| Env var | `export LLM_SIM_PORT=9090` |
| CLI arg | `--port 9090` |
| File format | `.yaml`, `.json`, `.toml` |
