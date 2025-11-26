// ============================================================================
// LLM-Simulator: Enterprise-Grade Configuration Management System
// ============================================================================
// Module: config
// Purpose: Flexible, multi-format configuration with hot-reload and validation
// Features: YAML/JSON/TOML, env vars, CLI args, schema validation, migrations
//
// Architecture:
// - config::schema - Complete configuration schema with validation
// - config::loader - Multi-format configuration loading (YAML/JSON/TOML)
// - config::env - Environment variable mapping and parsing
// - config::cli - CLI argument parsing and override system
// - config::watcher - Hot-reload with file watching
// - config::validation - Schema validation and helpful error messages
// - config::migration - Configuration versioning and automatic migration
// - config::merger - Hierarchical configuration with precedence
// ============================================================================

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Configuration Schema - Complete Type System
// ============================================================================

/// Root configuration structure for LLM-Simulator
///
/// Configuration precedence (highest to lowest):
/// 1. CLI arguments
/// 2. Environment variables
/// 3. Local config file (simulator.local.yaml)
/// 4. Config file (simulator.yaml)
/// 5. Default values
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SimulatorConfig {
    /// Configuration schema version for migration
    #[serde(default = "default_version")]
    pub version: String,

    /// Server configuration
    #[serde(default)]
    pub server: ServerConfig,

    /// Provider profiles configuration
    #[serde(default)]
    pub providers: HashMap<String, ProviderProfile>,

    /// Simulation parameters
    #[serde(default)]
    pub simulation: SimulationConfig,

    /// Telemetry and observability settings
    #[serde(default)]
    pub telemetry: TelemetryConfig,

    /// Test scenarios and chaos engineering
    #[serde(default)]
    pub scenarios: HashMap<String, ScenarioConfig>,

    /// Security settings
    #[serde(default)]
    pub security: SecurityConfig,

    /// Feature flags
    #[serde(default)]
    pub features: FeatureFlags,
}

fn default_version() -> String {
    "1.0".to_string()
}

// ============================================================================
// Server Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Host address to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Maximum number of concurrent connections
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Request timeout in seconds
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Keep-alive timeout in seconds
    #[serde(default = "default_keepalive_timeout")]
    pub keepalive_timeout_secs: u64,

    /// TLS/SSL configuration
    pub tls: Option<TlsConfig>,

    /// CORS configuration
    #[serde(default)]
    pub cors: CorsConfig,

    /// Rate limiting
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// Graceful shutdown timeout in seconds
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            max_connections: default_max_connections(),
            request_timeout_secs: default_request_timeout(),
            keepalive_timeout_secs: default_keepalive_timeout(),
            tls: None,
            cors: CorsConfig::default(),
            rate_limit: RateLimitConfig::default(),
            shutdown_timeout_secs: default_shutdown_timeout(),
        }
    }
}

fn default_host() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8080 }
fn default_max_connections() -> usize { 10000 }
fn default_request_timeout() -> u64 { 300 }
fn default_keepalive_timeout() -> u64 { 75 }
fn default_shutdown_timeout() -> u64 { 30 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    /// Path to TLS certificate file
    pub cert_path: PathBuf,

    /// Path to TLS private key file
    pub key_path: PathBuf,

    /// Minimum TLS version (1.2 or 1.3)
    #[serde(default = "default_min_tls_version")]
    pub min_version: String,

    /// Client certificate verification mode
    #[serde(default)]
    pub client_auth: ClientAuthMode,
}

fn default_min_tls_version() -> String { "1.2".to_string() }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ClientAuthMode {
    None,
    Optional,
    Required,
}

impl Default for ClientAuthMode {
    fn default() -> Self { Self::None }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CorsConfig {
    /// Enable CORS
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Allowed origins (use "*" for all)
    #[serde(default = "default_allowed_origins")]
    pub allowed_origins: Vec<String>,

    /// Allowed methods
    #[serde(default = "default_allowed_methods")]
    pub allowed_methods: Vec<String>,

    /// Allowed headers
    #[serde(default = "default_allowed_headers")]
    pub allowed_headers: Vec<String>,

    /// Max age for preflight cache in seconds
    #[serde(default = "default_max_age")]
    pub max_age_secs: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_origins: default_allowed_origins(),
            allowed_methods: default_allowed_methods(),
            allowed_headers: default_allowed_headers(),
            max_age_secs: default_max_age(),
        }
    }
}

fn default_true() -> bool { true }
fn default_allowed_origins() -> Vec<String> { vec!["*".to_string()] }
fn default_allowed_methods() -> Vec<String> {
    vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()]
}
fn default_allowed_headers() -> Vec<String> {
    vec!["Content-Type".to_string(), "Authorization".to_string()]
}
fn default_max_age() -> u64 { 3600 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    #[serde(default)]
    pub enabled: bool,

    /// Requests per second per IP
    #[serde(default = "default_requests_per_second")]
    pub requests_per_second: u32,

    /// Burst capacity
    #[serde(default = "default_burst_capacity")]
    pub burst_capacity: u32,

    /// Rate limit by header (e.g., API key)
    pub rate_limit_header: Option<String>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: default_requests_per_second(),
            burst_capacity: default_burst_capacity(),
            rate_limit_header: None,
        }
    }
}

fn default_requests_per_second() -> u32 { 100 }
fn default_burst_capacity() -> u32 { 200 }

// ============================================================================
// Provider Profile Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProviderProfile {
    /// Human-readable name
    pub name: String,

    /// Provider identifier (openai, anthropic, google, etc.)
    pub provider: String,

    /// Model identifier
    pub model: String,

    /// Enable this profile
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Latency profile configuration
    pub latency: LatencyProfileConfig,

    /// Error injection configuration
    #[serde(default)]
    pub errors: ErrorConfig,

    /// Response behavior configuration
    #[serde(default)]
    pub response: ResponseConfig,

    /// Cost simulation (tokens/dollar)
    pub cost: Option<CostConfig>,

    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LatencyProfileConfig {
    /// Time to First Token distribution
    pub ttft: DistributionConfig,

    /// Inter-Token Latency distribution
    pub itl: DistributionConfig,

    /// Network jitter (optional)
    pub network_jitter: Option<DistributionConfig>,

    /// Load degradation model
    #[serde(default)]
    pub degradation: DegradationConfig,

    /// Validation metrics from real measurements
    pub validation_metrics: Option<ValidationMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistributionConfig {
    Normal {
        mean_ms: f64,
        std_dev_ms: f64,
        #[serde(default)]
        min_clamp_ms: Option<f64>,
    },
    LogNormal {
        p50_ms: f64,
        p99_ms: f64,
    },
    Exponential {
        mean_ms: f64,
    },
    Bimodal {
        fast_mean_ms: f64,
        fast_std_ms: f64,
        slow_mean_ms: f64,
        slow_std_ms: f64,
        fast_probability: f64,
    },
    Empirical {
        samples_ms: Vec<f64>,
        #[serde(default = "default_interpolation")]
        interpolation: InterpolationMethod,
    },
    Constant {
        value_ms: f64,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationMethod {
    Linear,
    Nearest,
}

fn default_interpolation() -> InterpolationMethod {
    InterpolationMethod::Linear
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DegradationConfig {
    #[serde(flatten)]
    pub model: DegradationModel,

    /// Baseline QPS for this profile
    #[serde(default = "default_baseline_qps")]
    pub baseline_qps: f64,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            model: DegradationModel::None,
            baseline_qps: default_baseline_qps(),
        }
    }
}

fn default_baseline_qps() -> f64 { 10.0 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DegradationModel {
    None,
    Linear {
        slope: f64,
    },
    Exponential {
        alpha: f64,
    },
    MMOne {
        service_rate: f64,
    },
    Piecewise {
        breakpoints: Vec<Breakpoint>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Breakpoint {
    pub qps: f64,
    pub multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ValidationMetrics {
    pub ttft_p50_ms: f64,
    pub ttft_p99_ms: f64,
    pub itl_mean_ms: f64,
    pub itl_p99_ms: f64,
    pub tokens_per_sec: f64,
    pub measurement_date: String,
    pub sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ErrorConfig {
    /// Enable error injection
    #[serde(default)]
    pub enabled: bool,

    /// Base error rate (0.0 - 1.0)
    #[serde(default)]
    pub base_rate: f64,

    /// Error types and their probabilities
    #[serde(default)]
    pub error_types: HashMap<String, ErrorTypeConfig>,

    /// Chaos scenarios
    #[serde(default)]
    pub chaos_scenarios: Vec<ChaosScenario>,
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_rate: 0.0,
            error_types: HashMap::new(),
            chaos_scenarios: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ErrorTypeConfig {
    /// Probability of this error type (0.0 - 1.0)
    pub probability: f64,

    /// HTTP status code
    pub status_code: u16,

    /// Error message template
    pub message: String,

    /// Retry-After header value in seconds
    pub retry_after: Option<u64>,

    /// Additional headers
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChaosScenario {
    /// Scenario name
    pub name: String,

    /// When to trigger (time-based, load-based, random)
    pub trigger: ChaosTrigger,

    /// Duration of the scenario
    pub duration_secs: u64,

    /// Error rate during scenario (0.0 - 1.0)
    pub error_rate: f64,

    /// Latency multiplier during scenario
    #[serde(default = "default_one")]
    pub latency_multiplier: f64,
}

fn default_one() -> f64 { 1.0 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChaosTrigger {
    Time {
        cron_expr: String,
    },
    Load {
        qps_threshold: f64,
    },
    Random {
        probability: f64,
    },
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResponseConfig {
    /// Response content generation strategy
    #[serde(default)]
    pub content_strategy: ContentStrategy,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Token vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Streaming chunk size
    #[serde(default = "default_chunk_size")]
    pub streaming_chunk_size: usize,

    /// Include usage metadata
    #[serde(default = "default_true")]
    pub include_usage: bool,

    /// Custom response templates
    #[serde(default)]
    pub templates: HashMap<String, String>,
}

impl Default for ResponseConfig {
    fn default() -> Self {
        Self {
            content_strategy: ContentStrategy::default(),
            max_tokens: default_max_tokens(),
            vocab_size: default_vocab_size(),
            streaming_chunk_size: default_chunk_size(),
            include_usage: true,
            templates: HashMap::new(),
        }
    }
}

fn default_max_tokens() -> usize { 4096 }
fn default_vocab_size() -> usize { 50000 }
fn default_chunk_size() -> usize { 1 }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ContentStrategy {
    /// Generate random tokens
    Random,
    /// Repeat pattern
    Pattern,
    /// Use Markov chain
    Markov,
    /// Echo input
    Echo,
    /// Load from templates
    Template,
}

impl Default for ContentStrategy {
    fn default() -> Self { Self::Random }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CostConfig {
    /// Cost per 1M input tokens (USD)
    pub input_tokens_per_million: f64,

    /// Cost per 1M output tokens (USD)
    pub output_tokens_per_million: f64,

    /// Cost per 1M cached tokens (USD, optional)
    pub cached_tokens_per_million: Option<f64>,

    /// Include cost in response metadata
    #[serde(default = "default_true")]
    pub include_in_response: bool,
}

// ============================================================================
// Simulation Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Maximum concurrent sessions
    #[serde(default = "default_max_sessions")]
    pub max_concurrent_sessions: usize,

    /// Session timeout in seconds
    #[serde(default = "default_session_timeout")]
    pub session_timeout_secs: u64,

    /// Enable session persistence
    #[serde(default)]
    pub persist_sessions: bool,

    /// Session storage path
    pub session_storage_path: Option<PathBuf>,

    /// Concurrency model
    #[serde(default)]
    pub concurrency: ConcurrencyConfig,

    /// Resource limits
    #[serde(default)]
    pub limits: ResourceLimits,

    /// Warm-up configuration
    pub warmup: Option<WarmupConfig>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            seed: None,
            max_concurrent_sessions: default_max_sessions(),
            session_timeout_secs: default_session_timeout(),
            persist_sessions: false,
            session_storage_path: None,
            concurrency: ConcurrencyConfig::default(),
            limits: ResourceLimits::default(),
            warmup: None,
        }
    }
}

fn default_max_sessions() -> usize { 1000 }
fn default_session_timeout() -> u64 { 3600 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConcurrencyConfig {
    /// Number of worker threads (0 = auto-detect)
    #[serde(default)]
    pub worker_threads: usize,

    /// Thread pool size for blocking operations
    #[serde(default = "default_blocking_threads")]
    pub blocking_threads: usize,

    /// Task queue size
    #[serde(default = "default_queue_size")]
    pub task_queue_size: usize,

    /// Back-pressure strategy
    #[serde(default)]
    pub backpressure: BackpressureStrategy,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            worker_threads: 0,
            blocking_threads: default_blocking_threads(),
            task_queue_size: default_queue_size(),
            backpressure: BackpressureStrategy::default(),
        }
    }
}

fn default_blocking_threads() -> usize { 512 }
fn default_queue_size() -> usize { 10000 }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BackpressureStrategy {
    /// Drop new requests
    Drop,
    /// Block until capacity available
    Block,
    /// Return 503 Service Unavailable
    Reject,
}

impl Default for BackpressureStrategy {
    fn default() -> Self { Self::Reject }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,

    /// Maximum CPU usage percentage (0-100)
    pub max_cpu_percent: Option<u8>,

    /// Maximum disk usage in MB
    pub max_disk_mb: Option<usize>,

    /// Maximum request body size in MB
    #[serde(default = "default_max_body_size")]
    pub max_body_size_mb: usize,

    /// Maximum response size in MB
    #[serde(default = "default_max_response_size")]
    pub max_response_size_mb: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            max_cpu_percent: None,
            max_disk_mb: None,
            max_body_size_mb: default_max_body_size(),
            max_response_size_mb: default_max_response_size(),
        }
    }
}

fn default_max_body_size() -> usize { 10 }
fn default_max_response_size() -> usize { 100 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WarmupConfig {
    /// Enable warmup phase
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Number of warmup requests per profile
    #[serde(default = "default_warmup_requests")]
    pub requests_per_profile: usize,

    /// Warmup duration in seconds
    #[serde(default = "default_warmup_duration")]
    pub duration_secs: u64,
}

fn default_warmup_requests() -> usize { 100 }
fn default_warmup_duration() -> u64 { 30 }

// ============================================================================
// Telemetry Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// Enable telemetry
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Service name for telemetry
    #[serde(default = "default_service_name")]
    pub service_name: String,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Tracing configuration
    #[serde(default)]
    pub tracing: TracingConfig,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            service_name: default_service_name(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
        }
    }
}

fn default_service_name() -> String { "llm-simulator".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log format (json, text, pretty)
    #[serde(default = "default_log_format")]
    pub format: String,

    /// Log output (stdout, stderr, file)
    #[serde(default = "default_log_output")]
    pub output: String,

    /// Log file path (if output=file)
    pub file_path: Option<PathBuf>,

    /// Enable log rotation
    #[serde(default)]
    pub rotation: Option<LogRotationConfig>,

    /// Include source location
    #[serde(default)]
    pub include_location: bool,

    /// Include thread ID
    #[serde(default)]
    pub include_thread_id: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            output: default_log_output(),
            file_path: None,
            rotation: None,
            include_location: false,
            include_thread_id: false,
        }
    }
}

fn default_log_level() -> String { "info".to_string() }
fn default_log_format() -> String { "json".to_string() }
fn default_log_output() -> String { "stdout".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LogRotationConfig {
    /// Maximum log file size in MB
    #[serde(default = "default_max_log_size")]
    pub max_size_mb: usize,

    /// Maximum number of archived log files
    #[serde(default = "default_max_backups")]
    pub max_backups: usize,

    /// Compress archived logs
    #[serde(default = "default_true")]
    pub compress: bool,
}

fn default_max_log_size() -> usize { 100 }
fn default_max_backups() -> usize { 10 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// Enable metrics collection
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Metrics exporter type (prometheus, otlp, statsd)
    #[serde(default = "default_metrics_exporter")]
    pub exporter: String,

    /// Prometheus endpoint
    #[serde(default = "default_metrics_endpoint")]
    pub endpoint: String,

    /// Metrics collection interval in seconds
    #[serde(default = "default_metrics_interval")]
    pub interval_secs: u64,

    /// Custom metrics
    #[serde(default)]
    pub custom_metrics: Vec<CustomMetric>,

    /// Histogram buckets
    #[serde(default = "default_histogram_buckets")]
    pub histogram_buckets: Vec<f64>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exporter: default_metrics_exporter(),
            endpoint: default_metrics_endpoint(),
            interval_secs: default_metrics_interval(),
            custom_metrics: Vec::new(),
            histogram_buckets: default_histogram_buckets(),
        }
    }
}

fn default_metrics_exporter() -> String { "prometheus".to_string() }
fn default_metrics_endpoint() -> String { "/metrics".to_string() }
fn default_metrics_interval() -> u64 { 60 }
fn default_histogram_buckets() -> Vec<f64> {
    vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    #[serde(default)]
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TracingConfig {
    /// Enable distributed tracing
    #[serde(default)]
    pub enabled: bool,

    /// Tracing exporter (otlp, jaeger, zipkin)
    #[serde(default = "default_tracing_exporter")]
    pub exporter: String,

    /// OTLP endpoint
    pub otlp_endpoint: Option<String>,

    /// Sampling rate (0.0 - 1.0)
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: f64,

    /// Trace ID generation strategy
    #[serde(default)]
    pub trace_id_strategy: TraceIdStrategy,

    /// Propagation format (w3c, b3, jaeger)
    #[serde(default = "default_propagation_format")]
    pub propagation_format: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            exporter: default_tracing_exporter(),
            otlp_endpoint: None,
            sampling_rate: default_sampling_rate(),
            trace_id_strategy: TraceIdStrategy::default(),
            propagation_format: default_propagation_format(),
        }
    }
}

fn default_tracing_exporter() -> String { "otlp".to_string() }
fn default_sampling_rate() -> f64 { 1.0 }
fn default_propagation_format() -> String { "w3c".to_string() }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TraceIdStrategy {
    Random,
    Deterministic,
}

impl Default for TraceIdStrategy {
    fn default() -> Self { Self::Random }
}

// ============================================================================
// Scenario Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioConfig {
    /// Scenario name
    pub name: String,

    /// Scenario description
    #[serde(default)]
    pub description: String,

    /// Enable this scenario
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Profile to use for this scenario
    pub profile: String,

    /// Request pattern
    pub pattern: RequestPattern,

    /// Expected outcomes
    #[serde(default)]
    pub assertions: Vec<Assertion>,

    /// Scenario tags
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestPattern {
    Constant {
        qps: f64,
        duration_secs: u64,
    },
    Ramp {
        start_qps: f64,
        end_qps: f64,
        duration_secs: u64,
    },
    Spike {
        baseline_qps: f64,
        spike_qps: f64,
        spike_duration_secs: u64,
        total_duration_secs: u64,
    },
    Poisson {
        mean_qps: f64,
        duration_secs: u64,
    },
    Custom {
        script: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Assertion {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    #[serde(default)]
    pub percentile: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
}

// ============================================================================
// Security Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SecurityConfig {
    /// Enable authentication
    #[serde(default)]
    pub authentication: Option<AuthenticationConfig>,

    /// API keys for access control
    #[serde(default)]
    pub api_keys: Vec<ApiKeyConfig>,

    /// IP allowlist
    #[serde(default)]
    pub ip_allowlist: Vec<String>,

    /// IP blocklist
    #[serde(default)]
    pub ip_blocklist: Vec<String>,

    /// Enable request signing
    #[serde(default)]
    pub request_signing: bool,

    /// Secret key for request signing
    pub signing_secret: Option<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            authentication: None,
            api_keys: Vec::new(),
            ip_allowlist: Vec::new(),
            ip_blocklist: Vec::new(),
            request_signing: false,
            signing_secret: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticationConfig {
    /// Authentication method (api_key, jwt, oauth2)
    pub method: String,

    /// JWT configuration
    pub jwt: Option<JwtConfig>,

    /// OAuth2 configuration
    pub oauth2: Option<OAuth2Config>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JwtConfig {
    /// JWT secret or public key
    pub secret: String,

    /// JWT algorithm (HS256, RS256, etc.)
    #[serde(default = "default_jwt_algorithm")]
    pub algorithm: String,

    /// Token expiration in seconds
    #[serde(default = "default_jwt_expiration")]
    pub expiration_secs: u64,

    /// Issuer
    pub issuer: Option<String>,

    /// Audience
    pub audience: Option<String>,
}

fn default_jwt_algorithm() -> String { "HS256".to_string() }
fn default_jwt_expiration() -> u64 { 3600 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OAuth2Config {
    /// OAuth2 provider
    pub provider: String,

    /// Client ID
    pub client_id: String,

    /// Client secret
    pub client_secret: String,

    /// Authorization URL
    pub auth_url: String,

    /// Token URL
    pub token_url: String,

    /// Scopes
    #[serde(default)]
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyConfig {
    /// API key value
    pub key: String,

    /// Key name/identifier
    pub name: String,

    /// Permissions
    #[serde(default)]
    pub permissions: Vec<String>,

    /// Expiration timestamp
    pub expires_at: Option<String>,

    /// Rate limit override for this key
    pub rate_limit: Option<RateLimitConfig>,
}

// ============================================================================
// Feature Flags
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeatureFlags {
    /// Enable streaming responses
    #[serde(default = "default_true")]
    pub streaming: bool,

    /// Enable function calling simulation
    #[serde(default)]
    pub function_calling: bool,

    /// Enable vision/multimodal simulation
    #[serde(default)]
    pub vision: bool,

    /// Enable embeddings endpoint
    #[serde(default)]
    pub embeddings: bool,

    /// Enable fine-tuning simulation
    #[serde(default)]
    pub fine_tuning: bool,

    /// Enable batch processing
    #[serde(default)]
    pub batch_processing: bool,

    /// Enable caching layer
    #[serde(default)]
    pub caching: bool,

    /// Enable prompt caching simulation
    #[serde(default)]
    pub prompt_caching: bool,

    /// Custom feature flags
    #[serde(default)]
    pub custom: HashMap<String, bool>,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            streaming: true,
            function_calling: false,
            vision: false,
            embeddings: false,
            fine_tuning: false,
            batch_processing: false,
            caching: false,
            prompt_caching: false,
            custom: HashMap::new(),
        }
    }
}

// ============================================================================
// Configuration Validation
// ============================================================================

/// Validation errors with detailed context
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub constraint: String,
    pub actual_value: String,
}

impl ValidationError {
    pub fn new(field: &str, message: &str, constraint: &str, actual_value: &str) -> Self {
        Self {
            field: field.to_string(),
            message: message.to_string(),
            constraint: constraint.to_string(),
            actual_value: actual_value.to_string(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Validation error in field '{}': {}\n  Constraint: {}\n  Actual value: {}",
            self.field, self.message, self.constraint, self.actual_value
        )
    }
}

impl std::error::Error for ValidationError {}

pub type ValidationResult = Result<(), Vec<ValidationError>>;

/// Trait for configuration validation
pub trait Validate {
    fn validate(&self) -> ValidationResult;
}

impl Validate for SimulatorConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate version
        if !is_valid_version(&self.version) {
            errors.push(ValidationError::new(
                "version",
                "Invalid version format",
                "Must be semantic version (e.g., '1.0', '1.2.3')",
                &self.version,
            ));
        }

        // Validate server config
        if let Err(mut server_errors) = self.server.validate() {
            errors.append(&mut server_errors);
        }

        // Validate providers
        for (name, profile) in &self.providers {
            if let Err(mut profile_errors) = profile.validate() {
                for error in &mut profile_errors {
                    error.field = format!("providers.{}.{}", name, error.field);
                }
                errors.append(&mut profile_errors);
            }
        }

        // Validate simulation config
        if let Err(mut sim_errors) = self.simulation.validate() {
            errors.append(&mut sim_errors);
        }

        // Validate telemetry config
        if let Err(mut telemetry_errors) = self.telemetry.validate() {
            errors.append(&mut telemetry_errors);
        }

        // Validate scenarios
        for (name, scenario) in &self.scenarios {
            if let Err(mut scenario_errors) = scenario.validate(&self.providers) {
                for error in &mut scenario_errors {
                    error.field = format!("scenarios.{}.{}", name, error.field);
                }
                errors.append(&mut scenario_errors);
            }
        }

        // Validate security config
        if let Err(mut security_errors) = self.security.validate() {
            errors.append(&mut security_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for ServerConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate port
        if self.port == 0 {
            errors.push(ValidationError::new(
                "server.port",
                "Port must be greater than 0",
                "1-65535",
                &self.port.to_string(),
            ));
        }

        // Validate max connections
        if self.max_connections == 0 {
            errors.push(ValidationError::new(
                "server.max_connections",
                "max_connections must be greater than 0",
                "> 0",
                &self.max_connections.to_string(),
            ));
        }

        // Validate timeouts
        if self.request_timeout_secs == 0 {
            errors.push(ValidationError::new(
                "server.request_timeout_secs",
                "request_timeout_secs must be greater than 0",
                "> 0",
                &self.request_timeout_secs.to_string(),
            ));
        }

        // Validate TLS config
        if let Some(ref tls) = self.tls {
            if !tls.cert_path.exists() {
                errors.push(ValidationError::new(
                    "server.tls.cert_path",
                    "Certificate file does not exist",
                    "Must be valid file path",
                    tls.cert_path.to_string_lossy().as_ref(),
                ));
            }
            if !tls.key_path.exists() {
                errors.push(ValidationError::new(
                    "server.tls.key_path",
                    "Private key file does not exist",
                    "Must be valid file path",
                    tls.key_path.to_string_lossy().as_ref(),
                ));
            }
            if tls.min_version != "1.2" && tls.min_version != "1.3" {
                errors.push(ValidationError::new(
                    "server.tls.min_version",
                    "Invalid TLS version",
                    "Must be '1.2' or '1.3'",
                    &tls.min_version,
                ));
            }
        }

        // Validate rate limiting
        if let Err(mut rate_limit_errors) = self.rate_limit.validate() {
            errors.append(&mut rate_limit_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for RateLimitConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if self.enabled {
            if self.requests_per_second == 0 {
                errors.push(ValidationError::new(
                    "server.rate_limit.requests_per_second",
                    "requests_per_second must be greater than 0 when rate limiting is enabled",
                    "> 0",
                    &self.requests_per_second.to_string(),
                ));
            }

            if self.burst_capacity < self.requests_per_second {
                errors.push(ValidationError::new(
                    "server.rate_limit.burst_capacity",
                    "burst_capacity must be >= requests_per_second",
                    &format!(">= {}", self.requests_per_second),
                    &self.burst_capacity.to_string(),
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for ProviderProfile {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate latency config
        if let Err(mut latency_errors) = self.latency.validate() {
            errors.append(&mut latency_errors);
        }

        // Validate error config
        if let Err(mut error_errors) = self.errors.validate() {
            errors.append(&mut error_errors);
        }

        // Validate cost config
        if let Some(ref cost) = self.cost {
            if cost.input_tokens_per_million < 0.0 {
                errors.push(ValidationError::new(
                    "cost.input_tokens_per_million",
                    "Cost cannot be negative",
                    ">= 0.0",
                    &cost.input_tokens_per_million.to_string(),
                ));
            }
            if cost.output_tokens_per_million < 0.0 {
                errors.push(ValidationError::new(
                    "cost.output_tokens_per_million",
                    "Cost cannot be negative",
                    ">= 0.0",
                    &cost.output_tokens_per_million.to_string(),
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for LatencyProfileConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate TTFT distribution
        if let Err(e) = self.ttft.validate("latency.ttft") {
            errors.push(e);
        }

        // Validate ITL distribution
        if let Err(e) = self.itl.validate("latency.itl") {
            errors.push(e);
        }

        // Validate network jitter
        if let Some(ref jitter) = self.network_jitter {
            if let Err(e) = jitter.validate("latency.network_jitter") {
                errors.push(e);
            }
        }

        // Validate degradation config
        if let Err(mut deg_errors) = self.degradation.validate() {
            errors.append(&mut deg_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl DistributionConfig {
    fn validate(&self, field_prefix: &str) -> Result<(), ValidationError> {
        match self {
            DistributionConfig::Normal { mean_ms, std_dev_ms, .. } => {
                if *mean_ms <= 0.0 {
                    return Err(ValidationError::new(
                        &format!("{}.mean_ms", field_prefix),
                        "Mean must be positive",
                        "> 0.0",
                        &mean_ms.to_string(),
                    ));
                }
                if *std_dev_ms < 0.0 {
                    return Err(ValidationError::new(
                        &format!("{}.std_dev_ms", field_prefix),
                        "Standard deviation cannot be negative",
                        ">= 0.0",
                        &std_dev_ms.to_string(),
                    ));
                }
            }
            DistributionConfig::LogNormal { p50_ms, p99_ms } => {
                if *p50_ms <= 0.0 {
                    return Err(ValidationError::new(
                        &format!("{}.p50_ms", field_prefix),
                        "p50 must be positive",
                        "> 0.0",
                        &p50_ms.to_string(),
                    ));
                }
                if *p99_ms <= *p50_ms {
                    return Err(ValidationError::new(
                        &format!("{}.p99_ms", field_prefix),
                        "p99 must be greater than p50",
                        &format!("> {}", p50_ms),
                        &p99_ms.to_string(),
                    ));
                }
            }
            DistributionConfig::Exponential { mean_ms } => {
                if *mean_ms <= 0.0 {
                    return Err(ValidationError::new(
                        &format!("{}.mean_ms", field_prefix),
                        "Mean must be positive",
                        "> 0.0",
                        &mean_ms.to_string(),
                    ));
                }
            }
            DistributionConfig::Bimodal { fast_probability, .. } => {
                if *fast_probability < 0.0 || *fast_probability > 1.0 {
                    return Err(ValidationError::new(
                        &format!("{}.fast_probability", field_prefix),
                        "Probability must be between 0 and 1",
                        "0.0 - 1.0",
                        &fast_probability.to_string(),
                    ));
                }
            }
            DistributionConfig::Empirical { samples_ms, .. } => {
                if samples_ms.is_empty() {
                    return Err(ValidationError::new(
                        &format!("{}.samples_ms", field_prefix),
                        "Samples cannot be empty",
                        "At least 1 sample required",
                        "[]",
                    ));
                }
            }
            DistributionConfig::Constant { value_ms } => {
                if *value_ms < 0.0 {
                    return Err(ValidationError::new(
                        &format!("{}.value_ms", field_prefix),
                        "Value cannot be negative",
                        ">= 0.0",
                        &value_ms.to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Validate for DegradationConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if self.baseline_qps <= 0.0 {
            errors.push(ValidationError::new(
                "degradation.baseline_qps",
                "Baseline QPS must be positive",
                "> 0.0",
                &self.baseline_qps.to_string(),
            ));
        }

        match &self.model {
            DegradationModel::Linear { slope } => {
                if *slope < 0.0 {
                    errors.push(ValidationError::new(
                        "degradation.slope",
                        "Slope cannot be negative",
                        ">= 0.0",
                        &slope.to_string(),
                    ));
                }
            }
            DegradationModel::Exponential { alpha } => {
                if *alpha < 0.0 {
                    errors.push(ValidationError::new(
                        "degradation.alpha",
                        "Alpha cannot be negative",
                        ">= 0.0",
                        &alpha.to_string(),
                    ));
                }
            }
            DegradationModel::MMOne { service_rate } => {
                if *service_rate <= 0.0 {
                    errors.push(ValidationError::new(
                        "degradation.service_rate",
                        "Service rate must be positive",
                        "> 0.0",
                        &service_rate.to_string(),
                    ));
                }
            }
            DegradationModel::Piecewise { breakpoints } => {
                if breakpoints.is_empty() {
                    errors.push(ValidationError::new(
                        "degradation.breakpoints",
                        "Breakpoints cannot be empty for piecewise model",
                        "At least 1 breakpoint required",
                        "[]",
                    ));
                }
                // Verify breakpoints are sorted
                for i in 1..breakpoints.len() {
                    if breakpoints[i].qps <= breakpoints[i - 1].qps {
                        errors.push(ValidationError::new(
                            "degradation.breakpoints",
                            "Breakpoints must be sorted by QPS in ascending order",
                            "Monotonically increasing QPS values",
                            &format!("{:?}", breakpoints),
                        ));
                        break;
                    }
                }
            }
            DegradationModel::None => {}
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for ErrorConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if self.base_rate < 0.0 || self.base_rate > 1.0 {
            errors.push(ValidationError::new(
                "errors.base_rate",
                "Base error rate must be between 0 and 1",
                "0.0 - 1.0",
                &self.base_rate.to_string(),
            ));
        }

        // Validate error types
        let mut total_probability = 0.0;
        for (name, error_type) in &self.error_types {
            if error_type.probability < 0.0 || error_type.probability > 1.0 {
                errors.push(ValidationError::new(
                    &format!("errors.error_types.{}.probability", name),
                    "Probability must be between 0 and 1",
                    "0.0 - 1.0",
                    &error_type.probability.to_string(),
                ));
            }
            total_probability += error_type.probability;

            if error_type.status_code < 100 || error_type.status_code >= 600 {
                errors.push(ValidationError::new(
                    &format!("errors.error_types.{}.status_code", name),
                    "Invalid HTTP status code",
                    "100-599",
                    &error_type.status_code.to_string(),
                ));
            }
        }

        if (total_probability - 1.0).abs() > 0.001 && !self.error_types.is_empty() {
            errors.push(ValidationError::new(
                "errors.error_types",
                "Error type probabilities must sum to 1.0",
                "Sum = 1.0",
                &total_probability.to_string(),
            ));
        }

        // Validate chaos scenarios
        for (idx, scenario) in self.chaos_scenarios.iter().enumerate() {
            if scenario.error_rate < 0.0 || scenario.error_rate > 1.0 {
                errors.push(ValidationError::new(
                    &format!("errors.chaos_scenarios[{}].error_rate", idx),
                    "Error rate must be between 0 and 1",
                    "0.0 - 1.0",
                    &scenario.error_rate.to_string(),
                ));
            }
            if scenario.latency_multiplier < 0.0 {
                errors.push(ValidationError::new(
                    &format!("errors.chaos_scenarios[{}].latency_multiplier", idx),
                    "Latency multiplier cannot be negative",
                    ">= 0.0",
                    &scenario.latency_multiplier.to_string(),
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for SimulationConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if self.max_concurrent_sessions == 0 {
            errors.push(ValidationError::new(
                "simulation.max_concurrent_sessions",
                "max_concurrent_sessions must be greater than 0",
                "> 0",
                &self.max_concurrent_sessions.to_string(),
            ));
        }

        if self.persist_sessions && self.session_storage_path.is_none() {
            errors.push(ValidationError::new(
                "simulation.session_storage_path",
                "session_storage_path required when persist_sessions is enabled",
                "Valid directory path",
                "None",
            ));
        }

        if let Err(mut limits_errors) = self.limits.validate() {
            errors.append(&mut limits_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for ResourceLimits {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if let Some(cpu) = self.max_cpu_percent {
            if cpu == 0 || cpu > 100 {
                errors.push(ValidationError::new(
                    "simulation.limits.max_cpu_percent",
                    "CPU percentage must be 1-100",
                    "1-100",
                    &cpu.to_string(),
                ));
            }
        }

        if self.max_body_size_mb == 0 {
            errors.push(ValidationError::new(
                "simulation.limits.max_body_size_mb",
                "max_body_size_mb must be greater than 0",
                "> 0",
                &self.max_body_size_mb.to_string(),
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for TelemetryConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate logging config
        if let Err(mut log_errors) = self.logging.validate() {
            errors.append(&mut log_errors);
        }

        // Validate metrics config
        if let Err(mut metrics_errors) = self.metrics.validate() {
            errors.append(&mut metrics_errors);
        }

        // Validate tracing config
        if let Err(mut trace_errors) = self.tracing.validate() {
            errors.append(&mut trace_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for LoggingConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.level.as_str()) {
            errors.push(ValidationError::new(
                "telemetry.logging.level",
                "Invalid log level",
                "One of: trace, debug, info, warn, error",
                &self.level,
            ));
        }

        let valid_formats = ["json", "text", "pretty"];
        if !valid_formats.contains(&self.format.as_str()) {
            errors.push(ValidationError::new(
                "telemetry.logging.format",
                "Invalid log format",
                "One of: json, text, pretty",
                &self.format,
            ));
        }

        let valid_outputs = ["stdout", "stderr", "file"];
        if !valid_outputs.contains(&self.output.as_str()) {
            errors.push(ValidationError::new(
                "telemetry.logging.output",
                "Invalid log output",
                "One of: stdout, stderr, file",
                &self.output,
            ));
        }

        if self.output == "file" && self.file_path.is_none() {
            errors.push(ValidationError::new(
                "telemetry.logging.file_path",
                "file_path required when output is 'file'",
                "Valid file path",
                "None",
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for MetricsConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        let valid_exporters = ["prometheus", "otlp", "statsd"];
        if !valid_exporters.contains(&self.exporter.as_str()) {
            errors.push(ValidationError::new(
                "telemetry.metrics.exporter",
                "Invalid metrics exporter",
                "One of: prometheus, otlp, statsd",
                &self.exporter,
            ));
        }

        if self.interval_secs == 0 {
            errors.push(ValidationError::new(
                "telemetry.metrics.interval_secs",
                "Interval must be greater than 0",
                "> 0",
                &self.interval_secs.to_string(),
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for TracingConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if self.enabled {
            let valid_exporters = ["otlp", "jaeger", "zipkin"];
            if !valid_exporters.contains(&self.exporter.as_str()) {
                errors.push(ValidationError::new(
                    "telemetry.tracing.exporter",
                    "Invalid tracing exporter",
                    "One of: otlp, jaeger, zipkin",
                    &self.exporter,
                ));
            }

            if self.exporter == "otlp" && self.otlp_endpoint.is_none() {
                errors.push(ValidationError::new(
                    "telemetry.tracing.otlp_endpoint",
                    "otlp_endpoint required when exporter is 'otlp'",
                    "Valid URL",
                    "None",
                ));
            }
        }

        if self.sampling_rate < 0.0 || self.sampling_rate > 1.0 {
            errors.push(ValidationError::new(
                "telemetry.tracing.sampling_rate",
                "Sampling rate must be between 0 and 1",
                "0.0 - 1.0",
                &self.sampling_rate.to_string(),
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl ScenarioConfig {
    fn validate(&self, providers: &HashMap<String, ProviderProfile>) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate that referenced profile exists
        if !providers.contains_key(&self.profile) {
            errors.push(ValidationError::new(
                "profile",
                "Referenced profile does not exist",
                "Must reference existing provider profile",
                &self.profile,
            ));
        }

        // Validate request pattern
        match &self.pattern {
            RequestPattern::Constant { qps, duration_secs } => {
                if *qps <= 0.0 {
                    errors.push(ValidationError::new(
                        "pattern.qps",
                        "QPS must be positive",
                        "> 0.0",
                        &qps.to_string(),
                    ));
                }
                if *duration_secs == 0 {
                    errors.push(ValidationError::new(
                        "pattern.duration_secs",
                        "Duration must be greater than 0",
                        "> 0",
                        &duration_secs.to_string(),
                    ));
                }
            }
            RequestPattern::Ramp { start_qps, end_qps, duration_secs } => {
                if *start_qps < 0.0 {
                    errors.push(ValidationError::new(
                        "pattern.start_qps",
                        "Start QPS cannot be negative",
                        ">= 0.0",
                        &start_qps.to_string(),
                    ));
                }
                if *end_qps <= 0.0 {
                    errors.push(ValidationError::new(
                        "pattern.end_qps",
                        "End QPS must be positive",
                        "> 0.0",
                        &end_qps.to_string(),
                    ));
                }
                if *duration_secs == 0 {
                    errors.push(ValidationError::new(
                        "pattern.duration_secs",
                        "Duration must be greater than 0",
                        "> 0",
                        &duration_secs.to_string(),
                    ));
                }
            }
            RequestPattern::Spike { baseline_qps, spike_qps, spike_duration_secs, total_duration_secs } => {
                if *baseline_qps < 0.0 {
                    errors.push(ValidationError::new(
                        "pattern.baseline_qps",
                        "Baseline QPS cannot be negative",
                        ">= 0.0",
                        &baseline_qps.to_string(),
                    ));
                }
                if *spike_qps <= *baseline_qps {
                    errors.push(ValidationError::new(
                        "pattern.spike_qps",
                        "Spike QPS must be greater than baseline QPS",
                        &format!("> {}", baseline_qps),
                        &spike_qps.to_string(),
                    ));
                }
                if *spike_duration_secs >= *total_duration_secs {
                    errors.push(ValidationError::new(
                        "pattern.spike_duration_secs",
                        "Spike duration must be less than total duration",
                        &format!("< {}", total_duration_secs),
                        &spike_duration_secs.to_string(),
                    ));
                }
            }
            RequestPattern::Poisson { mean_qps, duration_secs } => {
                if *mean_qps <= 0.0 {
                    errors.push(ValidationError::new(
                        "pattern.mean_qps",
                        "Mean QPS must be positive",
                        "> 0.0",
                        &mean_qps.to_string(),
                    ));
                }
                if *duration_secs == 0 {
                    errors.push(ValidationError::new(
                        "pattern.duration_secs",
                        "Duration must be greater than 0",
                        "> 0",
                        &duration_secs.to_string(),
                    ));
                }
            }
            RequestPattern::Custom { .. } => {
                // Custom scripts can't be validated without execution
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Validate for SecurityConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if let Some(ref auth) = self.authentication {
            let valid_methods = ["api_key", "jwt", "oauth2"];
            if !valid_methods.contains(&auth.method.as_str()) {
                errors.push(ValidationError::new(
                    "security.authentication.method",
                    "Invalid authentication method",
                    "One of: api_key, jwt, oauth2",
                    &auth.method,
                ));
            }

            if auth.method == "jwt" && auth.jwt.is_none() {
                errors.push(ValidationError::new(
                    "security.authentication.jwt",
                    "JWT config required when method is 'jwt'",
                    "JWT configuration",
                    "None",
                ));
            }

            if auth.method == "oauth2" && auth.oauth2.is_none() {
                errors.push(ValidationError::new(
                    "security.authentication.oauth2",
                    "OAuth2 config required when method is 'oauth2'",
                    "OAuth2 configuration",
                    "None",
                ));
            }
        }

        if self.request_signing && self.signing_secret.is_none() {
            errors.push(ValidationError::new(
                "security.signing_secret",
                "signing_secret required when request_signing is enabled",
                "Secret key",
                "None",
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn is_valid_version(version: &str) -> bool {
    // Simple semantic version validation
    let parts: Vec<&str> = version.split('.').collect();
    if parts.is_empty() || parts.len() > 3 {
        return false;
    }
    parts.iter().all(|p| p.parse::<u32>().is_ok())
}

// ============================================================================
// Configuration Loading - Multi-Format Support
// ============================================================================

#[derive(Debug)]
pub enum ConfigFormat {
    Yaml,
    Json,
    Toml,
}

impl ConfigFormat {
    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => Ok(ConfigFormat::Yaml),
            Some("json") => Ok(ConfigFormat::Json),
            Some("toml") => Ok(ConfigFormat::Toml),
            _ => Err(ConfigError::UnsupportedFormat(
                path.to_string_lossy().to_string()
            )),
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    IoError(std::io::Error),
    ParseError { format: String, message: String },
    ValidationError(Vec<ValidationError>),
    UnsupportedFormat(String),
    MigrationError(String),
    NotFound(PathBuf),
    InvalidEnvironmentVariable { key: String, message: String },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::IoError(e) => write!(f, "I/O error: {}", e),
            ConfigError::ParseError { format, message } => {
                write!(f, "Failed to parse {} config: {}", format, message)
            }
            ConfigError::ValidationError(errors) => {
                writeln!(f, "Configuration validation failed with {} error(s):", errors.len())?;
                for error in errors {
                    writeln!(f, "  - {}", error)?;
                }
                Ok(())
            }
            ConfigError::UnsupportedFormat(path) => {
                write!(f, "Unsupported configuration format: {}", path)
            }
            ConfigError::MigrationError(msg) => write!(f, "Migration error: {}", msg),
            ConfigError::NotFound(path) => write!(f, "Configuration file not found: {}", path.display()),
            ConfigError::InvalidEnvironmentVariable { key, message } => {
                write!(f, "Invalid environment variable {}: {}", key, message)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self {
        ConfigError::IoError(err)
    }
}

/// Configuration loader with multi-format support
pub struct ConfigLoader {
    search_paths: Vec<PathBuf>,
    env_prefix: String,
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self {
            search_paths: vec![
                PathBuf::from("."),
                PathBuf::from("/etc/llm-simulator"),
                PathBuf::from("~/.config/llm-simulator"),
            ],
            env_prefix: "LLM_SIM".to_string(),
        }
    }

    pub fn with_search_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.search_paths = paths;
        self
    }

    pub fn with_env_prefix(mut self, prefix: String) -> Self {
        self.env_prefix = prefix;
        self
    }

    /// Load configuration from file
    pub fn load_from_file(&self, path: &Path) -> Result<SimulatorConfig, ConfigError> {
        if !path.exists() {
            return Err(ConfigError::NotFound(path.to_path_buf()));
        }

        let content = std::fs::read_to_string(path)?;
        let format = ConfigFormat::from_path(path)?;

        self.parse_config(&content, format)
    }

    /// Parse configuration from string
    fn parse_config(&self, content: &str, format: ConfigFormat) -> Result<SimulatorConfig, ConfigError> {
        let config: SimulatorConfig = match format {
            ConfigFormat::Yaml => {
                serde_yaml::from_str(content).map_err(|e| ConfigError::ParseError {
                    format: "YAML".to_string(),
                    message: e.to_string(),
                })?
            }
            ConfigFormat::Json => {
                serde_json::from_str(content).map_err(|e| ConfigError::ParseError {
                    format: "JSON".to_string(),
                    message: e.to_string(),
                })?
            }
            ConfigFormat::Toml => {
                toml::from_str(content).map_err(|e| ConfigError::ParseError {
                    format: "TOML".to_string(),
                    message: e.to_string(),
                })?
            }
        };

        Ok(config)
    }

    /// Load configuration with automatic discovery
    pub fn load(&self) -> Result<SimulatorConfig, ConfigError> {
        // Try to find config file in search paths
        let config_names = ["simulator.yaml", "simulator.yml", "simulator.json", "simulator.toml"];

        let mut base_config = None;
        for search_path in &self.search_paths {
            for config_name in &config_names {
                let path = search_path.join(config_name);
                if path.exists() {
                    base_config = Some(self.load_from_file(&path)?);
                    break;
                }
            }
            if base_config.is_some() {
                break;
            }
        }

        let mut config = base_config.unwrap_or_default();

        // Load local overrides if present
        for search_path in &self.search_paths {
            let local_path = search_path.join("simulator.local.yaml");
            if local_path.exists() {
                let local_config = self.load_from_file(&local_path)?;
                config = ConfigMerger::merge(config, local_config);
            }
        }

        Ok(config)
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&self, mut config: SimulatorConfig) -> Result<SimulatorConfig, ConfigError> {
        let env_parser = EnvironmentParser::new(&self.env_prefix);
        env_parser.apply_to_config(&mut config)?;
        Ok(config)
    }

    /// Load configuration with full cascade
    pub fn load_with_overrides(&self, cli_args: Option<&CliArgs>) -> Result<SimulatorConfig, ConfigError> {
        // 1. Load base config from file
        let mut config = self.load()?;

        // 2. Apply environment variables
        config = self.apply_env_overrides(config)?;

        // 3. Apply CLI arguments
        if let Some(args) = cli_args {
            config = CliArgsApplicator::apply(config, args);
        }

        // 4. Validate final configuration
        config.validate().map_err(ConfigError::ValidationError)?;

        // 5. Migrate if needed
        config = ConfigMigrator::migrate(config)?;

        Ok(config)
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            version: default_version(),
            server: ServerConfig::default(),
            providers: HashMap::new(),
            simulation: SimulationConfig::default(),
            telemetry: TelemetryConfig::default(),
            scenarios: HashMap::new(),
            security: SecurityConfig::default(),
            features: FeatureFlags::default(),
        }
    }
}

// ============================================================================
// Environment Variable Parsing
// ============================================================================

/// Environment variable parser with type-safe mappings
struct EnvironmentParser {
    prefix: String,
}

impl EnvironmentParser {
    fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    fn apply_to_config(&self, config: &mut SimulatorConfig) -> Result<(), ConfigError> {
        // Server config
        self.apply_if_present("HOST", |v| config.server.host = v)?;
        self.apply_if_present_parse("PORT", |v| config.server.port = v)?;
        self.apply_if_present_parse("MAX_CONNECTIONS", |v| config.server.max_connections = v)?;
        self.apply_if_present_parse("REQUEST_TIMEOUT", |v| config.server.request_timeout_secs = v)?;

        // Simulation config
        self.apply_if_present_parse("SEED", |v| config.simulation.seed = Some(v))?;
        self.apply_if_present_parse("MAX_SESSIONS", |v| config.simulation.max_concurrent_sessions = v)?;

        // Telemetry config
        self.apply_if_present("LOG_LEVEL", |v| config.telemetry.logging.level = v)?;
        self.apply_if_present("LOG_FORMAT", |v| config.telemetry.logging.format = v)?;
        self.apply_if_present_bool("METRICS_ENABLED", |v| config.telemetry.metrics.enabled = v)?;
        self.apply_if_present("METRICS_ENDPOINT", |v| config.telemetry.metrics.endpoint = v)?;

        // TLS config
        if let Some(cert_path) = std::env::var(format!("{}_TLS_CERT", self.prefix)).ok() {
            if config.server.tls.is_none() {
                config.server.tls = Some(TlsConfig {
                    cert_path: PathBuf::from(&cert_path),
                    key_path: PathBuf::new(),
                    min_version: default_min_tls_version(),
                    client_auth: ClientAuthMode::default(),
                });
            }
            if let Some(ref mut tls) = config.server.tls {
                tls.cert_path = PathBuf::from(cert_path);
            }
        }

        if let Some(key_path) = std::env::var(format!("{}_TLS_KEY", self.prefix)).ok() {
            if let Some(ref mut tls) = config.server.tls {
                tls.key_path = PathBuf::from(key_path);
            }
        }

        Ok(())
    }

    fn apply_if_present<F>(&self, suffix: &str, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(String),
    {
        if let Ok(value) = std::env::var(format!("{}_{}", self.prefix, suffix)) {
            f(value);
        }
        Ok(())
    }

    fn apply_if_present_parse<T, F>(&self, suffix: &str, f: F) -> Result<(), ConfigError>
    where
        T: std::str::FromStr,
        F: FnOnce(T),
        <T as std::str::FromStr>::Err: fmt::Display,
    {
        if let Ok(value) = std::env::var(format!("{}_{}", self.prefix, suffix)) {
            let parsed = value.parse::<T>().map_err(|e| ConfigError::InvalidEnvironmentVariable {
                key: format!("{}_{}", self.prefix, suffix),
                message: e.to_string(),
            })?;
            f(parsed);
        }
        Ok(())
    }

    fn apply_if_present_bool<F>(&self, suffix: &str, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(bool),
    {
        if let Ok(value) = std::env::var(format!("{}_{}", self.prefix, suffix)) {
            let parsed = match value.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => true,
                "false" | "0" | "no" | "off" => false,
                _ => return Err(ConfigError::InvalidEnvironmentVariable {
                    key: format!("{}_{}", self.prefix, suffix),
                    message: format!("Invalid boolean value: {}", value),
                }),
            };
            f(parsed);
        }
        Ok(())
    }
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

/// CLI arguments structure
#[derive(Debug, Clone, Default)]
pub struct CliArgs {
    pub config_file: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub log_level: Option<String>,
    pub enable_metrics: Option<bool>,
    pub enable_tracing: Option<bool>,
    pub seed: Option<u64>,
    pub profile_overrides: HashMap<String, String>,
}

/// Applicator for CLI arguments
struct CliArgsApplicator;

impl CliArgsApplicator {
    fn apply(mut config: SimulatorConfig, args: &CliArgs) -> SimulatorConfig {
        if let Some(ref host) = args.host {
            config.server.host = host.clone();
        }
        if let Some(port) = args.port {
            config.server.port = port;
        }
        if let Some(ref log_level) = args.log_level {
            config.telemetry.logging.level = log_level.clone();
        }
        if let Some(enable_metrics) = args.enable_metrics {
            config.telemetry.metrics.enabled = enable_metrics;
        }
        if let Some(enable_tracing) = args.enable_tracing {
            config.telemetry.tracing.enabled = enable_tracing;
        }
        if let Some(seed) = args.seed {
            config.simulation.seed = Some(seed);
        }

        config
    }
}

// ============================================================================
// Configuration Merging
// ============================================================================

/// Hierarchical configuration merger
struct ConfigMerger;

impl ConfigMerger {
    fn merge(base: SimulatorConfig, override_config: SimulatorConfig) -> SimulatorConfig {
        // For simplicity, this is a shallow merge
        // In production, you'd want a deep merge strategy
        SimulatorConfig {
            version: override_config.version,
            server: Self::merge_server(base.server, override_config.server),
            providers: Self::merge_maps(base.providers, override_config.providers),
            simulation: base.simulation, // Override would replace entirely
            telemetry: Self::merge_telemetry(base.telemetry, override_config.telemetry),
            scenarios: Self::merge_maps(base.scenarios, override_config.scenarios),
            security: base.security,
            features: base.features,
        }
    }

    fn merge_server(base: ServerConfig, override_cfg: ServerConfig) -> ServerConfig {
        ServerConfig {
            host: if override_cfg.host != default_host() { override_cfg.host } else { base.host },
            port: if override_cfg.port != default_port() { override_cfg.port } else { base.port },
            max_connections: override_cfg.max_connections,
            request_timeout_secs: override_cfg.request_timeout_secs,
            keepalive_timeout_secs: override_cfg.keepalive_timeout_secs,
            tls: override_cfg.tls.or(base.tls),
            cors: override_cfg.cors,
            rate_limit: override_cfg.rate_limit,
            shutdown_timeout_secs: override_cfg.shutdown_timeout_secs,
        }
    }

    fn merge_telemetry(base: TelemetryConfig, override_cfg: TelemetryConfig) -> TelemetryConfig {
        TelemetryConfig {
            enabled: override_cfg.enabled,
            service_name: override_cfg.service_name,
            logging: override_cfg.logging,
            metrics: override_cfg.metrics,
            tracing: override_cfg.tracing,
        }
    }

    fn merge_maps<K, V>(mut base: HashMap<K, V>, override_map: HashMap<K, V>) -> HashMap<K, V>
    where
        K: Eq + std::hash::Hash,
    {
        base.extend(override_map);
        base
    }
}

// ============================================================================
// Configuration Hot-Reload System
// ============================================================================

/// Configuration watcher for hot-reload support
pub struct ConfigWatcher {
    config_path: PathBuf,
    current_config: Arc<RwLock<SimulatorConfig>>,
    loader: ConfigLoader,
    poll_interval: Duration,
    last_modified: SystemTime,
}

impl ConfigWatcher {
    pub fn new(
        config_path: PathBuf,
        initial_config: SimulatorConfig,
        loader: ConfigLoader,
    ) -> Result<Self, ConfigError> {
        let metadata = std::fs::metadata(&config_path)?;
        let last_modified = metadata.modified()?;

        Ok(Self {
            config_path,
            current_config: Arc::new(RwLock::new(initial_config)),
            loader,
            poll_interval: Duration::from_secs(5),
            last_modified,
        })
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Get current configuration (read-only)
    pub fn get_config(&self) -> SimulatorConfig {
        self.current_config.read().unwrap().clone()
    }

    /// Get shared reference to configuration
    pub fn get_config_ref(&self) -> Arc<RwLock<SimulatorConfig>> {
        Arc::clone(&self.current_config)
    }

    /// Check for configuration changes and reload if needed
    pub fn check_and_reload(&mut self) -> Result<bool, ConfigError> {
        let metadata = std::fs::metadata(&self.config_path)?;
        let modified = metadata.modified()?;

        if modified > self.last_modified {
            println!("Configuration file changed, reloading...");

            // Load new configuration
            let new_config = self.loader.load_from_file(&self.config_path)?;

            // Validate before applying
            new_config.validate().map_err(ConfigError::ValidationError)?;

            // Update configuration
            let mut config = self.current_config.write().unwrap();
            *config = new_config;

            self.last_modified = modified;
            println!("Configuration reloaded successfully");

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Start watching in background (spawns thread)
    pub fn watch<F>(mut self, on_reload: F) -> std::thread::JoinHandle<()>
    where
        F: Fn(&SimulatorConfig) + Send + 'static,
    {
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(self.poll_interval);

                match self.check_and_reload() {
                    Ok(true) => {
                        let config = self.get_config();
                        on_reload(&config);
                    }
                    Ok(false) => {
                        // No changes
                    }
                    Err(e) => {
                        eprintln!("Error reloading configuration: {}", e);
                    }
                }
            }
        })
    }
}

// ============================================================================
// Configuration Migration System
// ============================================================================

/// Configuration migrator for version upgrades
struct ConfigMigrator;

impl ConfigMigrator {
    fn migrate(config: SimulatorConfig) -> Result<SimulatorConfig, ConfigError> {
        let current_version = &config.version;
        let target_version = "1.0";

        if current_version == target_version {
            return Ok(config);
        }

        // Migration chain
        let mut migrated = config;

        // Example migration: 0.9 -> 1.0
        if current_version == "0.9" {
            migrated = Self::migrate_0_9_to_1_0(migrated)?;
        }

        // Validate migrated config
        migrated.validate().map_err(ConfigError::ValidationError)?;

        Ok(migrated)
    }

    fn migrate_0_9_to_1_0(mut config: SimulatorConfig) -> Result<SimulatorConfig, ConfigError> {
        // Example migration logic
        config.version = "1.0".to_string();

        // Add new default feature flags if missing
        if config.features.custom.is_empty() {
            // Migration logic here
        }

        println!("Migrated configuration from version 0.9 to 1.0");
        Ok(config)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Pretty-print validation errors
pub fn print_validation_errors(errors: &[ValidationError]) {
    eprintln!("\n❌ Configuration validation failed with {} error(s):\n", errors.len());
    for (idx, error) in errors.iter().enumerate() {
        eprintln!("{}. Field: {}", idx + 1, error.field);
        eprintln!("   Message: {}", error.message);
        eprintln!("   Constraint: {}", error.constraint);
        eprintln!("   Actual: {}\n", error.actual_value);
    }
}

// ============================================================================
// Usage Examples
// ============================================================================

#[cfg(test)]
mod examples {
    use super::*;

    fn demonstrate_config_system() {
        // Example 1: Load configuration from file
        println!("=== Loading Configuration ===");
        let loader = ConfigLoader::new()
            .with_search_paths(vec![PathBuf::from("./config")])
            .with_env_prefix("LLM_SIM".to_string());

        match loader.load() {
            Ok(config) => {
                println!("✓ Configuration loaded successfully");
                println!("  Server: {}:{}", config.server.host, config.server.port);
                println!("  Providers: {}", config.providers.len());
            }
            Err(e) => {
                eprintln!("✗ Failed to load configuration: {}", e);
            }
        }

        // Example 2: Load with CLI overrides
        println!("\n=== Loading with Overrides ===");
        let cli_args = CliArgs {
            port: Some(9090),
            log_level: Some("debug".to_string()),
            ..Default::default()
        };

        match loader.load_with_overrides(Some(&cli_args)) {
            Ok(config) => {
                println!("✓ Configuration loaded with overrides");
                println!("  Port (overridden): {}", config.server.port);
                println!("  Log level (overridden): {}", config.telemetry.logging.level);
            }
            Err(e) => {
                eprintln!("✗ Failed to load configuration: {}", e);
            }
        }

        // Example 3: Hot-reload demonstration
        println!("\n=== Hot-Reload Setup ===");
        let config_path = PathBuf::from("./config/simulator.yaml");
        let initial_config = loader.load().unwrap_or_default();

        match ConfigWatcher::new(config_path, initial_config, loader) {
            Ok(watcher) => {
                println!("✓ Config watcher initialized");

                let _handle = watcher.watch(|config| {
                    println!("🔄 Configuration reloaded!");
                    println!("  Active providers: {}", config.providers.len());
                });

                println!("  Watching for changes...");
            }
            Err(e) => {
                eprintln!("✗ Failed to create watcher: {}", e);
            }
        }

        // Example 4: Validation errors
        println!("\n=== Validation Example ===");
        let mut invalid_config = SimulatorConfig::default();
        invalid_config.server.port = 0; // Invalid!
        invalid_config.telemetry.logging.level = "invalid".to_string(); // Invalid!

        match invalid_config.validate() {
            Ok(_) => println!("✓ Configuration is valid"),
            Err(errors) => {
                print_validation_errors(&errors);
            }
        }

        // Example 5: Programmatic configuration
        println!("\n=== Programmatic Configuration ===");
        let mut config = SimulatorConfig::default();
        config.server.port = 8080;
        config.server.host = "0.0.0.0".to_string();

        // Add a provider profile
        config.providers.insert(
            "gpt-4-turbo".to_string(),
            ProviderProfile {
                name: "GPT-4 Turbo".to_string(),
                provider: "OpenAI".to_string(),
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

        if config.validate().is_ok() {
            println!("✓ Programmatically created configuration is valid");
            println!("  Server configured: {}:{}", config.server.host, config.server.port);
            println!("  Profiles configured: {}", config.providers.len());
        }
    }
}

// ============================================================================
// Production Integration Notes
// ============================================================================

/*
CONFIGURATION MANAGEMENT INTEGRATION CHECKLIST:

1. File Formats:
   ✓ YAML support (primary format, human-friendly)
   ✓ JSON support (API-friendly, strict parsing)
   ✓ TOML support (alternative, Rust-native)
   ✓ Automatic format detection from file extension

2. Configuration Sources (Precedence Order):
   ✓ CLI arguments (highest precedence)
   ✓ Environment variables
   ✓ Local config file (simulator.local.yaml)
   ✓ Main config file (simulator.yaml)
   ✓ Built-in defaults (lowest precedence)

3. Environment Variables:
   ✓ Prefix-based (LLM_SIM_*)
   ✓ Type-safe parsing
   ✓ Helpful error messages
   ✓ Support for nested config via underscore notation

4. Hot-Reload:
   ✓ File system polling
   ✓ Thread-safe configuration updates
   ✓ Validation before applying
   ✓ Callback system for reload notifications
   ✓ Configurable poll interval

5. Validation:
   ✓ Comprehensive schema validation
   ✓ Detailed error messages with field paths
   ✓ Constraint documentation
   ✓ Actual value reporting
   ✓ Batch error reporting (all errors at once)

6. Migration:
   ✓ Version tracking
   ✓ Automatic migration chains
   ✓ Validation after migration
   ✓ Migration error handling

7. CLI Integration:
   - Integrate with clap or structopt for argument parsing
   - Support --config flag for custom config file
   - Support individual override flags (--port, --host, etc.)
   - Generate help text from schema

8. Security:
   ✓ API key configuration
   ✓ JWT/OAuth2 configuration
   ✓ TLS certificate paths
   ✓ Secret rotation support
   - Consider using secret management service (Vault, AWS Secrets Manager)
   - Encrypt sensitive config values at rest

9. Documentation:
   - Auto-generate config reference from schema
   - Provide example configurations for common use cases
   - Document environment variable mappings
   - Create migration guide for version upgrades

10. Testing:
    - Unit tests for each validator
    - Integration tests for config loading
    - Test all format parsers (YAML/JSON/TOML)
    - Test environment variable parsing
    - Test configuration merging logic
    - Test hot-reload mechanism
    - Test migration paths

11. Observability:
    - Log configuration loading events
    - Metric for config reload count
    - Metric for validation errors
    - Trace config source hierarchy
    - Alert on validation failures

12. Best Practices:
    - Use deny_unknown_fields to catch typos
    - Provide sensible defaults
    - Make breaking changes opt-in via feature flags
    - Version configuration schema
    - Support configuration profiles (dev, staging, prod)
    - Implement configuration dry-run mode
    - Add configuration dump command for debugging

EXAMPLE CONFIGURATION FILES:

simulator.yaml (Main Config):
---
version: "1.0"

server:
  host: "127.0.0.1"
  port: 8080
  max_connections: 10000
  request_timeout_secs: 300

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

simulation:
  max_concurrent_sessions: 1000
  session_timeout_secs: 3600

telemetry:
  enabled: true
  logging:
    level: "info"
    format: "json"
    output: "stdout"
  metrics:
    enabled: true
    exporter: "prometheus"
    endpoint: "/metrics"

.env (Environment Variables):
LLM_SIM_HOST=0.0.0.0
LLM_SIM_PORT=9090
LLM_SIM_LOG_LEVEL=debug
LLM_SIM_METRICS_ENABLED=true

CLI Usage:
./llm-simulator \
  --config /etc/llm-simulator/simulator.yaml \
  --port 9090 \
  --log-level debug \
  --enable-metrics
*/
