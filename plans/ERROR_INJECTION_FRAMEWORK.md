# Error Injection Framework - Production-Ready Pseudocode

> **Module**: LLM-Simulator - Error Injection & Chaos Engineering
> **Author**: Senior Rust Systems Architect
> **Version**: 1.0.0
> **Date**: 2025-11-26
> **Status**: Production-Ready Design

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Types and Configuration](#2-core-types-and-configuration)
3. [Error Injection Strategies](#3-error-injection-strategies)
4. [Provider-Specific Error Formatters](#4-provider-specific-error-formatters)
5. [Circuit Breaker Simulation](#5-circuit-breaker-simulation)
6. [Chaos Engineering Scenarios](#6-chaos-engineering-scenarios)
7. [Retry Header Generation](#7-retry-header-generation)
8. [Integration Points](#8-integration-points)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Error Injection Framework                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐      ┌──────────────────────────────────┐  │
│  │ Request Context│─────▶│   Injection Orchestrator         │  │
│  │ - Request ID   │      │   - Strategy Selection           │  │
│  │ - Timestamp    │      │   - Probability Evaluation       │  │
│  │ - Provider     │      │   - Sequence Pattern Matching    │  │
│  │ - Model        │      └──────────────┬───────────────────┘  │
│  └────────────────┘                     │                       │
│                                         ▼                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Error Injection Strategies                     │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ • ProbabilisticStrategy    │ • SequenceStrategy         │  │
│  │ • TimeBasedStrategy        │ • BudgetExhaustion         │  │
│  │ • ConditionalStrategy      │ • LoadDependentStrategy    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                         │                       │
│                                         ▼                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Provider-Specific Error Formatters              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ OpenAI │ Anthropic │ Google │ Azure │ Cohere │ Custom   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                         │                       │
│                                         ▼                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          HTTP Response Builder                           │  │
│  │  - Status Code        - Retry Headers                    │  │
│  │  - Error Body         - Rate Limit Headers               │  │
│  │  - Correlation IDs    - Custom Headers                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

              ┌────────────────────────────────┐
              │   Circuit Breaker Simulator    │
              │   - Open/Closed/Half-Open     │
              │   - Failure Threshold          │
              │   - Recovery Timeout           │
              └────────────────────────────────┘
```

---

## 2. Core Types and Configuration

### 2.1 Module Structure

```rust
// File: src/errors/mod.rs

pub mod injection;      // Error injection strategies and orchestration
pub mod patterns;       // Sequence and pattern matching
pub mod responses;      // Provider-specific error formatters
pub mod circuit_breaker;// Circuit breaker state machine
pub mod scenarios;      // Pre-configured chaos scenarios
pub mod headers;        // Retry and rate limit header generation

use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use rand::Rng;
use thiserror::Error;

/// Re-exports for convenient access
pub use injection::{
    ErrorInjector,
    ErrorInjectionConfig,
    ErrorInjectionStrategy,
    InjectedError,
};
pub use responses::{ErrorFormatter, ProviderErrorFormat};
pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use scenarios::ChaosScenario;
```

### 2.2 Error Injection Configuration

```rust
// File: src/errors/injection/config.rs

/// Complete configuration for error injection behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInjectionConfig {
    /// Global toggle for all error injection
    pub enabled: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// List of active injection strategies
    pub strategies: Vec<StrategyConfig>,

    /// Circuit breaker configuration
    pub circuit_breaker: Option<CircuitBreakerConfig>,

    /// Rate limit simulation parameters
    pub rate_limits: Option<RateLimitConfig>,

    /// Quota/budget simulation
    pub quota: Option<QuotaConfig>,

    /// Per-provider error customization
    pub provider_overrides: HashMap<Provider, ProviderErrorConfig>,

    /// Telemetry configuration for error events
    pub telemetry: TelemetryConfig,
}

impl Default for ErrorInjectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: None,
            strategies: vec![],
            circuit_breaker: None,
            rate_limits: None,
            quota: None,
            provider_overrides: HashMap::new(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

/// Strategy-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StrategyConfig {
    /// Inject errors based on probability
    Probabilistic {
        /// Error type to inject
        error_type: ErrorType,

        /// Probability [0.0, 1.0] of injection
        probability: f64,

        /// Optional time window for this strategy
        time_window: Option<TimeWindow>,

        /// Whether to continue after injection
        continue_on_inject: bool,
    },

    /// Inject errors in sequence patterns
    Sequence {
        /// Pattern definition (e.g., "fail every Nth request")
        pattern: SequencePattern,

        /// Error type to inject
        error_type: ErrorType,

        /// Maximum injections (None = unlimited)
        max_injections: Option<usize>,
    },

    /// Inject errors based on time conditions
    TimeBased {
        /// Trigger condition
        trigger: TimeTrigger,

        /// Error type to inject
        error_type: ErrorType,

        /// Duration of error injection
        duration: Duration,
    },

    /// Inject based on request characteristics
    Conditional {
        /// Condition to evaluate
        condition: RequestCondition,

        /// Error type to inject when condition matches
        error_type: ErrorType,

        /// Probability when condition is true
        probability: f64,
    },

    /// Simulate budget/quota exhaustion
    BudgetExhaustion {
        /// Token budget limit
        token_limit: u64,

        /// Reset period
        reset_period: Duration,

        /// Error type when budget exceeded
        error_type: ErrorType,
    },

    /// Inject errors based on system load
    LoadDependent {
        /// Concurrent request threshold
        concurrent_threshold: usize,

        /// Error type to inject when threshold exceeded
        error_type: ErrorType,

        /// Probability multiplier based on load
        load_multiplier: f64,
    },
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: usize,

    /// Time window for counting failures
    pub window_duration: Duration,

    /// Duration to keep circuit open
    pub open_duration: Duration,

    /// Number of successful requests in half-open state before closing
    pub half_open_success_threshold: usize,

    /// Error type to return when circuit is open
    pub open_circuit_error: ErrorType,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute limit
    pub requests_per_minute: u32,

    /// Tokens per minute limit
    pub tokens_per_minute: Option<u64>,

    /// Concurrent request limit
    pub max_concurrent_requests: Option<u32>,

    /// Whether to track limits per API key
    pub per_key_limits: bool,

    /// Behavior when limit is exceeded
    pub exceed_behavior: ExceedBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceedBehavior {
    /// Return 429 error immediately
    Reject,

    /// Queue request and introduce delay
    Queue { max_queue_size: usize },

    /// Probabilistic rejection based on current load
    Probabilistic { base_probability: f64 },
}

/// Quota/budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaConfig {
    /// Daily token quota
    pub daily_token_quota: Option<u64>,

    /// Monthly token quota
    pub monthly_token_quota: Option<u64>,

    /// Reset time (UTC hour)
    pub reset_hour_utc: u8,

    /// Current usage (simulated)
    #[serde(skip)]
    pub current_usage: Arc<AtomicU64>,

    /// Last reset timestamp
    #[serde(skip)]
    pub last_reset: Arc<RwLock<Instant>>,
}

/// Per-provider error customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderErrorConfig {
    /// Override error messages
    pub custom_error_messages: HashMap<ErrorType, String>,

    /// Additional HTTP headers
    pub additional_headers: HashMap<String, String>,

    /// Custom retry delay calculation
    pub retry_delay_strategy: RetryDelayStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryDelayStrategy {
    /// Fixed delay
    Fixed { seconds: u32 },

    /// Exponential backoff
    Exponential { base_ms: u32, max_ms: u32 },

    /// Provider-specific default
    ProviderDefault,
}
```

### 2.3 Error Types and Injected Errors

```rust
// File: src/errors/types.rs

/// Comprehensive error types matching real LLM provider errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    // Rate Limiting Errors (429)
    RateLimit,
    RateLimitRPM,        // Requests per minute
    RateLimitTPM,        // Tokens per minute
    RateLimitConcurrent, // Concurrent requests

    // Authentication/Authorization Errors
    AuthenticationFailed,  // 401
    InvalidApiKey,         // 401
    PermissionDenied,      // 403
    InsufficientQuota,     // 403

    // Bad Request Errors (400)
    BadRequest,
    InvalidParameter,
    MissingParameter,
    ContextLengthExceeded,
    TokenLimitExceeded,
    ContentFilterTriggered,
    InvalidModel,
    InvalidFormat,

    // Server Errors (5xx)
    InternalServerError,   // 500
    BadGateway,            // 502
    ServiceUnavailable,    // 503
    GatewayTimeout,        // 504

    // Timeout Errors
    RequestTimeout,        // 408
    UpstreamTimeout,       // 504

    // Quota Errors
    QuotaExceeded,         // 429
    DailyQuotaExceeded,    // 429
    MonthlyQuotaExceeded,  // 429

    // Network Errors
    ConnectionError,
    DNSResolutionFailed,
    SSLHandshakeFailed,

    // Degraded Performance
    SlowResponse,          // Not an error, but increased latency
    PartialResponse,       // Incomplete response

    // Circuit Breaker
    CircuitBreakerOpen,    // 503
}

impl ErrorType {
    /// Get the HTTP status code for this error type
    pub fn http_status_code(&self) -> u16 {
        match self {
            // 400 Bad Request
            Self::BadRequest
            | Self::InvalidParameter
            | Self::MissingParameter
            | Self::ContextLengthExceeded
            | Self::TokenLimitExceeded
            | Self::ContentFilterTriggered
            | Self::InvalidModel
            | Self::InvalidFormat => 400,

            // 401 Unauthorized
            Self::AuthenticationFailed | Self::InvalidApiKey => 401,

            // 403 Forbidden
            Self::PermissionDenied | Self::InsufficientQuota => 403,

            // 408 Request Timeout
            Self::RequestTimeout => 408,

            // 429 Too Many Requests
            Self::RateLimit
            | Self::RateLimitRPM
            | Self::RateLimitTPM
            | Self::RateLimitConcurrent
            | Self::QuotaExceeded
            | Self::DailyQuotaExceeded
            | Self::MonthlyQuotaExceeded => 429,

            // 500 Internal Server Error
            Self::InternalServerError => 500,

            // 502 Bad Gateway
            Self::BadGateway => 502,

            // 503 Service Unavailable
            Self::ServiceUnavailable | Self::CircuitBreakerOpen => 503,

            // 504 Gateway Timeout
            Self::GatewayTimeout | Self::UpstreamTimeout => 504,

            // Network errors (no HTTP status)
            Self::ConnectionError
            | Self::DNSResolutionFailed
            | Self::SSLHandshakeFailed => 0,

            // Performance degradation (not errors)
            Self::SlowResponse | Self::PartialResponse => 200,
        }
    }

    /// Determine if this error should trigger retry behavior
    pub fn is_retryable(&self) -> bool {
        match self {
            // Retryable errors
            Self::RateLimit
            | Self::RateLimitRPM
            | Self::RateLimitTPM
            | Self::RateLimitConcurrent
            | Self::InternalServerError
            | Self::BadGateway
            | Self::ServiceUnavailable
            | Self::GatewayTimeout
            | Self::RequestTimeout
            | Self::UpstreamTimeout
            | Self::ConnectionError => true,

            // Non-retryable errors
            Self::AuthenticationFailed
            | Self::InvalidApiKey
            | Self::PermissionDenied
            | Self::BadRequest
            | Self::InvalidParameter
            | Self::MissingParameter
            | Self::ContextLengthExceeded
            | Self::TokenLimitExceeded
            | Self::ContentFilterTriggered
            | Self::InvalidModel
            | Self::InvalidFormat
            | Self::InsufficientQuota
            | Self::QuotaExceeded
            | Self::DailyQuotaExceeded
            | Self::MonthlyQuotaExceeded
            | Self::DNSResolutionFailed
            | Self::SSLHandshakeFailed
            | Self::CircuitBreakerOpen
            | Self::SlowResponse
            | Self::PartialResponse => false,
        }
    }

    /// Get recommended retry delay in seconds
    pub fn suggested_retry_delay(&self) -> Option<u32> {
        match self {
            Self::RateLimit
            | Self::RateLimitRPM
            | Self::RateLimitTPM
            | Self::RateLimitConcurrent => Some(60),

            Self::QuotaExceeded => Some(3600), // 1 hour
            Self::DailyQuotaExceeded => Some(86400), // 24 hours
            Self::MonthlyQuotaExceeded => Some(2592000), // 30 days

            Self::ServiceUnavailable
            | Self::BadGateway
            | Self::InternalServerError => Some(5),

            Self::GatewayTimeout
            | Self::RequestTimeout
            | Self::UpstreamTimeout => Some(10),

            Self::ConnectionError => Some(2),

            _ => None,
        }
    }
}

/// Complete injected error with all metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectedError {
    /// Type of error injected
    pub error_type: ErrorType,

    /// HTTP status code
    pub status_code: u16,

    /// Error message (provider-specific format)
    pub message: String,

    /// Error code (provider-specific)
    pub error_code: Option<String>,

    /// Retry-After header value (seconds)
    pub retry_after: Option<u32>,

    /// Additional HTTP headers
    pub headers: HashMap<String, String>,

    /// Rate limit information
    pub rate_limit_info: Option<RateLimitInfo>,

    /// Request ID for tracking
    pub request_id: String,

    /// Timestamp when error was injected
    pub timestamp: Instant,

    /// Provider that would have generated this error
    pub provider: Provider,

    /// Strategy that injected this error
    pub injected_by: String,

    /// Whether client should retry
    pub retryable: bool,

    /// Additional context for debugging
    pub context: HashMap<String, String>,
}

/// Rate limit information for headers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    /// Requests remaining in current window
    pub remaining_requests: Option<u32>,

    /// Tokens remaining in current window
    pub remaining_tokens: Option<u64>,

    /// Total request limit per window
    pub limit_requests: Option<u32>,

    /// Total token limit per window
    pub limit_tokens: Option<u64>,

    /// Timestamp when limits reset (Unix epoch seconds)
    pub reset_timestamp: u64,

    /// Seconds until reset
    pub reset_seconds: u32,
}

/// Request context for error injection decisions
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request identifier
    pub request_id: String,

    /// Request sequence number (for pattern matching)
    pub sequence_number: u64,

    /// Timestamp when request was received
    pub timestamp: Instant,

    /// Target provider
    pub provider: Provider,

    /// Model being requested
    pub model: String,

    /// Estimated prompt tokens
    pub prompt_tokens: Option<u64>,

    /// Estimated completion tokens
    pub max_tokens: Option<u64>,

    /// API key (hashed for privacy)
    pub api_key_hash: Option<String>,

    /// Current concurrent request count
    pub concurrent_requests: usize,

    /// Request headers
    pub headers: HashMap<String, String>,

    /// Request body size
    pub body_size: usize,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Provider enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    Azure,
    Cohere,
    Custom(u32), // Custom provider ID
}
```

---

## 3. Error Injection Strategies

### 3.1 Core Strategy Trait

```rust
// File: src/errors/injection/strategy.rs

use async_trait::async_trait;
use std::fmt::Debug;

/// Trait for error injection strategies
#[async_trait]
pub trait ErrorInjectionStrategy: Send + Sync + Debug {
    /// Evaluate whether to inject an error for this request
    ///
    /// Returns Some(InjectedError) if error should be injected,
    /// None if request should proceed normally.
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError>;

    /// Reset strategy state (for testing/reproducibility)
    async fn reset(&mut self);

    /// Get strategy name for telemetry
    fn name(&self) -> &str;

    /// Get strategy configuration
    fn config(&self) -> StrategyConfig;
}

#[derive(Debug, Error)]
pub enum StrategyError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("State error: {0}")]
    StateError(String),

    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}
```

### 3.2 Probabilistic Strategy

```rust
// File: src/errors/injection/probabilistic.rs

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::sync::Mutex;

/// Probabilistic error injection strategy
#[derive(Debug)]
pub struct ProbabilisticStrategy {
    /// Error type to inject
    error_type: ErrorType,

    /// Injection probability [0.0, 1.0]
    probability: f64,

    /// Optional time window
    time_window: Option<TimeWindow>,

    /// Whether to continue checking other strategies after injection
    continue_on_inject: bool,

    /// Random number generator (seeded for reproducibility)
    rng: Mutex<ChaCha8Rng>,

    /// Error formatter for this error type
    formatter: Arc<dyn ErrorFormatter>,
}

impl ProbabilisticStrategy {
    pub fn new(
        error_type: ErrorType,
        probability: f64,
        seed: Option<u64>,
        formatter: Arc<dyn ErrorFormatter>,
    ) -> Result<Self, StrategyError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(StrategyError::ConfigError(
                format!("Probability must be in [0.0, 1.0], got {}", probability)
            ));
        }

        let rng = if let Some(seed_val) = seed {
            ChaCha8Rng::seed_from_u64(seed_val)
        } else {
            ChaCha8Rng::from_entropy()
        };

        Ok(Self {
            error_type,
            probability,
            time_window: None,
            continue_on_inject: true,
            rng: Mutex::new(rng),
            formatter,
        })
    }

    pub fn with_time_window(mut self, window: TimeWindow) -> Self {
        self.time_window = Some(window);
        self
    }

    /// Check if current time is within the configured window
    fn is_in_time_window(&self, timestamp: Instant) -> bool {
        if let Some(window) = &self.time_window {
            window.contains(timestamp)
        } else {
            true // No window = always active
        }
    }
}

#[async_trait]
impl ErrorInjectionStrategy for ProbabilisticStrategy {
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError> {
        // Check time window
        if !self.is_in_time_window(context.timestamp) {
            return Ok(None);
        }

        // Roll the dice
        let roll = {
            let mut rng = self.rng.lock().unwrap();
            rng.gen::<f64>()
        };

        if roll < self.probability {
            // Generate error using formatter
            let error = self.formatter.format_error(
                self.error_type,
                context,
                None, // Use default message
            )?;

            Ok(Some(error))
        } else {
            Ok(None)
        }
    }

    async fn reset(&mut self) {
        // Re-seed RNG for reproducible testing
        let mut rng = self.rng.lock().unwrap();
        *rng = ChaCha8Rng::seed_from_u64(0);
    }

    fn name(&self) -> &str {
        "probabilistic"
    }

    fn config(&self) -> StrategyConfig {
        StrategyConfig::Probabilistic {
            error_type: self.error_type,
            probability: self.probability,
            time_window: self.time_window.clone(),
            continue_on_inject: self.continue_on_inject,
        }
    }
}

/// Time window for strategy activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start time relative to simulation start
    pub start: Duration,

    /// End time relative to simulation start
    pub end: Duration,
}

impl TimeWindow {
    pub fn contains(&self, timestamp: Instant) -> bool {
        // This would compare against simulation start time
        // Implementation depends on global simulation clock
        true // Placeholder
    }
}
```

### 3.3 Sequence Pattern Strategy

```rust
// File: src/errors/injection/sequence.rs

use std::sync::atomic::{AtomicU64, Ordering};

/// Sequence-based error injection strategy
#[derive(Debug)]
pub struct SequenceStrategy {
    /// Pattern to match
    pattern: SequencePattern,

    /// Error type to inject
    error_type: ErrorType,

    /// Maximum number of injections (None = unlimited)
    max_injections: Option<usize>,

    /// Counter for injections performed
    injection_count: AtomicU64,

    /// Error formatter
    formatter: Arc<dyn ErrorFormatter>,
}

/// Sequence patterns for error injection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SequencePattern {
    /// Fail every Nth request
    EveryNth {
        n: u64,
        #[serde(default)]
        offset: u64,
    },

    /// Fail requests in a specific range
    Range {
        start: u64,
        end: u64,
    },

    /// Fail specific request numbers
    Specific {
        request_numbers: Vec<u64>,
    },

    /// Fail in bursts: N failures, then M successes
    Burst {
        fail_count: u64,
        success_count: u64,
    },

    /// Fail first N requests
    First {
        count: u64,
    },

    /// Fail after N successful requests
    After {
        success_count: u64,
    },
}

impl SequenceStrategy {
    pub fn new(
        pattern: SequencePattern,
        error_type: ErrorType,
        formatter: Arc<dyn ErrorFormatter>,
    ) -> Self {
        Self {
            pattern,
            error_type,
            max_injections: None,
            injection_count: AtomicU64::new(0),
            formatter,
        }
    }

    pub fn with_max_injections(mut self, max: usize) -> Self {
        self.max_injections = Some(max);
        self
    }

    fn pattern_matches(&self, sequence_number: u64) -> bool {
        match &self.pattern {
            SequencePattern::EveryNth { n, offset } => {
                (sequence_number + offset) % n == 0
            }

            SequencePattern::Range { start, end } => {
                sequence_number >= *start && sequence_number < *end
            }

            SequencePattern::Specific { request_numbers } => {
                request_numbers.contains(&sequence_number)
            }

            SequencePattern::Burst { fail_count, success_count } => {
                let cycle_length = fail_count + success_count;
                let position = sequence_number % cycle_length;
                position < *fail_count
            }

            SequencePattern::First { count } => {
                sequence_number < *count
            }

            SequencePattern::After { success_count } => {
                sequence_number >= *success_count
            }
        }
    }
}

#[async_trait]
impl ErrorInjectionStrategy for SequenceStrategy {
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError> {
        // Check if we've reached max injections
        if let Some(max) = self.max_injections {
            let current = self.injection_count.load(Ordering::Relaxed);
            if current >= max as u64 {
                return Ok(None);
            }
        }

        // Check if pattern matches
        if self.pattern_matches(context.sequence_number) {
            // Increment counter
            self.injection_count.fetch_add(1, Ordering::Relaxed);

            // Format error
            let error = self.formatter.format_error(
                self.error_type,
                context,
                None,
            )?;

            Ok(Some(error))
        } else {
            Ok(None)
        }
    }

    async fn reset(&mut self) {
        self.injection_count.store(0, Ordering::Relaxed);
    }

    fn name(&self) -> &str {
        "sequence"
    }

    fn config(&self) -> StrategyConfig {
        StrategyConfig::Sequence {
            pattern: self.pattern.clone(),
            error_type: self.error_type,
            max_injections: self.max_injections,
        }
    }
}
```

### 3.4 Time-Based Strategy

```rust
// File: src/errors/injection/time_based.rs

/// Time-based error injection strategy
#[derive(Debug)]
pub struct TimeBasedStrategy {
    /// Trigger condition
    trigger: TimeTrigger,

    /// Error type to inject
    error_type: ErrorType,

    /// Duration of error injection
    duration: Duration,

    /// Activation timestamp (when trigger fires)
    activation_time: RwLock<Option<Instant>>,

    /// Error formatter
    formatter: Arc<dyn ErrorFormatter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TimeTrigger {
    /// Trigger at specific time offset from simulation start
    AtTime {
        offset: Duration,
    },

    /// Trigger after N requests
    AfterRequests {
        count: u64,
    },

    /// Trigger at random time within range
    RandomInRange {
        min_offset: Duration,
        max_offset: Duration,
    },

    /// Trigger on specific wall-clock time (UTC)
    WallClock {
        hour: u8,
        minute: u8,
        second: u8,
    },
}

impl TimeBasedStrategy {
    pub fn new(
        trigger: TimeTrigger,
        error_type: ErrorType,
        duration: Duration,
        formatter: Arc<dyn ErrorFormatter>,
    ) -> Self {
        Self {
            trigger,
            error_type,
            duration,
            activation_time: RwLock::new(None),
            formatter,
        }
    }

    fn is_active(&self, context: &RequestContext) -> bool {
        let activation = self.activation_time.read().unwrap();

        if let Some(activated_at) = *activation {
            // Check if we're still within the duration window
            context.timestamp.duration_since(activated_at) < self.duration
        } else {
            // Check if trigger condition is met
            match &self.trigger {
                TimeTrigger::AtTime { offset } => {
                    // Compare against simulation start time
                    // Placeholder: context.timestamp >= simulation_start + offset
                    false
                }

                TimeTrigger::AfterRequests { count } => {
                    context.sequence_number >= *count
                }

                TimeTrigger::RandomInRange { .. } => {
                    // Would be determined once at initialization
                    false
                }

                TimeTrigger::WallClock { hour, minute, second } => {
                    // Check current UTC time
                    use chrono::Utc;
                    let now = Utc::now();
                    now.hour() == *hour as u32
                        && now.minute() == *minute as u32
                        && now.second() == *second as u32
                }
            }
        }
    }

    fn activate(&self, timestamp: Instant) {
        let mut activation = self.activation_time.write().unwrap();
        if activation.is_none() {
            *activation = Some(timestamp);
        }
    }
}

#[async_trait]
impl ErrorInjectionStrategy for TimeBasedStrategy {
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError> {
        if self.is_active(context) {
            self.activate(context.timestamp);

            let error = self.formatter.format_error(
                self.error_type,
                context,
                None,
            )?;

            Ok(Some(error))
        } else {
            Ok(None)
        }
    }

    async fn reset(&mut self) {
        let mut activation = self.activation_time.write().unwrap();
        *activation = None;
    }

    fn name(&self) -> &str {
        "time_based"
    }

    fn config(&self) -> StrategyConfig {
        StrategyConfig::TimeBased {
            trigger: self.trigger.clone(),
            error_type: self.error_type,
            duration: self.duration,
        }
    }
}
```

### 3.5 Conditional Strategy

```rust
// File: src/errors/injection/conditional.rs

/// Conditional error injection based on request characteristics
#[derive(Debug)]
pub struct ConditionalStrategy {
    /// Condition to evaluate
    condition: RequestCondition,

    /// Error type when condition matches
    error_type: ErrorType,

    /// Probability when condition is true
    probability: f64,

    /// RNG for probability evaluation
    rng: Mutex<ChaCha8Rng>,

    /// Error formatter
    formatter: Arc<dyn ErrorFormatter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestCondition {
    /// Match specific model
    Model {
        model_pattern: String, // Regex pattern
    },

    /// Match based on token count
    TokenCount {
        min_tokens: Option<u64>,
        max_tokens: Option<u64>,
    },

    /// Match based on request body size
    BodySize {
        min_bytes: Option<usize>,
        max_bytes: Option<usize>,
    },

    /// Match specific headers
    Header {
        name: String,
        value_pattern: String, // Regex pattern
    },

    /// Match specific provider
    Provider {
        provider: Provider,
    },

    /// Match based on concurrent load
    ConcurrentRequests {
        min_concurrent: Option<usize>,
        max_concurrent: Option<usize>,
    },

    /// Composite AND condition
    And {
        conditions: Vec<RequestCondition>,
    },

    /// Composite OR condition
    Or {
        conditions: Vec<RequestCondition>,
    },

    /// Negation
    Not {
        condition: Box<RequestCondition>,
    },
}

impl RequestCondition {
    fn evaluate(&self, context: &RequestContext) -> bool {
        match self {
            Self::Model { model_pattern } => {
                regex::Regex::new(model_pattern)
                    .map(|re| re.is_match(&context.model))
                    .unwrap_or(false)
            }

            Self::TokenCount { min_tokens, max_tokens } => {
                let total_tokens = context.prompt_tokens.unwrap_or(0)
                    + context.max_tokens.unwrap_or(0);

                let min_ok = min_tokens.map(|min| total_tokens >= min).unwrap_or(true);
                let max_ok = max_tokens.map(|max| total_tokens <= max).unwrap_or(true);

                min_ok && max_ok
            }

            Self::BodySize { min_bytes, max_bytes } => {
                let min_ok = min_bytes.map(|min| context.body_size >= min).unwrap_or(true);
                let max_ok = max_bytes.map(|max| context.body_size <= max).unwrap_or(true);

                min_ok && max_ok
            }

            Self::Header { name, value_pattern } => {
                context.headers.get(name)
                    .and_then(|value| {
                        regex::Regex::new(value_pattern)
                            .ok()
                            .map(|re| re.is_match(value))
                    })
                    .unwrap_or(false)
            }

            Self::Provider { provider } => {
                context.provider == *provider
            }

            Self::ConcurrentRequests { min_concurrent, max_concurrent } => {
                let min_ok = min_concurrent
                    .map(|min| context.concurrent_requests >= min)
                    .unwrap_or(true);
                let max_ok = max_concurrent
                    .map(|max| context.concurrent_requests <= max)
                    .unwrap_or(true);

                min_ok && max_ok
            }

            Self::And { conditions } => {
                conditions.iter().all(|c| c.evaluate(context))
            }

            Self::Or { conditions } => {
                conditions.iter().any(|c| c.evaluate(context))
            }

            Self::Not { condition } => {
                !condition.evaluate(context)
            }
        }
    }
}

#[async_trait]
impl ErrorInjectionStrategy for ConditionalStrategy {
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError> {
        // Evaluate condition
        if !self.condition.evaluate(context) {
            return Ok(None);
        }

        // Roll for probability
        let roll = {
            let mut rng = self.rng.lock().unwrap();
            rng.gen::<f64>()
        };

        if roll < self.probability {
            let error = self.formatter.format_error(
                self.error_type,
                context,
                None,
            )?;

            Ok(Some(error))
        } else {
            Ok(None)
        }
    }

    async fn reset(&mut self) {
        let mut rng = self.rng.lock().unwrap();
        *rng = ChaCha8Rng::seed_from_u64(0);
    }

    fn name(&self) -> &str {
        "conditional"
    }

    fn config(&self) -> StrategyConfig {
        StrategyConfig::Conditional {
            condition: self.condition.clone(),
            error_type: self.error_type,
            probability: self.probability,
        }
    }
}
```

### 3.6 Budget Exhaustion Strategy

```rust
// File: src/errors/injection/budget.rs

/// Simulates budget/quota exhaustion
#[derive(Debug)]
pub struct BudgetExhaustionStrategy {
    /// Token limit
    token_limit: u64,

    /// Reset period
    reset_period: Duration,

    /// Error type when budget exceeded
    error_type: ErrorType,

    /// Current token usage
    current_usage: Arc<AtomicU64>,

    /// Last reset time
    last_reset: Arc<RwLock<Instant>>,

    /// Error formatter
    formatter: Arc<dyn ErrorFormatter>,
}

impl BudgetExhaustionStrategy {
    pub fn new(
        token_limit: u64,
        reset_period: Duration,
        error_type: ErrorType,
        formatter: Arc<dyn ErrorFormatter>,
    ) -> Self {
        Self {
            token_limit,
            reset_period,
            error_type,
            current_usage: Arc::new(AtomicU64::new(0)),
            last_reset: Arc::new(RwLock::new(Instant::now())),
            formatter,
        }
    }

    fn check_and_reset(&self, current_time: Instant) {
        let last_reset = *self.last_reset.read().unwrap();

        if current_time.duration_since(last_reset) >= self.reset_period {
            // Reset budget
            self.current_usage.store(0, Ordering::Relaxed);

            let mut last_reset_write = self.last_reset.write().unwrap();
            *last_reset_write = current_time;
        }
    }
}

#[async_trait]
impl ErrorInjectionStrategy for BudgetExhaustionStrategy {
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError> {
        // Check if we need to reset
        self.check_and_reset(context.timestamp);

        // Calculate request token usage
        let request_tokens = context.prompt_tokens.unwrap_or(0)
            + context.max_tokens.unwrap_or(0);

        // Check current usage
        let current = self.current_usage.load(Ordering::Relaxed);

        if current + request_tokens > self.token_limit {
            // Budget exceeded - inject error
            let mut error = self.formatter.format_error(
                self.error_type,
                context,
                Some(format!(
                    "Token quota exceeded. Used: {}, Limit: {}, Request: {}",
                    current, self.token_limit, request_tokens
                )),
            )?;

            // Add quota information to headers
            error.rate_limit_info = Some(RateLimitInfo {
                remaining_requests: None,
                remaining_tokens: Some(0),
                limit_requests: None,
                limit_tokens: Some(self.token_limit),
                reset_timestamp: (*self.last_reset.read().unwrap() + self.reset_period)
                    .duration_since(Instant::now())
                    .as_secs(),
                reset_seconds: self.reset_period.as_secs() as u32,
            });

            Ok(Some(error))
        } else {
            // Budget OK - increment usage
            self.current_usage.fetch_add(request_tokens, Ordering::Relaxed);
            Ok(None)
        }
    }

    async fn reset(&mut self) {
        self.current_usage.store(0, Ordering::Relaxed);
        let mut last_reset = self.last_reset.write().unwrap();
        *last_reset = Instant::now();
    }

    fn name(&self) -> &str {
        "budget_exhaustion"
    }

    fn config(&self) -> StrategyConfig {
        StrategyConfig::BudgetExhaustion {
            token_limit: self.token_limit,
            reset_period: self.reset_period,
            error_type: self.error_type,
        }
    }
}
```

### 3.7 Load-Dependent Strategy

```rust
// File: src/errors/injection/load_dependent.rs

/// Inject errors based on system load
#[derive(Debug)]
pub struct LoadDependentStrategy {
    /// Concurrent request threshold
    concurrent_threshold: usize,

    /// Error type to inject
    error_type: ErrorType,

    /// Probability multiplier based on load
    load_multiplier: f64,

    /// RNG
    rng: Mutex<ChaCha8Rng>,

    /// Error formatter
    formatter: Arc<dyn ErrorFormatter>,
}

impl LoadDependentStrategy {
    pub fn new(
        concurrent_threshold: usize,
        error_type: ErrorType,
        load_multiplier: f64,
        seed: Option<u64>,
        formatter: Arc<dyn ErrorFormatter>,
    ) -> Self {
        let rng = if let Some(seed_val) = seed {
            ChaCha8Rng::seed_from_u64(seed_val)
        } else {
            ChaCha8Rng::from_entropy()
        };

        Self {
            concurrent_threshold,
            error_type,
            load_multiplier,
            rng: Mutex::new(rng),
            formatter,
        }
    }

    fn calculate_probability(&self, concurrent: usize) -> f64 {
        if concurrent <= self.concurrent_threshold {
            return 0.0;
        }

        // Linear increase in probability above threshold
        let excess = (concurrent - self.concurrent_threshold) as f64;
        let base_probability = 0.1; // 10% base when at threshold

        (base_probability + (excess * self.load_multiplier)).min(1.0)
    }
}

#[async_trait]
impl ErrorInjectionStrategy for LoadDependentStrategy {
    async fn should_inject(
        &self,
        context: &RequestContext,
    ) -> Result<Option<InjectedError>, StrategyError> {
        let probability = self.calculate_probability(context.concurrent_requests);

        if probability == 0.0 {
            return Ok(None);
        }

        let roll = {
            let mut rng = self.rng.lock().unwrap();
            rng.gen::<f64>()
        };

        if roll < probability {
            let mut error = self.formatter.format_error(
                self.error_type,
                context,
                Some(format!(
                    "Service degraded due to high load. Concurrent requests: {}",
                    context.concurrent_requests
                )),
            )?;

            // Add Retry-After header based on load
            let retry_delay = ((context.concurrent_requests - self.concurrent_threshold) * 2)
                .min(300) as u32; // Max 5 minutes
            error.retry_after = Some(retry_delay);

            Ok(Some(error))
        } else {
            Ok(None)
        }
    }

    async fn reset(&mut self) {
        let mut rng = self.rng.lock().unwrap();
        *rng = ChaCha8Rng::seed_from_u64(0);
    }

    fn name(&self) -> &str {
        "load_dependent"
    }

    fn config(&self) -> StrategyConfig {
        StrategyConfig::LoadDependent {
            concurrent_threshold: self.concurrent_threshold,
            error_type: self.error_type,
            load_multiplier: self.load_multiplier,
        }
    }
}
```

---

## 4. Provider-Specific Error Formatters

### 4.1 Error Formatter Trait

```rust
// File: src/errors/responses/formatter.rs

/// Trait for provider-specific error formatting
pub trait ErrorFormatter: Send + Sync + Debug {
    /// Format an error in provider-specific format
    fn format_error(
        &self,
        error_type: ErrorType,
        context: &RequestContext,
        custom_message: Option<String>,
    ) -> Result<InjectedError, FormatterError>;

    /// Get provider this formatter is for
    fn provider(&self) -> Provider;

    /// Generate retry headers for this provider
    fn retry_headers(
        &self,
        error_type: ErrorType,
        retry_after: Option<u32>,
    ) -> HashMap<String, String>;

    /// Generate rate limit headers
    fn rate_limit_headers(
        &self,
        rate_limit_info: &RateLimitInfo,
    ) -> HashMap<String, String>;
}

#[derive(Debug, Error)]
pub enum FormatterError {
    #[error("Unsupported error type: {0:?}")]
    UnsupportedError(ErrorType),

    #[error("Formatting error: {0}")]
    FormatError(String),
}

/// Provider error format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderErrorFormat {
    OpenAI(OpenAIError),
    Anthropic(AnthropicError),
    Google(GoogleError),
    Azure(AzureError),
    Cohere(CohereError),
}
```

### 4.2 OpenAI Error Formatter

```rust
// File: src/errors/responses/openai.rs

/// OpenAI error formatter
#[derive(Debug)]
pub struct OpenAIFormatter;

impl ErrorFormatter for OpenAIFormatter {
    fn format_error(
        &self,
        error_type: ErrorType,
        context: &RequestContext,
        custom_message: Option<String>,
    ) -> Result<InjectedError, FormatterError> {
        let (error_code, default_message) = self.error_details(error_type);
        let message = custom_message.unwrap_or_else(|| default_message.to_string());

        let status_code = error_type.http_status_code();
        let retry_after = error_type.suggested_retry_delay();

        // Build headers
        let mut headers = HashMap::new();

        // Add retry headers if applicable
        if let Some(delay) = retry_after {
            headers.extend(self.retry_headers(error_type, Some(delay)));
        }

        // Add rate limit headers for 429 errors
        if status_code == 429 {
            let rate_limit_info = self.generate_rate_limit_info(context, retry_after);
            headers.extend(self.rate_limit_headers(&rate_limit_info));
        }

        // Standard OpenAI headers
        headers.insert("content-type".to_string(), "application/json".to_string());
        headers.insert("x-request-id".to_string(), context.request_id.clone());

        // OpenAI-specific organization header
        headers.insert(
            "openai-organization".to_string(),
            "org-simulator".to_string(),
        );

        let rate_limit_info = if status_code == 429 {
            Some(self.generate_rate_limit_info(context, retry_after))
        } else {
            None
        };

        Ok(InjectedError {
            error_type,
            status_code,
            message: message.clone(),
            error_code: Some(error_code.to_string()),
            retry_after,
            headers,
            rate_limit_info,
            request_id: context.request_id.clone(),
            timestamp: context.timestamp,
            provider: Provider::OpenAI,
            injected_by: "error_injector".to_string(),
            retryable: error_type.is_retryable(),
            context: HashMap::new(),
        })
    }

    fn provider(&self) -> Provider {
        Provider::OpenAI
    }

    fn retry_headers(
        &self,
        error_type: ErrorType,
        retry_after: Option<u32>,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(seconds) = retry_after {
            headers.insert("retry-after".to_string(), seconds.to_string());
            headers.insert(
                "retry-after-ms".to_string(),
                (seconds * 1000).to_string(),
            );
        }

        headers
    }

    fn rate_limit_headers(
        &self,
        info: &RateLimitInfo,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // OpenAI rate limit headers
        if let Some(limit) = info.limit_requests {
            headers.insert(
                "x-ratelimit-limit-requests".to_string(),
                limit.to_string(),
            );
        }

        if let Some(remaining) = info.remaining_requests {
            headers.insert(
                "x-ratelimit-remaining-requests".to_string(),
                remaining.to_string(),
            );
        }

        if let Some(limit) = info.limit_tokens {
            headers.insert(
                "x-ratelimit-limit-tokens".to_string(),
                limit.to_string(),
            );
        }

        if let Some(remaining) = info.remaining_tokens {
            headers.insert(
                "x-ratelimit-remaining-tokens".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "x-ratelimit-reset-requests".to_string(),
            format!("{}s", info.reset_seconds),
        );

        headers.insert(
            "x-ratelimit-reset-tokens".to_string(),
            format!("{}s", info.reset_seconds),
        );

        headers
    }
}

impl OpenAIFormatter {
    fn error_details(&self, error_type: ErrorType) -> (&'static str, &'static str) {
        match error_type {
            ErrorType::RateLimit | ErrorType::RateLimitRPM => (
                "rate_limit_exceeded",
                "Rate limit exceeded. Please retry after the specified time.",
            ),
            ErrorType::RateLimitTPM => (
                "rate_limit_exceeded",
                "Token rate limit exceeded. Please reduce request frequency.",
            ),
            ErrorType::RateLimitConcurrent => (
                "rate_limit_exceeded",
                "Too many concurrent requests. Please retry later.",
            ),
            ErrorType::AuthenticationFailed | ErrorType::InvalidApiKey => (
                "invalid_api_key",
                "Incorrect API key provided. Please check your API key.",
            ),
            ErrorType::PermissionDenied => (
                "permission_denied",
                "You do not have permission to access this resource.",
            ),
            ErrorType::InsufficientQuota => (
                "insufficient_quota",
                "You have insufficient quota for this request.",
            ),
            ErrorType::ContextLengthExceeded => (
                "context_length_exceeded",
                "Maximum context length exceeded. Please reduce the length of the messages or completion.",
            ),
            ErrorType::TokenLimitExceeded => (
                "token_limit_exceeded",
                "This model's maximum token limit is exceeded by your request.",
            ),
            ErrorType::ContentFilterTriggered => (
                "content_filter",
                "Your request was rejected as a result of our safety system.",
            ),
            ErrorType::InvalidModel => (
                "invalid_model",
                "The model specified does not exist or you do not have access to it.",
            ),
            ErrorType::InvalidParameter => (
                "invalid_request_error",
                "Invalid parameter provided in request.",
            ),
            ErrorType::MissingParameter => (
                "invalid_request_error",
                "Required parameter is missing from request.",
            ),
            ErrorType::BadRequest | ErrorType::InvalidFormat => (
                "invalid_request_error",
                "Invalid request format or parameters.",
            ),
            ErrorType::InternalServerError => (
                "internal_error",
                "The server had an error processing your request. Please retry.",
            ),
            ErrorType::ServiceUnavailable => (
                "service_unavailable",
                "The service is temporarily unavailable. Please retry after a short wait.",
            ),
            ErrorType::BadGateway => (
                "bad_gateway",
                "Bad gateway error. Please retry your request.",
            ),
            ErrorType::GatewayTimeout | ErrorType::UpstreamTimeout => (
                "timeout",
                "Request timed out. Please retry.",
            ),
            ErrorType::RequestTimeout => (
                "timeout",
                "Your request timed out. Please retry with a shorter request.",
            ),
            ErrorType::QuotaExceeded | ErrorType::DailyQuotaExceeded => (
                "quota_exceeded",
                "You have exceeded your quota. Please check your plan and billing details.",
            ),
            ErrorType::MonthlyQuotaExceeded => (
                "quota_exceeded",
                "Monthly quota exceeded. Your quota will reset at the start of next month.",
            ),
            _ => (
                "unknown_error",
                "An unknown error occurred.",
            ),
        }
    }

    fn generate_rate_limit_info(
        &self,
        context: &RequestContext,
        retry_after: Option<u32>,
    ) -> RateLimitInfo {
        // Simulate realistic rate limit values
        RateLimitInfo {
            remaining_requests: Some(0),
            remaining_tokens: Some(0),
            limit_requests: Some(10000),
            limit_tokens: Some(1000000),
            reset_timestamp: (Instant::now() + Duration::from_secs(retry_after.unwrap_or(60) as u64))
                .duration_since(Instant::now())
                .as_secs(),
            reset_seconds: retry_after.unwrap_or(60),
        }
    }
}

/// OpenAI error response body format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIError {
    pub error: OpenAIErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIErrorDetail {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl OpenAIFormatter {
    /// Generate JSON body for OpenAI error
    pub fn error_body(&self, error: &InjectedError) -> String {
        let error_detail = OpenAIError {
            error: OpenAIErrorDetail {
                message: error.message.clone(),
                r#type: error.error_code.clone().unwrap_or_else(|| "error".to_string()),
                param: None,
                code: error.error_code.clone(),
            },
        };

        serde_json::to_string(&error_detail).unwrap_or_else(|_| {
            r#"{"error": {"message": "Unknown error", "type": "error"}}"#.to_string()
        })
    }
}
```

### 4.3 Anthropic Error Formatter

```rust
// File: src/errors/responses/anthropic.rs

/// Anthropic (Claude) error formatter
#[derive(Debug)]
pub struct AnthropicFormatter;

impl ErrorFormatter for AnthropicFormatter {
    fn format_error(
        &self,
        error_type: ErrorType,
        context: &RequestContext,
        custom_message: Option<String>,
    ) -> Result<InjectedError, FormatterError> {
        let (error_code, default_message) = self.error_details(error_type);
        let message = custom_message.unwrap_or_else(|| default_message.to_string());

        let status_code = error_type.http_status_code();
        let retry_after = error_type.suggested_retry_delay();

        let mut headers = HashMap::new();

        // Anthropic-specific headers
        headers.insert("content-type".to_string(), "application/json".to_string());
        headers.insert("request-id".to_string(), context.request_id.clone());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());

        // Retry headers
        if let Some(delay) = retry_after {
            headers.insert("retry-after".to_string(), delay.to_string());
        }

        // Rate limit headers for 429
        let rate_limit_info = if status_code == 429 {
            let info = self.generate_rate_limit_info(context, retry_after);
            headers.extend(self.rate_limit_headers(&info));
            Some(info)
        } else {
            None
        };

        Ok(InjectedError {
            error_type,
            status_code,
            message: message.clone(),
            error_code: Some(error_code.to_string()),
            retry_after,
            headers,
            rate_limit_info,
            request_id: context.request_id.clone(),
            timestamp: context.timestamp,
            provider: Provider::Anthropic,
            injected_by: "error_injector".to_string(),
            retryable: error_type.is_retryable(),
            context: HashMap::new(),
        })
    }

    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    fn retry_headers(
        &self,
        error_type: ErrorType,
        retry_after: Option<u32>,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(seconds) = retry_after {
            headers.insert("retry-after".to_string(), seconds.to_string());
        }

        headers
    }

    fn rate_limit_headers(
        &self,
        info: &RateLimitInfo,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Anthropic uses different header format
        if let Some(limit) = info.limit_requests {
            headers.insert(
                "anthropic-ratelimit-requests-limit".to_string(),
                limit.to_string(),
            );
        }

        if let Some(remaining) = info.remaining_requests {
            headers.insert(
                "anthropic-ratelimit-requests-remaining".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "anthropic-ratelimit-requests-reset".to_string(),
            info.reset_timestamp.to_string(),
        );

        if let Some(limit) = info.limit_tokens {
            headers.insert(
                "anthropic-ratelimit-tokens-limit".to_string(),
                limit.to_string(),
            );
        }

        if let Some(remaining) = info.remaining_tokens {
            headers.insert(
                "anthropic-ratelimit-tokens-remaining".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "anthropic-ratelimit-tokens-reset".to_string(),
            info.reset_timestamp.to_string(),
        );

        headers
    }
}

impl AnthropicFormatter {
    fn error_details(&self, error_type: ErrorType) -> (&'static str, &'static str) {
        match error_type {
            ErrorType::RateLimit | ErrorType::RateLimitRPM | ErrorType::RateLimitTPM => (
                "rate_limit_error",
                "Your request has been rate limited. Please wait before retrying.",
            ),
            ErrorType::AuthenticationFailed | ErrorType::InvalidApiKey => (
                "authentication_error",
                "Invalid authentication credentials provided.",
            ),
            ErrorType::PermissionDenied => (
                "permission_error",
                "You do not have permission to perform this action.",
            ),
            ErrorType::ContextLengthExceeded => (
                "invalid_request_error",
                "The prompt exceeds the model's maximum context length.",
            ),
            ErrorType::InvalidModel => (
                "not_found_error",
                "The requested model does not exist.",
            ),
            ErrorType::InvalidParameter | ErrorType::MissingParameter => (
                "invalid_request_error",
                "Invalid or missing parameter in request.",
            ),
            ErrorType::InternalServerError => (
                "api_error",
                "An internal server error occurred. Please retry.",
            ),
            ErrorType::ServiceUnavailable => (
                "overloaded_error",
                "The API is currently overloaded. Please retry shortly.",
            ),
            ErrorType::QuotaExceeded | ErrorType::DailyQuotaExceeded => (
                "rate_limit_error",
                "Usage quota exceeded. Please check your usage limits.",
            ),
            _ => (
                "api_error",
                "An error occurred processing your request.",
            ),
        }
    }

    fn generate_rate_limit_info(
        &self,
        context: &RequestContext,
        retry_after: Option<u32>,
    ) -> RateLimitInfo {
        RateLimitInfo {
            remaining_requests: Some(0),
            remaining_tokens: Some(0),
            limit_requests: Some(1000),
            limit_tokens: Some(100000),
            reset_timestamp: (Instant::now() + Duration::from_secs(retry_after.unwrap_or(60) as u64))
                .duration_since(Instant::now())
                .as_secs(),
            reset_seconds: retry_after.unwrap_or(60),
        }
    }
}

/// Anthropic error response body format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicError {
    pub r#type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorDetail {
    pub r#type: String,
    pub message: String,
}

impl AnthropicFormatter {
    pub fn error_body(&self, error: &InjectedError) -> String {
        let error_body = AnthropicError {
            r#type: "error".to_string(),
            error: AnthropicErrorDetail {
                r#type: error.error_code.clone().unwrap_or_else(|| "api_error".to_string()),
                message: error.message.clone(),
            },
        };

        serde_json::to_string(&error_body).unwrap_or_else(|_| {
            r#"{"type": "error", "error": {"type": "api_error", "message": "Unknown error"}}"#.to_string()
        })
    }
}
```

### 4.4 Google (Gemini) Error Formatter

```rust
// File: src/errors/responses/google.rs

/// Google Gemini error formatter
#[derive(Debug)]
pub struct GoogleFormatter;

impl ErrorFormatter for GoogleFormatter {
    fn format_error(
        &self,
        error_type: ErrorType,
        context: &RequestContext,
        custom_message: Option<String>,
    ) -> Result<InjectedError, FormatterError> {
        let (error_code, default_message, grpc_status) = self.error_details(error_type);
        let message = custom_message.unwrap_or_else(|| default_message.to_string());

        let status_code = error_type.http_status_code();
        let retry_after = error_type.suggested_retry_delay();

        let mut headers = HashMap::new();

        // Google-specific headers
        headers.insert("content-type".to_string(), "application/json".to_string());
        headers.insert("x-request-id".to_string(), context.request_id.clone());
        headers.insert("x-goog-api-version".to_string(), "v1".to_string());

        // gRPC status mapping
        headers.insert("grpc-status".to_string(), grpc_status.to_string());

        if let Some(delay) = retry_after {
            headers.insert("retry-after".to_string(), delay.to_string());
        }

        let rate_limit_info = if status_code == 429 {
            let info = self.generate_rate_limit_info(context, retry_after);
            Some(info)
        } else {
            None
        };

        Ok(InjectedError {
            error_type,
            status_code,
            message: message.clone(),
            error_code: Some(error_code.to_string()),
            retry_after,
            headers,
            rate_limit_info,
            request_id: context.request_id.clone(),
            timestamp: context.timestamp,
            provider: Provider::Google,
            injected_by: "error_injector".to_string(),
            retryable: error_type.is_retryable(),
            context: HashMap::new(),
        })
    }

    fn provider(&self) -> Provider {
        Provider::Google
    }

    fn retry_headers(
        &self,
        _error_type: ErrorType,
        retry_after: Option<u32>,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(seconds) = retry_after {
            headers.insert("retry-after".to_string(), seconds.to_string());
        }

        headers
    }

    fn rate_limit_headers(
        &self,
        _info: &RateLimitInfo,
    ) -> HashMap<String, String> {
        // Google doesn't expose detailed rate limit headers
        HashMap::new()
    }
}

impl GoogleFormatter {
    fn error_details(&self, error_type: ErrorType) -> (&'static str, &'static str, u32) {
        match error_type {
            ErrorType::RateLimit | ErrorType::RateLimitRPM => (
                "RESOURCE_EXHAUSTED",
                "Quota exceeded for quota metric 'Generate Content API requests per minute'.",
                8, // gRPC RESOURCE_EXHAUSTED
            ),
            ErrorType::AuthenticationFailed | ErrorType::InvalidApiKey => (
                "UNAUTHENTICATED",
                "API key not valid. Please pass a valid API key.",
                16, // gRPC UNAUTHENTICATED
            ),
            ErrorType::PermissionDenied => (
                "PERMISSION_DENIED",
                "The caller does not have permission to execute the specified operation.",
                7, // gRPC PERMISSION_DENIED
            ),
            ErrorType::InvalidParameter | ErrorType::MissingParameter => (
                "INVALID_ARGUMENT",
                "Request contains an invalid argument.",
                3, // gRPC INVALID_ARGUMENT
            ),
            ErrorType::ContextLengthExceeded | ErrorType::TokenLimitExceeded => (
                "INVALID_ARGUMENT",
                "The input is too long. Maximum token limit exceeded.",
                3, // gRPC INVALID_ARGUMENT
            ),
            ErrorType::InternalServerError => (
                "INTERNAL",
                "An internal error occurred.",
                13, // gRPC INTERNAL
            ),
            ErrorType::ServiceUnavailable => (
                "UNAVAILABLE",
                "The service is currently unavailable.",
                14, // gRPC UNAVAILABLE
            ),
            ErrorType::GatewayTimeout | ErrorType::RequestTimeout => (
                "DEADLINE_EXCEEDED",
                "The request deadline was exceeded.",
                4, // gRPC DEADLINE_EXCEEDED
            ),
            _ => (
                "UNKNOWN",
                "An unknown error occurred.",
                2, // gRPC UNKNOWN
            ),
        }
    }

    fn generate_rate_limit_info(
        &self,
        _context: &RequestContext,
        retry_after: Option<u32>,
    ) -> RateLimitInfo {
        RateLimitInfo {
            remaining_requests: None,
            remaining_tokens: None,
            limit_requests: Some(360),
            limit_tokens: Some(120000),
            reset_timestamp: (Instant::now() + Duration::from_secs(retry_after.unwrap_or(60) as u64))
                .duration_since(Instant::now())
                .as_secs(),
            reset_seconds: retry_after.unwrap_or(60),
        }
    }
}

/// Google error response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleError {
    pub error: GoogleErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleErrorDetail {
    pub code: u32,
    pub message: String,
    pub status: String,
    pub details: Option<Vec<serde_json::Value>>,
}

impl GoogleFormatter {
    pub fn error_body(&self, error: &InjectedError) -> String {
        let error_body = GoogleError {
            error: GoogleErrorDetail {
                code: error.status_code as u32,
                message: error.message.clone(),
                status: error.error_code.clone().unwrap_or_else(|| "UNKNOWN".to_string()),
                details: None,
            },
        };

        serde_json::to_string(&error_body).unwrap_or_else(|_| {
            r#"{"error": {"code": 500, "message": "Unknown error", "status": "UNKNOWN"}}"#.to_string()
        })
    }
}
```

### 4.5 Azure OpenAI Error Formatter

```rust
// File: src/errors/responses/azure.rs

/// Azure OpenAI error formatter (extends OpenAI format)
#[derive(Debug)]
pub struct AzureFormatter {
    openai_formatter: OpenAIFormatter,
}

impl AzureFormatter {
    pub fn new() -> Self {
        Self {
            openai_formatter: OpenAIFormatter,
        }
    }
}

impl ErrorFormatter for AzureFormatter {
    fn format_error(
        &self,
        error_type: ErrorType,
        context: &RequestContext,
        custom_message: Option<String>,
    ) -> Result<InjectedError, FormatterError> {
        // Use OpenAI formatting as base
        let mut error = self.openai_formatter.format_error(
            error_type,
            context,
            custom_message,
        )?;

        // Override provider
        error.provider = Provider::Azure;

        // Add Azure-specific headers
        error.headers.insert(
            "x-ms-region".to_string(),
            "eastus".to_string(),
        );
        error.headers.insert(
            "x-ms-client-request-id".to_string(),
            context.request_id.clone(),
        );
        error.headers.insert(
            "apim-request-id".to_string(),
            format!("azure-{}", context.request_id),
        );

        // Azure uses different rate limit header names
        if error.status_code == 429 {
            // Remove OpenAI headers
            error.headers.retain(|k, _| !k.starts_with("x-ratelimit-"));

            // Add Azure headers
            if let Some(info) = &error.rate_limit_info {
                error.headers.extend(self.rate_limit_headers(info));
            }
        }

        Ok(error)
    }

    fn provider(&self) -> Provider {
        Provider::Azure
    }

    fn retry_headers(
        &self,
        error_type: ErrorType,
        retry_after: Option<u32>,
    ) -> HashMap<String, String> {
        self.openai_formatter.retry_headers(error_type, retry_after)
    }

    fn rate_limit_headers(
        &self,
        info: &RateLimitInfo,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Azure-specific rate limit headers
        if let Some(remaining) = info.remaining_requests {
            headers.insert(
                "x-ms-ratelimit-remaining-requests".to_string(),
                remaining.to_string(),
            );
        }

        if let Some(remaining) = info.remaining_tokens {
            headers.insert(
                "x-ms-ratelimit-remaining-tokens".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "retry-after-ms".to_string(),
            (info.reset_seconds * 1000).to_string(),
        );

        headers
    }
}
```

### 4.6 Cohere Error Formatter

```rust
// File: src/errors/responses/cohere.rs

/// Cohere error formatter
#[derive(Debug)]
pub struct CohereFormatter;

impl ErrorFormatter for CohereFormatter {
    fn format_error(
        &self,
        error_type: ErrorType,
        context: &RequestContext,
        custom_message: Option<String>,
    ) -> Result<InjectedError, FormatterError> {
        let default_message = self.error_message(error_type);
        let message = custom_message.unwrap_or_else(|| default_message.to_string());

        let status_code = error_type.http_status_code();
        let retry_after = error_type.suggested_retry_delay();

        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        headers.insert("x-request-id".to_string(), context.request_id.clone());

        if let Some(delay) = retry_after {
            headers.insert("retry-after".to_string(), delay.to_string());
        }

        Ok(InjectedError {
            error_type,
            status_code,
            message: message.clone(),
            error_code: None, // Cohere doesn't use error codes
            retry_after,
            headers,
            rate_limit_info: None,
            request_id: context.request_id.clone(),
            timestamp: context.timestamp,
            provider: Provider::Cohere,
            injected_by: "error_injector".to_string(),
            retryable: error_type.is_retryable(),
            context: HashMap::new(),
        })
    }

    fn provider(&self) -> Provider {
        Provider::Cohere
    }

    fn retry_headers(
        &self,
        _error_type: ErrorType,
        retry_after: Option<u32>,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(seconds) = retry_after {
            headers.insert("retry-after".to_string(), seconds.to_string());
        }

        headers
    }

    fn rate_limit_headers(
        &self,
        _info: &RateLimitInfo,
    ) -> HashMap<String, String> {
        HashMap::new()
    }
}

impl CohereFormatter {
    fn error_message(&self, error_type: ErrorType) -> &'static str {
        match error_type {
            ErrorType::RateLimit | ErrorType::RateLimitRPM => {
                "You are being rate limited. Please retry after some time."
            }
            ErrorType::AuthenticationFailed | ErrorType::InvalidApiKey => {
                "Invalid API key. Please check your credentials."
            }
            ErrorType::InvalidParameter | ErrorType::MissingParameter => {
                "Invalid request parameters."
            }
            ErrorType::ContextLengthExceeded => {
                "The input text is too long."
            }
            ErrorType::InternalServerError => {
                "Internal server error. Please retry."
            }
            ErrorType::ServiceUnavailable => {
                "Service temporarily unavailable. Please retry shortly."
            }
            _ => "An error occurred processing your request.",
        }
    }
}

/// Cohere error response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereError {
    pub message: String,
}

impl CohereFormatter {
    pub fn error_body(&self, error: &InjectedError) -> String {
        let error_body = CohereError {
            message: error.message.clone(),
        };

        serde_json::to_string(&error_body).unwrap_or_else(|_| {
            r#"{"message": "Unknown error"}"#.to_string()
        })
    }
}
```

---

## 5. Circuit Breaker Simulation

```rust
// File: src/errors/circuit_breaker/mod.rs

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

/// Circuit breaker state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed - requests flow normally
    Closed,

    /// Circuit is open - all requests fail immediately
    Open,

    /// Circuit is half-open - testing if service recovered
    HalfOpen,
}

/// Circuit breaker for simulating cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state
    state: Arc<RwLock<CircuitState>>,

    /// Configuration
    config: CircuitBreakerConfig,

    /// Failure counter (within time window)
    failure_count: Arc<AtomicU64>,

    /// Success counter (in half-open state)
    success_count: Arc<AtomicU64>,

    /// Timestamp when circuit opened
    opened_at: Arc<RwLock<Option<Instant>>>,

    /// Timestamp of first failure in window
    window_start: Arc<RwLock<Option<Instant>>>,

    /// Error formatter
    formatter: Arc<dyn ErrorFormatter>,
}

impl CircuitBreaker {
    pub fn new(
        config: CircuitBreakerConfig,
        formatter: Arc<dyn ErrorFormatter>,
    ) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            opened_at: Arc::new(RwLock::new(None)),
            window_start: Arc::new(RwLock::new(None)),
            formatter,
        }
    }

    /// Check if request should be allowed through
    pub async fn allow_request(&self) -> bool {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                self.check_open_timeout().await;
                false
            }
            CircuitState::HalfOpen => true, // Allow limited requests
        }
    }

    /// Record a successful request
    pub async fn record_success(&self) {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => {
                // Reset failure counter
                self.failure_count.store(0, Ordering::Relaxed);
                let mut window = self.window_start.write().await;
                *window = None;
            }

            CircuitState::HalfOpen => {
                // Increment success counter
                let count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;

                // Check if we should close the circuit
                if count >= self.config.half_open_success_threshold as u64 {
                    self.transition_to_closed().await;
                }
            }

            CircuitState::Open => {
                // Shouldn't happen, but ignore
            }
        }
    }

    /// Record a failed request
    pub async fn record_failure(&self) {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => {
                // Initialize window if needed
                {
                    let mut window = self.window_start.write().await;
                    if window.is_none() {
                        *window = Some(Instant::now());
                    }
                }

                // Check if window expired
                let window_start = self.window_start.read().await;
                if let Some(start) = *window_start {
                    if Instant::now().duration_since(start) > self.config.window_duration {
                        // Reset window
                        drop(window_start);
                        let mut window = self.window_start.write().await;
                        *window = Some(Instant::now());
                        self.failure_count.store(0, Ordering::Relaxed);
                    }
                }

                // Increment failure counter
                let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

                // Check threshold
                if count >= self.config.failure_threshold as u64 {
                    self.transition_to_open().await;
                }
            }

            CircuitState::HalfOpen => {
                // Any failure in half-open state re-opens circuit
                self.transition_to_open().await;
            }

            CircuitState::Open => {
                // Already open
            }
        }
    }

    /// Get current circuit state
    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Generate error for open circuit
    pub async fn circuit_open_error(
        &self,
        context: &RequestContext,
    ) -> InjectedError {
        self.formatter.format_error(
            self.config.open_circuit_error,
            context,
            Some("Circuit breaker is open. Service is currently unavailable.".to_string()),
        ).unwrap_or_else(|_| {
            // Fallback error
            InjectedError {
                error_type: ErrorType::CircuitBreakerOpen,
                status_code: 503,
                message: "Circuit breaker is open".to_string(),
                error_code: Some("circuit_breaker_open".to_string()),
                retry_after: Some(self.config.open_duration.as_secs() as u32),
                headers: HashMap::new(),
                rate_limit_info: None,
                request_id: context.request_id.clone(),
                timestamp: context.timestamp,
                provider: context.provider,
                injected_by: "circuit_breaker".to_string(),
                retryable: true,
                context: HashMap::new(),
            }
        })
    }

    // Private state transition methods

    async fn transition_to_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Open;

        let mut opened = self.opened_at.write().await;
        *opened = Some(Instant::now());

        self.success_count.store(0, Ordering::Relaxed);

        tracing::warn!("Circuit breaker transitioned to OPEN state");
    }

    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::HalfOpen;

        self.success_count.store(0, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);

        tracing::info!("Circuit breaker transitioned to HALF-OPEN state");
    }

    async fn transition_to_closed(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Closed;

        let mut opened = self.opened_at.write().await;
        *opened = None;

        let mut window = self.window_start.write().await;
        *window = None;

        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);

        tracing::info!("Circuit breaker transitioned to CLOSED state");
    }

    async fn check_open_timeout(&self) {
        let opened = self.opened_at.read().await;

        if let Some(opened_time) = *opened {
            if Instant::now().duration_since(opened_time) >= self.config.open_duration {
                drop(opened);
                self.transition_to_half_open().await;
            }
        }
    }

    /// Reset circuit breaker to initial state
    pub async fn reset(&self) {
        self.transition_to_closed().await;
    }
}

/// Circuit breaker metrics for observability
#[derive(Debug, Clone, Serialize)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub failure_count: u64,
    pub success_count: u64,
    pub opened_at: Option<Instant>,
    pub time_in_open_state: Option<Duration>,
}

impl CircuitBreaker {
    pub async fn metrics(&self) -> CircuitBreakerMetrics {
        let state = *self.state.read().await;
        let opened_at = *self.opened_at.read().await;

        CircuitBreakerMetrics {
            state,
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            opened_at,
            time_in_open_state: opened_at.map(|t| Instant::now().duration_since(t)),
        }
    }
}
```

---

## 6. Chaos Engineering Scenarios

```rust
// File: src/errors/scenarios/mod.rs

/// Pre-configured chaos engineering scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosScenario {
    /// Scenario name
    pub name: String,

    /// Description
    pub description: String,

    /// Strategies to apply
    pub strategies: Vec<StrategyConfig>,

    /// Circuit breaker config (optional)
    pub circuit_breaker: Option<CircuitBreakerConfig>,

    /// Duration of scenario
    pub duration: Option<Duration>,
}

impl ChaosScenario {
    /// Intermittent 429 rate limit errors
    pub fn intermittent_rate_limits() -> Self {
        Self {
            name: "intermittent_rate_limits".to_string(),
            description: "Random 429 errors with 5% probability".to_string(),
            strategies: vec![
                StrategyConfig::Probabilistic {
                    error_type: ErrorType::RateLimit,
                    probability: 0.05,
                    time_window: None,
                    continue_on_inject: true,
                },
            ],
            circuit_breaker: None,
            duration: None,
        }
    }

    /// Cascading failures with circuit breaker
    pub fn cascading_failures() -> Self {
        Self {
            name: "cascading_failures".to_string(),
            description: "Progressive failures triggering circuit breaker".to_string(),
            strategies: vec![
                StrategyConfig::Probabilistic {
                    error_type: ErrorType::ServiceUnavailable,
                    probability: 0.15,
                    time_window: None,
                    continue_on_inject: true,
                },
            ],
            circuit_breaker: Some(CircuitBreakerConfig {
                failure_threshold: 5,
                window_duration: Duration::from_secs(10),
                open_duration: Duration::from_secs(30),
                half_open_success_threshold: 3,
                open_circuit_error: ErrorType::CircuitBreakerOpen,
            }),
            duration: Some(Duration::from_secs(120)),
        }
    }

    /// Gradual degradation
    pub fn gradual_degradation() -> Self {
        Self {
            name: "gradual_degradation".to_string(),
            description: "Increasing error rates over time".to_string(),
            strategies: vec![
                StrategyConfig::TimeBased {
                    trigger: TimeTrigger::AtTime {
                        offset: Duration::from_secs(30),
                    },
                    error_type: ErrorType::SlowResponse,
                    duration: Duration::from_secs(60),
                },
                StrategyConfig::TimeBased {
                    trigger: TimeTrigger::AtTime {
                        offset: Duration::from_secs(90),
                    },
                    error_type: ErrorType::ServiceUnavailable,
                    duration: Duration::from_secs(30),
                },
            ],
            circuit_breaker: None,
            duration: Some(Duration::from_secs(120)),
        }
    }

    /// Token quota exhaustion
    pub fn quota_exhaustion() -> Self {
        Self {
            name: "quota_exhaustion".to_string(),
            description: "Simulate hitting token quota limits".to_string(),
            strategies: vec![
                StrategyConfig::BudgetExhaustion {
                    token_limit: 100000,
                    reset_period: Duration::from_secs(3600),
                    error_type: ErrorType::QuotaExceeded,
                },
            ],
            circuit_breaker: None,
            duration: None,
        }
    }

    /// High load simulation
    pub fn high_load() -> Self {
        Self {
            name: "high_load".to_string(),
            description: "Errors increase under high concurrent load".to_string(),
            strategies: vec![
                StrategyConfig::LoadDependent {
                    concurrent_threshold: 50,
                    error_type: ErrorType::ServiceUnavailable,
                    load_multiplier: 0.02,
                },
            ],
            circuit_breaker: None,
            duration: None,
        }
    }

    /// Context length failures
    pub fn context_length_errors() -> Self {
        Self {
            name: "context_length_errors".to_string(),
            description: "Fail requests with large token counts".to_string(),
            strategies: vec![
                StrategyConfig::Conditional {
                    condition: RequestCondition::TokenCount {
                        min_tokens: Some(100000),
                        max_tokens: None,
                    },
                    error_type: ErrorType::ContextLengthExceeded,
                    probability: 1.0,
                },
            ],
            circuit_breaker: None,
            duration: None,
        }
    }

    /// Periodic outages
    pub fn periodic_outages() -> Self {
        Self {
            name: "periodic_outages".to_string(),
            description: "Service outages every 10th request".to_string(),
            strategies: vec![
                StrategyConfig::Sequence {
                    pattern: SequencePattern::EveryNth { n: 10, offset: 0 },
                    error_type: ErrorType::ServiceUnavailable,
                    max_injections: Some(5),
                },
            ],
            circuit_breaker: None,
            duration: None,
        }
    }

    /// Multi-error chaos
    pub fn multi_error_chaos() -> Self {
        Self {
            name: "multi_error_chaos".to_string(),
            description: "Random mix of different error types".to_string(),
            strategies: vec![
                StrategyConfig::Probabilistic {
                    error_type: ErrorType::RateLimit,
                    probability: 0.03,
                    time_window: None,
                    continue_on_inject: false,
                },
                StrategyConfig::Probabilistic {
                    error_type: ErrorType::InternalServerError,
                    probability: 0.02,
                    time_window: None,
                    continue_on_inject: false,
                },
                StrategyConfig::Probabilistic {
                    error_type: ErrorType::GatewayTimeout,
                    probability: 0.01,
                    time_window: None,
                    continue_on_inject: false,
                },
            ],
            circuit_breaker: None,
            duration: None,
        }
    }

    /// Get all built-in scenarios
    pub fn all_scenarios() -> Vec<Self> {
        vec![
            Self::intermittent_rate_limits(),
            Self::cascading_failures(),
            Self::gradual_degradation(),
            Self::quota_exhaustion(),
            Self::high_load(),
            Self::context_length_errors(),
            Self::periodic_outages(),
            Self::multi_error_chaos(),
        ]
    }
}
```

---

## 7. Retry Header Generation

```rust
// File: src/errors/headers/mod.rs

use chrono::{DateTime, Utc, Duration as ChronoDuration};

/// Generate retry and rate limit headers
pub struct HeaderGenerator;

impl HeaderGenerator {
    /// Generate Retry-After header
    pub fn retry_after_header(seconds: u32) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Retry-After as seconds
        headers.insert("retry-after".to_string(), seconds.to_string());

        // Also provide HTTP-date format (RFC 7231)
        let retry_time = Utc::now() + ChronoDuration::seconds(seconds as i64);
        headers.insert(
            "retry-after-http-date".to_string(),
            retry_time.format("%a, %d %b %Y %H:%M:%S GMT").to_string(),
        );

        headers
    }

    /// Generate X-RateLimit-* headers (standard format)
    pub fn rate_limit_headers(
        limit: u32,
        remaining: u32,
        reset_timestamp: u64,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        headers.insert("x-ratelimit-limit".to_string(), limit.to_string());
        headers.insert("x-ratelimit-remaining".to_string(), remaining.to_string());
        headers.insert("x-ratelimit-reset".to_string(), reset_timestamp.to_string());

        // Calculate seconds until reset
        let now = Utc::now().timestamp() as u64;
        let reset_seconds = reset_timestamp.saturating_sub(now);
        headers.insert("x-ratelimit-reset-after".to_string(), reset_seconds.to_string());

        headers
    }

    /// Generate RateLimit header (RFC draft)
    pub fn ratelimit_rfc_header(
        limit: u32,
        remaining: u32,
        reset_seconds: u32,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // RateLimit header format (RFC draft)
        let ratelimit_value = format!(
            "limit={}, remaining={}, reset={}",
            limit, remaining, reset_seconds
        );

        headers.insert("ratelimit".to_string(), ratelimit_value);
        headers.insert("ratelimit-limit".to_string(), limit.to_string());
        headers.insert("ratelimit-remaining".to_string(), remaining.to_string());
        headers.insert("ratelimit-reset".to_string(), reset_seconds.to_string());

        headers
    }

    /// Generate provider-specific rate limit headers
    pub fn provider_rate_limit_headers(
        provider: Provider,
        info: &RateLimitInfo,
    ) -> HashMap<String, String> {
        match provider {
            Provider::OpenAI => Self::openai_rate_limit_headers(info),
            Provider::Anthropic => Self::anthropic_rate_limit_headers(info),
            Provider::Azure => Self::azure_rate_limit_headers(info),
            Provider::Google => HashMap::new(), // Google doesn't expose detailed headers
            Provider::Cohere => HashMap::new(),  // Cohere doesn't expose detailed headers
            Provider::Custom(_) => Self::rate_limit_headers(
                info.limit_requests.unwrap_or(1000),
                info.remaining_requests.unwrap_or(0),
                info.reset_timestamp,
            ),
        }
    }

    fn openai_rate_limit_headers(info: &RateLimitInfo) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(limit) = info.limit_requests {
            headers.insert("x-ratelimit-limit-requests".to_string(), limit.to_string());
        }

        if let Some(remaining) = info.remaining_requests {
            headers.insert("x-ratelimit-remaining-requests".to_string(), remaining.to_string());
        }

        if let Some(limit) = info.limit_tokens {
            headers.insert("x-ratelimit-limit-tokens".to_string(), limit.to_string());
        }

        if let Some(remaining) = info.remaining_tokens {
            headers.insert("x-ratelimit-remaining-tokens".to_string(), remaining.to_string());
        }

        headers.insert(
            "x-ratelimit-reset-requests".to_string(),
            format!("{}s", info.reset_seconds),
        );

        headers.insert(
            "x-ratelimit-reset-tokens".to_string(),
            format!("{}s", info.reset_seconds),
        );

        headers
    }

    fn anthropic_rate_limit_headers(info: &RateLimitInfo) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(limit) = info.limit_requests {
            headers.insert("anthropic-ratelimit-requests-limit".to_string(), limit.to_string());
        }

        if let Some(remaining) = info.remaining_requests {
            headers.insert(
                "anthropic-ratelimit-requests-remaining".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "anthropic-ratelimit-requests-reset".to_string(),
            info.reset_timestamp.to_string(),
        );

        if let Some(limit) = info.limit_tokens {
            headers.insert("anthropic-ratelimit-tokens-limit".to_string(), limit.to_string());
        }

        if let Some(remaining) = info.remaining_tokens {
            headers.insert(
                "anthropic-ratelimit-tokens-remaining".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "anthropic-ratelimit-tokens-reset".to_string(),
            info.reset_timestamp.to_string(),
        );

        headers
    }

    fn azure_rate_limit_headers(info: &RateLimitInfo) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(remaining) = info.remaining_requests {
            headers.insert(
                "x-ms-ratelimit-remaining-requests".to_string(),
                remaining.to_string(),
            );
        }

        if let Some(remaining) = info.remaining_tokens {
            headers.insert(
                "x-ms-ratelimit-remaining-tokens".to_string(),
                remaining.to_string(),
            );
        }

        headers.insert(
            "retry-after-ms".to_string(),
            (info.reset_seconds * 1000).to_string(),
        );

        headers
    }
}
```

---

## 8. Integration Points

### 8.1 Error Injection Orchestrator

```rust
// File: src/errors/injection/orchestrator.rs

/// Main orchestrator for error injection
pub struct ErrorInjectionOrchestrator {
    /// Configuration
    config: ErrorInjectionConfig,

    /// Active strategies
    strategies: Vec<Box<dyn ErrorInjectionStrategy>>,

    /// Circuit breaker (optional)
    circuit_breaker: Option<Arc<CircuitBreaker>>,

    /// Request counter for sequence tracking
    request_counter: Arc<AtomicU64>,

    /// Concurrent request tracker
    concurrent_requests: Arc<AtomicUsize>,

    /// Telemetry
    metrics: Arc<ErrorInjectionMetrics>,
}

impl ErrorInjectionOrchestrator {
    pub fn new(config: ErrorInjectionConfig) -> Result<Self, OrchestratorError> {
        let strategies = Self::build_strategies(&config)?;

        let circuit_breaker = config.circuit_breaker.as_ref().map(|cb_config| {
            // Choose formatter based on primary provider
            let formatter: Arc<dyn ErrorFormatter> = Arc::new(OpenAIFormatter);
            Arc::new(CircuitBreaker::new(cb_config.clone(), formatter))
        });

        Ok(Self {
            config,
            strategies,
            circuit_breaker,
            request_counter: Arc::new(AtomicU64::new(0)),
            concurrent_requests: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(ErrorInjectionMetrics::new()),
        })
    }

    /// Process request and determine if error should be injected
    pub async fn process_request(
        &self,
        mut context: RequestContext,
    ) -> Result<Option<InjectedError>, OrchestratorError> {
        if !self.config.enabled {
            return Ok(None);
        }

        // Update context with sequence number
        let seq = self.request_counter.fetch_add(1, Ordering::Relaxed);
        context.sequence_number = seq;

        // Update concurrent request count
        context.concurrent_requests = self.concurrent_requests.load(Ordering::Relaxed);

        // Increment concurrent counter
        self.concurrent_requests.fetch_add(1, Ordering::Relaxed);

        // Check circuit breaker first
        if let Some(cb) = &self.circuit_breaker {
            if !cb.allow_request().await {
                let error = cb.circuit_open_error(&context).await;
                self.metrics.record_injection(&error);
                return Ok(Some(error));
            }
        }

        // Evaluate strategies in order
        for strategy in &self.strategies {
            match strategy.should_inject(&context).await {
                Ok(Some(error)) => {
                    // Error injected
                    self.metrics.record_injection(&error);

                    // Record failure in circuit breaker
                    if let Some(cb) = &self.circuit_breaker {
                        cb.record_failure().await;
                    }

                    return Ok(Some(error));
                }
                Ok(None) => {
                    // No error, continue to next strategy
                    continue;
                }
                Err(e) => {
                    // Strategy error - log and continue
                    tracing::warn!("Strategy error: {:?}", e);
                    continue;
                }
            }
        }

        // No error injected - record success
        if let Some(cb) = &self.circuit_breaker {
            cb.record_success().await;
        }

        Ok(None)
    }

    /// Called when request completes (for concurrent tracking)
    pub fn request_complete(&self) {
        self.concurrent_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current metrics
    pub fn metrics(&self) -> Arc<ErrorInjectionMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Reset all strategies and state
    pub async fn reset(&mut self) {
        self.request_counter.store(0, Ordering::Relaxed);
        self.concurrent_requests.store(0, Ordering::Relaxed);

        for strategy in &mut self.strategies {
            strategy.reset().await;
        }

        if let Some(cb) = &self.circuit_breaker {
            cb.reset().await;
        }

        self.metrics.reset();
    }

    // Private helper methods

    fn build_strategies(
        config: &ErrorInjectionConfig,
    ) -> Result<Vec<Box<dyn ErrorInjectionStrategy>>, OrchestratorError> {
        let mut strategies: Vec<Box<dyn ErrorInjectionStrategy>> = Vec::new();

        for strategy_config in &config.strategies {
            let strategy = Self::build_strategy(strategy_config, config.seed)?;
            strategies.push(strategy);
        }

        Ok(strategies)
    }

    fn build_strategy(
        config: &StrategyConfig,
        seed: Option<u64>,
    ) -> Result<Box<dyn ErrorInjectionStrategy>, OrchestratorError> {
        // Choose formatter (would be provider-specific in real implementation)
        let formatter: Arc<dyn ErrorFormatter> = Arc::new(OpenAIFormatter);

        let strategy: Box<dyn ErrorInjectionStrategy> = match config {
            StrategyConfig::Probabilistic {
                error_type,
                probability,
                time_window,
                continue_on_inject: _,
            } => {
                let mut strat = ProbabilisticStrategy::new(
                    *error_type,
                    *probability,
                    seed,
                    formatter,
                )?;

                if let Some(window) = time_window {
                    strat = strat.with_time_window(window.clone());
                }

                Box::new(strat)
            }

            StrategyConfig::Sequence {
                pattern,
                error_type,
                max_injections,
            } => {
                let mut strat = SequenceStrategy::new(
                    pattern.clone(),
                    *error_type,
                    formatter,
                );

                if let Some(max) = max_injections {
                    strat = strat.with_max_injections(*max);
                }

                Box::new(strat)
            }

            StrategyConfig::TimeBased {
                trigger,
                error_type,
                duration,
            } => {
                Box::new(TimeBasedStrategy::new(
                    trigger.clone(),
                    *error_type,
                    *duration,
                    formatter,
                ))
            }

            StrategyConfig::Conditional {
                condition,
                error_type,
                probability,
            } => {
                Box::new(ConditionalStrategy {
                    condition: condition.clone(),
                    error_type: *error_type,
                    probability: *probability,
                    rng: Mutex::new(ChaCha8Rng::seed_from_u64(seed.unwrap_or(0))),
                    formatter,
                })
            }

            StrategyConfig::BudgetExhaustion {
                token_limit,
                reset_period,
                error_type,
            } => {
                Box::new(BudgetExhaustionStrategy::new(
                    *token_limit,
                    *reset_period,
                    *error_type,
                    formatter,
                ))
            }

            StrategyConfig::LoadDependent {
                concurrent_threshold,
                error_type,
                load_multiplier,
            } => {
                Box::new(LoadDependentStrategy::new(
                    *concurrent_threshold,
                    *error_type,
                    *load_multiplier,
                    seed,
                    formatter,
                ))
            }
        };

        Ok(strategy)
    }
}

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("Strategy build error: {0}")]
    StrategyBuildError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Metrics for error injection
#[derive(Debug)]
pub struct ErrorInjectionMetrics {
    total_requests: AtomicU64,
    total_injections: AtomicU64,
    injections_by_type: RwLock<HashMap<ErrorType, u64>>,
}

impl ErrorInjectionMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_injections: AtomicU64::new(0),
            injections_by_type: RwLock::new(HashMap::new()),
        }
    }

    pub fn record_injection(&self, error: &InjectedError) {
        self.total_injections.fetch_add(1, Ordering::Relaxed);

        let mut by_type = self.injections_by_type.write().unwrap();
        *by_type.entry(error.error_type).or_insert(0) += 1;
    }

    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_injections.store(0, Ordering::Relaxed);

        let mut by_type = self.injections_by_type.write().unwrap();
        by_type.clear();
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let by_type = self.injections_by_type.read().unwrap().clone();

        MetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_injections: self.total_injections.load(Ordering::Relaxed),
            injections_by_type: by_type,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub total_injections: u64,
    pub injections_by_type: HashMap<ErrorType, u64>,
}
```

### 8.2 Example Usage

```rust
// File: examples/error_injection_demo.rs

use llm_simulator::errors::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure error injection
    let config = ErrorInjectionConfig {
        enabled: true,
        seed: Some(42), // Deterministic
        strategies: vec![
            // 5% chance of rate limit
            StrategyConfig::Probabilistic {
                error_type: ErrorType::RateLimit,
                probability: 0.05,
                time_window: None,
                continue_on_inject: true,
            },
            // Fail every 10th request
            StrategyConfig::Sequence {
                pattern: SequencePattern::EveryNth { n: 10, offset: 0 },
                error_type: ErrorType::InternalServerError,
                max_injections: Some(3),
            },
            // Quota limit of 100k tokens
            StrategyConfig::BudgetExhaustion {
                token_limit: 100000,
                reset_period: Duration::from_secs(3600),
                error_type: ErrorType::QuotaExceeded,
            },
        ],
        circuit_breaker: Some(CircuitBreakerConfig {
            failure_threshold: 5,
            window_duration: Duration::from_secs(10),
            open_duration: Duration::from_secs(30),
            half_open_success_threshold: 3,
            open_circuit_error: ErrorType::CircuitBreakerOpen,
        }),
        rate_limits: None,
        quota: None,
        provider_overrides: HashMap::new(),
        telemetry: TelemetryConfig::default(),
    };

    // Create orchestrator
    let orchestrator = ErrorInjectionOrchestrator::new(config)?;

    // Simulate requests
    for i in 0..100 {
        let context = RequestContext {
            request_id: format!("req-{}", i),
            sequence_number: 0, // Will be set by orchestrator
            timestamp: Instant::now(),
            provider: Provider::OpenAI,
            model: "gpt-4".to_string(),
            prompt_tokens: Some(100),
            max_tokens: Some(500),
            api_key_hash: None,
            concurrent_requests: 0, // Will be set by orchestrator
            headers: HashMap::new(),
            body_size: 1024,
            metadata: HashMap::new(),
        };

        match orchestrator.process_request(context).await? {
            Some(error) => {
                println!("❌ Request {} - Error injected: {:?} ({})",
                    i, error.error_type, error.status_code);
                println!("   Message: {}", error.message);
                println!("   Headers: {:?}", error.headers);
            }
            None => {
                println!("✅ Request {} - Success", i);
            }
        }

        orchestrator.request_complete();

        // Small delay
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Print metrics
    let metrics = orchestrator.metrics().snapshot();
    println!("\n📊 Metrics:");
    println!("   Total requests: {}", metrics.total_requests);
    println!("   Total injections: {}", metrics.total_injections);
    println!("   Injection rate: {:.2}%",
        (metrics.total_injections as f64 / metrics.total_requests as f64) * 100.0);
    println!("   By type:");
    for (error_type, count) in metrics.injections_by_type {
        println!("     {:?}: {}", error_type, count);
    }

    Ok(())
}
```

---

## Summary

This error injection framework provides:

1. **Comprehensive Error Coverage**: All documented LLM provider errors (429, 401, 403, 400, 500-504, timeouts, quotas, content filters)

2. **Multiple Injection Strategies**:
   - Probabilistic (random with configurable rates)
   - Sequence-based (every Nth, ranges, bursts)
   - Time-based (triggers and durations)
   - Conditional (based on request characteristics)
   - Budget exhaustion (quota simulation)
   - Load-dependent (based on concurrent requests)

3. **Provider-Accurate Formatting**:
   - OpenAI (detailed error codes, rate limit headers)
   - Anthropic (Claude-specific error types)
   - Google (gRPC status codes, Gemini format)
   - Azure (Microsoft-specific headers)
   - Cohere (simple format)

4. **Circuit Breaker Simulation**: Full state machine (Closed → Open → Half-Open) with configurable thresholds

5. **Chaos Engineering Scenarios**: Pre-configured patterns for common failure modes

6. **Accurate Retry Headers**: Provider-specific Retry-After and X-RateLimit-* headers

7. **Production-Ready**: Deterministic seeding, comprehensive metrics, full async support, thread-safe

This framework enables complete chaos engineering and failure testing for LLM applications!
