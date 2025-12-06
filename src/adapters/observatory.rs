//! LLM-Observatory Adapter
//!
//! Thin runtime consumer for telemetry data, trace spans, and runtime state
//! transitions from the LLM-Observatory service.
//!
//! This adapter provides:
//! - Consumption of external telemetry data
//! - Trace span integration
//! - Runtime state transition tracking
//!
//! ## Usage
//!
//! ```rust,ignore
//! use llm_simulator::adapters::observatory::{ObservatoryConsumer, ObservatoryAdapter};
//!
//! let adapter = ObservatoryAdapter::new();
//! let spans = adapter.consume_trace_spans("request-123").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Trace span data consumed from LLM-Observatory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedSpan {
    /// Unique span identifier
    pub span_id: String,
    /// Parent trace identifier
    pub trace_id: String,
    /// Parent span identifier (if nested)
    pub parent_span_id: Option<String>,
    /// Span name/operation
    pub name: String,
    /// Provider name
    pub provider: Option<String>,
    /// Model name
    pub model: Option<String>,
    /// Input data summary
    pub input_summary: Option<SpanInputSummary>,
    /// Output data summary
    pub output_summary: Option<SpanOutputSummary>,
    /// Token usage
    pub token_usage: Option<TokenUsage>,
    /// Cost in USD
    pub cost_usd: Option<f64>,
    /// Latency in milliseconds
    pub latency_ms: Option<f64>,
    /// Span status
    pub status: SpanStatus,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
    /// Span events
    pub events: Vec<SpanEvent>,
    /// Start timestamp (unix millis)
    pub start_time: u64,
    /// End timestamp (unix millis)
    pub end_time: Option<u64>,
}

/// Summary of span input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanInputSummary {
    /// Input type (text, chat, multimodal)
    pub input_type: String,
    /// Message count (for chat)
    pub message_count: Option<u32>,
    /// Estimated token count
    pub token_estimate: Option<u32>,
}

/// Summary of span output data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanOutputSummary {
    /// Generated content length
    pub content_length: usize,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Output token count
    pub token_count: Option<u32>,
}

/// Token usage data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Input/prompt tokens
    pub input_tokens: u32,
    /// Output/completion tokens
    pub output_tokens: u32,
    /// Total tokens
    pub total_tokens: u32,
}

/// Span execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpanStatus {
    /// Operation completed successfully
    Ok,
    /// Operation failed with error
    Error,
    /// Status not set
    Unset,
}

impl Default for SpanStatus {
    fn default() -> Self {
        Self::Unset
    }
}

/// Event within a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp (unix millis)
    pub timestamp: u64,
    /// Event attributes
    pub attributes: HashMap<String, String>,
}

/// Runtime state transition consumed from LLM-Observatory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedStateTransition {
    /// Transition identifier
    pub id: String,
    /// Source state
    pub from_state: RuntimeState,
    /// Target state
    pub to_state: RuntimeState,
    /// Transition trigger
    pub trigger: String,
    /// Timestamp (unix millis)
    pub timestamp: u64,
    /// Associated request ID
    pub request_id: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Runtime state enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RuntimeState {
    /// System initializing
    Initializing,
    /// Ready to accept requests
    Ready,
    /// Processing a request
    Processing,
    /// Generating streaming response
    Streaming,
    /// Request completed
    Completed,
    /// Error state
    Error,
    /// Shutting down
    ShuttingDown,
    /// Custom state
    Custom(String),
}

/// Telemetry metrics consumed from LLM-Observatory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedTelemetry {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Current value
    pub value: f64,
    /// Labels/tags
    pub labels: HashMap<String, String>,
    /// Timestamp (unix millis)
    pub timestamp: u64,
}

/// Type of metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Consumer trait for LLM-Observatory integration
///
/// Implementors consume telemetry data from the Observatory service
/// without modifying any existing simulator APIs.
#[async_trait]
pub trait ObservatoryConsumer: Send + Sync {
    /// Consume trace spans for a request
    async fn consume_trace_spans(
        &self,
        trace_id: &str,
    ) -> Result<Vec<ConsumedSpan>, AdapterError>;

    /// Consume a specific span by ID
    async fn consume_span(
        &self,
        span_id: &str,
    ) -> Result<Option<ConsumedSpan>, AdapterError>;

    /// Consume state transitions for a time range
    async fn consume_state_transitions(
        &self,
        since: u64,
        until: Option<u64>,
    ) -> Result<Vec<ConsumedStateTransition>, AdapterError>;

    /// Consume telemetry metrics
    async fn consume_telemetry(
        &self,
        metric_names: &[&str],
    ) -> Result<Vec<ConsumedTelemetry>, AdapterError>;

    /// Subscribe to real-time state transitions
    async fn subscribe_state_transitions(
        &self,
    ) -> Result<StateTransitionSubscription, AdapterError>;

    /// Check if the consumer is connected and healthy
    async fn health_check(&self) -> Result<bool, AdapterError>;
}

/// Subscription handle for state transitions
pub struct StateTransitionSubscription {
    _id: String,
}

impl StateTransitionSubscription {
    /// Create a new subscription
    pub fn new(id: String) -> Self {
        Self { _id: id }
    }
}

/// Error type for adapter operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum AdapterError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Data not found: {0}")]
    NotFound(String),

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Upstream service error: {0}")]
    UpstreamError(String),

    #[error("Subscription error: {0}")]
    SubscriptionError(String),
}

/// Default implementation that bridges to llm-observatory-core types
pub struct ObservatoryAdapter {
    /// Cached spans by trace ID
    span_cache: Arc<parking_lot::RwLock<HashMap<String, Vec<ConsumedSpan>>>>,
    /// Cache TTL
    cache_ttl: Duration,
}

impl ObservatoryAdapter {
    /// Create a new adapter with default settings
    pub fn new() -> Self {
        Self {
            span_cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            cache_ttl: Duration::from_secs(60), // 1 minute cache for spans
        }
    }

    /// Create adapter with custom cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Convert from llm-observatory-core LlmSpan to ConsumedSpan
    ///
    /// This bridge function is prepared for when upstream compilation is fixed.
    /// Currently uses placeholder implementation.
    #[allow(dead_code)]
    fn convert_from_upstream_span(
        span_id: &str,
        trace_id: &str,
        name: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> ConsumedSpan {
        // Bridge conversion - ready for upstream types when available
        ConsumedSpan {
            span_id: span_id.to_string(),
            trace_id: trace_id.to_string(),
            parent_span_id: None,
            name: name.to_string(),
            provider: provider.map(|s| s.to_string()),
            model: model.map(|s| s.to_string()),
            input_summary: None,
            output_summary: None,
            token_usage: None,
            cost_usd: None,
            latency_ms: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
            start_time: chrono::Utc::now().timestamp_millis() as u64,
            end_time: None,
        }
    }
}

impl Default for ObservatoryAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ObservatoryConsumer for ObservatoryAdapter {
    async fn consume_trace_spans(
        &self,
        trace_id: &str,
    ) -> Result<Vec<ConsumedSpan>, AdapterError> {
        // Check cache first
        if let Some(cached) = self.span_cache.read().get(trace_id) {
            return Ok(cached.clone());
        }

        // In full implementation, query llm-observatory-core
        // Return empty vec - simulator generates its own traces
        Ok(Vec::new())
    }

    async fn consume_span(
        &self,
        _span_id: &str,
    ) -> Result<Option<ConsumedSpan>, AdapterError> {
        // Single span lookup
        Ok(None)
    }

    async fn consume_state_transitions(
        &self,
        _since: u64,
        _until: Option<u64>,
    ) -> Result<Vec<ConsumedStateTransition>, AdapterError> {
        // State transition history
        Ok(Vec::new())
    }

    async fn consume_telemetry(
        &self,
        _metric_names: &[&str],
    ) -> Result<Vec<ConsumedTelemetry>, AdapterError> {
        // Telemetry metrics consumption
        Ok(Vec::new())
    }

    async fn subscribe_state_transitions(
        &self,
    ) -> Result<StateTransitionSubscription, AdapterError> {
        // Create subscription for real-time updates
        Ok(StateTransitionSubscription::new(uuid::Uuid::new_v4().to_string()))
    }

    async fn health_check(&self) -> Result<bool, AdapterError> {
        // Adapter is always "healthy" as a thin consumption layer
        Ok(true)
    }
}

/// Utility implementations for consumed data
impl ConsumedSpan {
    /// Check if span completed successfully
    pub fn is_success(&self) -> bool {
        self.status == SpanStatus::Ok
    }

    /// Check if span errored
    pub fn is_error(&self) -> bool {
        self.status == SpanStatus::Error
    }

    /// Get total tokens if available
    pub fn total_tokens(&self) -> Option<u32> {
        self.token_usage.as_ref().map(|u| u.total_tokens)
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> Option<f64> {
        self.latency_ms.or_else(|| {
            self.end_time.map(|end| (end - self.start_time) as f64)
        })
    }
}

impl ConsumedStateTransition {
    /// Check if this is an error transition
    pub fn is_error_transition(&self) -> bool {
        self.to_state == RuntimeState::Error
    }

    /// Get transition duration context if available
    pub fn transition_duration(&self) -> Option<Duration> {
        self.context
            .get("duration_ms")
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = ObservatoryAdapter::new();
        assert!(adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_consume_spans_returns_empty() {
        let adapter = ObservatoryAdapter::new();
        let result = adapter.consume_trace_spans("trace-123").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_subscribe_state_transitions() {
        let adapter = ObservatoryAdapter::new();
        let result = adapter.subscribe_state_transitions().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_span_status_default() {
        let status = SpanStatus::default();
        assert_eq!(status, SpanStatus::Unset);
    }

    #[test]
    fn test_span_utilities() {
        let span = ConsumedSpan {
            span_id: "span-1".to_string(),
            trace_id: "trace-1".to_string(),
            parent_span_id: None,
            name: "chat_completion".to_string(),
            provider: Some("openai".to_string()),
            model: Some("gpt-4".to_string()),
            input_summary: None,
            output_summary: None,
            token_usage: Some(TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
            }),
            cost_usd: Some(0.01),
            latency_ms: Some(250.0),
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
            events: Vec::new(),
            start_time: 1000,
            end_time: Some(1250),
        };

        assert!(span.is_success());
        assert!(!span.is_error());
        assert_eq!(span.total_tokens(), Some(150));
        assert_eq!(span.duration_ms(), Some(250.0));
    }

    #[test]
    fn test_state_transition_utilities() {
        let mut context = HashMap::new();
        context.insert("duration_ms".to_string(), "100".to_string());

        let transition = ConsumedStateTransition {
            id: "trans-1".to_string(),
            from_state: RuntimeState::Processing,
            to_state: RuntimeState::Completed,
            trigger: "response_complete".to_string(),
            timestamp: 1234567890,
            request_id: Some("req-1".to_string()),
            context,
        };

        assert!(!transition.is_error_transition());
        assert_eq!(transition.transition_duration(), Some(Duration::from_millis(100)));
    }
}
