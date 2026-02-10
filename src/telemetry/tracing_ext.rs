//! Tracing Extensions
//!
//! Provides utilities for distributed tracing:
//! - Span creation and context propagation
//! - Log-trace correlation
//! - Request/response tracking
//! - FEU (Foundational Execution Unit) span hierarchy and validation

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tracing::{info_span, Span};

use crate::adapters::observatory::{ConsumedSpan, FeuSpanKind, SpanEvent, SpanStatus};

// ============================================================================
// Existing types (preserved)
// ============================================================================

/// Request span for distributed tracing
#[derive(Debug)]
pub struct RequestSpan {
    span: Span,
    start: Instant,
    request_id: String,
    trace_id: Option<String>,
    span_id: Option<String>,
}

impl RequestSpan {
    /// Create a new request span
    pub fn new(
        request_id: &str,
        method: &str,
        path: &str,
        model: Option<&str>,
        provider: Option<&str>,
    ) -> Self {
        let span = info_span!(
            "http_request",
            request_id = %request_id,
            method = %method,
            path = %path,
            model = model.unwrap_or("unknown"),
            provider = provider.unwrap_or("unknown"),
            trace_id = tracing::field::Empty,
            span_id = tracing::field::Empty,
            status = tracing::field::Empty,
            latency_ms = tracing::field::Empty,
            input_tokens = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
        );

        // Extract trace context if available
        let (trace_id, span_id) = extract_trace_context();

        if let Some(ref tid) = trace_id {
            span.record("trace_id", tid.as_str());
        }
        if let Some(ref sid) = span_id {
            span.record("span_id", sid.as_str());
        }

        Self {
            span,
            start: Instant::now(),
            request_id: request_id.to_string(),
            trace_id,
            span_id,
        }
    }

    /// Get the underlying span
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Record response status
    pub fn record_status(&self, status: u16) {
        self.span.record("status", status);
    }

    /// Record token counts
    pub fn record_tokens(&self, input: u32, output: u32) {
        self.span.record("input_tokens", input);
        self.span.record("output_tokens", output);
    }

    /// Record latency
    pub fn record_latency(&self) {
        self.span.record("latency_ms", self.start.elapsed().as_millis() as u64);
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Get trace ID if available
    pub fn trace_id(&self) -> Option<&str> {
        self.trace_id.as_deref()
    }

    /// Get span ID if available
    pub fn span_id(&self) -> Option<&str> {
        self.span_id.as_deref()
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

/// Extract trace context from current span
/// Returns (trace_id, span_id) if available
fn extract_trace_context() -> (Option<String>, Option<String>) {
    // In a real OTEL implementation, we'd extract from the current span context
    // For now, we generate a pseudo trace ID based on the current time
    // This will be replaced with real OTEL context when enabled

    #[cfg(feature = "otel")]
    {
        use opentelemetry::trace::TraceContextExt;
        use tracing_opentelemetry::OpenTelemetrySpanExt;

        let context = tracing::Span::current().context();
        let span_ref = context.span();
        let span_context = span_ref.span_context();

        if span_context.is_valid() {
            return (
                Some(span_context.trace_id().to_string()),
                Some(span_context.span_id().to_string()),
            );
        }
    }

    // Fallback: no trace context available
    (None, None)
}

/// Log with trace context
#[macro_export]
macro_rules! trace_log {
    ($level:ident, $($arg:tt)*) => {
        tracing::$level!(
            target: "llm_simulator",
            $($arg)*
        )
    };
}

/// Create a span for database/external operations
pub fn create_operation_span(operation: &str, target: &str) -> Span {
    info_span!(
        "operation",
        operation = %operation,
        target = %target,
        duration_ms = tracing::field::Empty,
        success = tracing::field::Empty,
    )
}

/// Create a span for streaming operations
pub fn create_stream_span(request_id: &str, model: &str) -> Span {
    info_span!(
        "stream",
        request_id = %request_id,
        model = %model,
        chunks_sent = tracing::field::Empty,
        total_tokens = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
    )
}

/// Trace context holder for log correlation
#[derive(Debug, Clone, Default)]
pub struct TraceContext {
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub parent_span_id: Option<String>,
}

impl TraceContext {
    /// Create from current span
    pub fn current() -> Self {
        let (trace_id, span_id) = extract_trace_context();
        Self {
            trace_id,
            span_id,
            parent_span_id: None,
        }
    }

    /// Create with an explicit parent span ID (for FEU paths)
    pub fn with_parent(parent_span_id: String) -> Self {
        let (trace_id, span_id) = extract_trace_context();
        Self {
            trace_id,
            span_id,
            parent_span_id: Some(parent_span_id),
        }
    }

    /// Check if context is valid
    pub fn is_valid(&self) -> bool {
        self.trace_id.is_some() && self.span_id.is_some()
    }
}

// ============================================================================
// FEU (Foundational Execution Unit) Types
// ============================================================================

/// Sentinel parent for the repo-level root span.
/// The repo span uses this as its parent_span_id to indicate it is the root.
pub const FEU_ROOT_PARENT: &str = "ROOT";

/// An artifact attached to an agent-level span.
/// Wraps heterogeneous data (signals, lineage, etc.) as serialized JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanArtifact {
    /// Artifact kind (e.g., "decision_signal", "agent_lineage", "state_transition")
    pub kind: String,
    /// Serialized artifact payload
    pub payload: serde_json::Value,
    /// Timestamp of artifact creation (unix millis)
    pub created_at: u64,
}

/// Errors from FEU span validation
#[derive(Debug, Clone, thiserror::Error)]
pub enum FeuValidationError {
    #[error("Span {span_id} has no parent_span_id (FEU invariant violation)")]
    MissingParentSpanId { span_id: String },

    #[error("Duplicate agent span for agent: {agent_name}")]
    DuplicateAgentSpan { agent_name: String },

    #[error("Span not found: {span_id}")]
    SpanNotFound { span_id: String },

    #[error("Repo span not properly initialized")]
    RepoSpanMissing,
}

/// The complete execution trace returned to the caller.
/// Contains the repo span, all agent spans, and validates FEU invariants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Unique trace identifier (shared across all spans)
    pub trace_id: String,
    /// The repo-level root span
    pub repo_span: ConsumedSpan,
    /// All agent-level spans
    pub agent_spans: Vec<ConsumedSpan>,
    /// Artifacts attached at the agent level, keyed by span_id
    pub artifacts: HashMap<String, Vec<SpanArtifact>>,
    /// Overall execution status (derived from child span statuses)
    pub status: SpanStatus,
    /// Total wall-clock duration in milliseconds
    pub total_duration_ms: f64,
}

/// Collects spans during a single execution and enforces FEU invariants.
///
/// Usage:
///   1. Call `FeuSpanCollector::new(trace_id)` at the start of execution.
///   2. The repo span is created automatically.
///   3. Call `begin_agent_span(agent_name)` for each agent.
///   4. Attach artifacts via `attach_artifact(span_id, artifact)`.
///   5. Call `end_agent_span(span_id, status)` when agent completes.
///   6. Call `finalize()` to produce the `ExecutionTrace`.
pub struct FeuSpanCollector {
    trace_id: String,
    repo_span_id: String,
    repo_span: ConsumedSpan,
    agent_spans: Vec<ConsumedSpan>,
    artifacts: HashMap<String, Vec<SpanArtifact>>,
    started_agents: HashSet<String>,
}

impl FeuSpanCollector {
    /// Create a new collector. Generates trace_id and repo span automatically.
    pub fn new(trace_id: Option<String>) -> Self {
        let trace_id = trace_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let repo_span_id = format!("repo_{}", uuid::Uuid::new_v4().to_string().replace('-', "")[..16].to_string());

        let repo_span = ConsumedSpan {
            span_id: repo_span_id.clone(),
            trace_id: trace_id.clone(),
            parent_span_id: Some(FEU_ROOT_PARENT.to_string()),
            name: "repo_execution".to_string(),
            provider: None,
            model: None,
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
            span_kind: Some(FeuSpanKind::Repo),
        };

        Self {
            trace_id,
            repo_span_id,
            repo_span,
            agent_spans: Vec::new(),
            artifacts: HashMap::new(),
            started_agents: HashSet::new(),
        }
    }

    /// Returns the trace_id.
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    /// Returns the repo span_id (for use as parent by agent spans).
    pub fn repo_span_id(&self) -> &str {
        &self.repo_span_id
    }

    /// Begin a new agent-level span. Returns the span_id.
    /// Returns Err if agent_name was already started (no duplicates).
    pub fn begin_agent_span(&mut self, agent_name: &str) -> Result<String, FeuValidationError> {
        if self.started_agents.contains(agent_name) {
            return Err(FeuValidationError::DuplicateAgentSpan {
                agent_name: agent_name.to_string(),
            });
        }

        let span_id = format!(
            "agent_{}_{}", agent_name,
            uuid::Uuid::new_v4().to_string().replace('-', "")[..12].to_string()
        );

        let span = ConsumedSpan {
            span_id: span_id.clone(),
            trace_id: self.trace_id.clone(),
            parent_span_id: Some(self.repo_span_id.clone()),
            name: format!("agent:{}", agent_name),
            provider: None,
            model: None,
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
            span_kind: Some(FeuSpanKind::Agent),
        };

        self.started_agents.insert(agent_name.to_string());
        self.agent_spans.push(span);

        Ok(span_id)
    }

    /// Attach an artifact to a specific agent span.
    pub fn attach_artifact(&mut self, span_id: &str, artifact: SpanArtifact) -> Result<(), FeuValidationError> {
        // Verify the span exists
        if !self.agent_spans.iter().any(|s| s.span_id == span_id) {
            return Err(FeuValidationError::SpanNotFound {
                span_id: span_id.to_string(),
            });
        }

        self.artifacts
            .entry(span_id.to_string())
            .or_default()
            .push(artifact);

        Ok(())
    }

    /// End an agent span with a given status. Records end_time.
    pub fn end_agent_span(&mut self, span_id: &str, status: SpanStatus) -> Result<(), FeuValidationError> {
        let span = self.agent_spans.iter_mut()
            .find(|s| s.span_id == *span_id)
            .ok_or_else(|| FeuValidationError::SpanNotFound {
                span_id: span_id.to_string(),
            })?;

        span.status = status;
        span.end_time = Some(chrono::Utc::now().timestamp_millis() as u64);
        span.latency_ms = Some((span.end_time.unwrap() - span.start_time) as f64);

        Ok(())
    }

    /// Fail an agent span. Adds error event and propagates failure to the repo span.
    pub fn fail_agent_span(&mut self, span_id: &str, error_msg: &str) -> Result<(), FeuValidationError> {
        let span = self.agent_spans.iter_mut()
            .find(|s| s.span_id == *span_id)
            .ok_or_else(|| FeuValidationError::SpanNotFound {
                span_id: span_id.to_string(),
            })?;

        span.status = SpanStatus::Failed;
        span.end_time = Some(chrono::Utc::now().timestamp_millis() as u64);
        span.latency_ms = Some((span.end_time.unwrap() - span.start_time) as f64);

        span.events.push(SpanEvent {
            name: "error".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("error.message".to_string(), error_msg.to_string());
                attrs
            },
        });

        // Propagate failure to repo span
        self.repo_span.status = SpanStatus::Failed;

        Ok(())
    }

    /// Finalize the execution. Validates all invariants and returns ExecutionTrace.
    pub fn finalize(mut self) -> Result<ExecutionTrace, FeuValidationError> {
        // Validate all agent spans have parent_span_id
        for span in &self.agent_spans {
            Self::validate_span(span)?;
        }

        // Set repo span end_time
        self.repo_span.end_time = Some(chrono::Utc::now().timestamp_millis() as u64);
        self.repo_span.latency_ms = Some(
            (self.repo_span.end_time.unwrap() - self.repo_span.start_time) as f64,
        );

        // Derive overall status: if ANY agent span is Failed, repo is Failed
        let has_failure = self.agent_spans.iter().any(|s| s.status == SpanStatus::Failed);
        if has_failure {
            self.repo_span.status = SpanStatus::Failed;
        } else if self.repo_span.status == SpanStatus::Unset {
            self.repo_span.status = SpanStatus::Ok;
        }

        let total_duration_ms = self.repo_span.latency_ms.unwrap_or(0.0);

        Ok(ExecutionTrace {
            trace_id: self.trace_id,
            status: self.repo_span.status.clone(),
            repo_span: self.repo_span,
            agent_spans: self.agent_spans,
            artifacts: self.artifacts,
            total_duration_ms,
        })
    }

    /// Validate a single span's FEU compliance.
    fn validate_span(span: &ConsumedSpan) -> Result<(), FeuValidationError> {
        if span.parent_span_id.is_none() {
            return Err(FeuValidationError::MissingParentSpanId {
                span_id: span.span_id.clone(),
            });
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_span_creation() {
        let span = RequestSpan::new(
            "req-123",
            "POST",
            "/v1/chat/completions",
            Some("gpt-4"),
            Some("openai"),
        );

        assert_eq!(span.request_id(), "req-123");
    }

    #[test]
    fn test_request_span_recording() {
        let span = RequestSpan::new(
            "req-456",
            "POST",
            "/v1/chat/completions",
            Some("gpt-4"),
            None,
        );

        span.record_status(200);
        span.record_tokens(100, 50);
        span.record_latency();

        // Span should be recordable without panicking
    }

    #[test]
    fn test_trace_context() {
        let ctx = TraceContext::current();
        // Without OTEL, context should be empty
        assert!(!ctx.is_valid());
    }

    #[test]
    fn test_trace_context_with_parent() {
        let ctx = TraceContext::with_parent("parent-span-123".to_string());
        assert_eq!(ctx.parent_span_id, Some("parent-span-123".to_string()));
    }

    #[test]
    fn test_operation_span() {
        let _span = create_operation_span("query", "engine");
        // Should create without panicking
    }

    #[test]
    fn test_stream_span() {
        let _span = create_stream_span("req-789", "gpt-4");
        // Should create without panicking
    }

    // ========================================================================
    // FEU invariant tests
    // ========================================================================

    #[test]
    fn test_collector_creates_repo_span() {
        let collector = FeuSpanCollector::new(Some("trace-001".to_string()));
        assert_eq!(collector.trace_id(), "trace-001");
        assert!(collector.repo_span_id().starts_with("repo_"));
        assert_eq!(collector.repo_span.parent_span_id, Some(FEU_ROOT_PARENT.to_string()));
        assert_eq!(collector.repo_span.span_kind, Some(FeuSpanKind::Repo));
        assert_eq!(collector.repo_span.name, "repo_execution");
    }

    #[test]
    fn test_agent_span_has_parent() {
        let mut collector = FeuSpanCollector::new(None);
        let repo_id = collector.repo_span_id().to_string();

        let span_id = collector.begin_agent_span("intelligence").unwrap();

        let agent_span = collector.agent_spans.iter()
            .find(|s| s.span_id == span_id)
            .unwrap();

        assert_eq!(agent_span.parent_span_id, Some(repo_id));
        assert_eq!(agent_span.span_kind, Some(FeuSpanKind::Agent));
        assert_eq!(agent_span.name, "agent:intelligence");
    }

    #[test]
    fn test_duplicate_agent_rejected() {
        let mut collector = FeuSpanCollector::new(None);
        collector.begin_agent_span("intelligence").unwrap();

        let result = collector.begin_agent_span("intelligence");
        assert!(matches!(result, Err(FeuValidationError::DuplicateAgentSpan { .. })));
    }

    #[test]
    fn test_failure_propagates_to_repo() {
        let mut collector = FeuSpanCollector::new(None);
        let span_id = collector.begin_agent_span("engine").unwrap();

        collector.fail_agent_span(&span_id, "something broke").unwrap();

        // Repo span should be Failed due to propagation
        assert_eq!(collector.repo_span.status, SpanStatus::Failed);

        // Agent span should have error event
        let agent_span = collector.agent_spans.iter()
            .find(|s| s.span_id == span_id)
            .unwrap();
        assert_eq!(agent_span.status, SpanStatus::Failed);
        assert_eq!(agent_span.events.len(), 1);
        assert_eq!(agent_span.events[0].name, "error");
        assert_eq!(
            agent_span.events[0].attributes.get("error.message"),
            Some(&"something broke".to_string())
        );
    }

    #[test]
    fn test_finalize_validates_all_spans() {
        let mut collector = FeuSpanCollector::new(None);
        let span_id = collector.begin_agent_span("engine").unwrap();
        collector.end_agent_span(&span_id, SpanStatus::Ok).unwrap();

        let trace = collector.finalize().unwrap();

        // Repo span should be Ok (no failures)
        assert_eq!(trace.status, SpanStatus::Ok);
        assert_eq!(trace.repo_span.status, SpanStatus::Ok);
        assert!(trace.repo_span.end_time.is_some());
        assert!(trace.total_duration_ms >= 0.0);
    }

    #[test]
    fn test_artifact_attachment() {
        let mut collector = FeuSpanCollector::new(None);
        let span_id = collector.begin_agent_span("intelligence").unwrap();

        let artifact = SpanArtifact {
            kind: "decision_signal".to_string(),
            payload: serde_json::json!({"hypothesis": "test"}),
            created_at: 1234567890,
        };
        collector.attach_artifact(&span_id, artifact).unwrap();
        collector.end_agent_span(&span_id, SpanStatus::Ok).unwrap();

        let trace = collector.finalize().unwrap();

        let artifacts = trace.artifacts.get(&span_id).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, "decision_signal");
    }

    #[test]
    fn test_all_spans_returned_in_trace() {
        let mut collector = FeuSpanCollector::new(Some("trace-multi".to_string()));
        let id1 = collector.begin_agent_span("observatory").unwrap();
        let id2 = collector.begin_agent_span("intelligence").unwrap();
        let id3 = collector.begin_agent_span("engine").unwrap();

        collector.end_agent_span(&id1, SpanStatus::Ok).unwrap();
        collector.end_agent_span(&id2, SpanStatus::Ok).unwrap();
        collector.end_agent_span(&id3, SpanStatus::Ok).unwrap();

        let trace = collector.finalize().unwrap();

        assert_eq!(trace.trace_id, "trace-multi");
        assert_eq!(trace.agent_spans.len(), 3);

        // All agent spans should have the repo span as parent
        for span in &trace.agent_spans {
            assert_eq!(span.parent_span_id, Some(trace.repo_span.span_id.clone()));
        }
    }

    #[test]
    fn test_missing_parent_span_id_fails_validation() {
        // Construct a span with no parent and verify validation catches it
        let span = ConsumedSpan {
            span_id: "orphan".to_string(),
            trace_id: "trace".to_string(),
            parent_span_id: None,
            name: "orphan_span".to_string(),
            provider: None,
            model: None,
            input_summary: None,
            output_summary: None,
            token_usage: None,
            cost_usd: None,
            latency_ms: None,
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
            events: Vec::new(),
            start_time: 1000,
            end_time: Some(2000),
            span_kind: Some(FeuSpanKind::Agent),
        };

        let result = FeuSpanCollector::validate_span(&span);
        assert!(matches!(result, Err(FeuValidationError::MissingParentSpanId { .. })));
    }

    #[test]
    fn test_attach_to_nonexistent_span_fails() {
        let mut collector = FeuSpanCollector::new(None);
        let artifact = SpanArtifact {
            kind: "test".to_string(),
            payload: serde_json::Value::Null,
            created_at: 0,
        };

        let result = collector.attach_artifact("nonexistent", artifact);
        assert!(matches!(result, Err(FeuValidationError::SpanNotFound { .. })));
    }

    #[test]
    fn test_finalize_failure_propagation() {
        let mut collector = FeuSpanCollector::new(None);
        let id1 = collector.begin_agent_span("observatory").unwrap();
        let id2 = collector.begin_agent_span("engine").unwrap();

        collector.end_agent_span(&id1, SpanStatus::Ok).unwrap();
        collector.fail_agent_span(&id2, "engine failed").unwrap();

        let trace = collector.finalize().unwrap();

        // Overall status should be Failed because engine failed
        assert_eq!(trace.status, SpanStatus::Failed);
        assert_eq!(trace.repo_span.status, SpanStatus::Failed);
    }
}
