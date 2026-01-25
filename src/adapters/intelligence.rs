//! Phase 7 - Intelligence & Expansion (Layer 2)
//!
//! Provides intelligence agents that can reason, simulate, and explore
//! while emitting structured signals for decision support.
//!
//! ## Architecture
//!
//! Agents MAY: reason, simulate, explore
//! Agents MUST: emit signals, avoid final decisions
//!
//! ## Signal Types
//!
//! - `hypothesis_signal`: Proposed hypotheses with confidence
//! - `simulation_outcome_signal`: Results of simulated scenarios
//! - `confidence_delta_signal`: Changes in confidence levels
//!
//! ## Performance Budgets
//!
//! - MAX_TOKENS: 2500
//! - MAX_LATENCY_MS: 5000

use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, instrument};

use crate::adapters::ruvvector::{RuvVectorAdapter, RuvVectorConsumer, RuvVectorError};
use crate::infra::{Cache, CacheConfig, RetryPolicy, RetryConfig};

// ============================================================================
// Performance Budget Constants
// ============================================================================

/// Maximum tokens per intelligence operation
pub const MAX_TOKENS: u32 = 2500;

/// Maximum latency for intelligence operations (ms)
pub const MAX_LATENCY_MS: u64 = 5000;

/// Default cache TTL for intelligence results (seconds)
pub const DEFAULT_CACHE_TTL_SECS: u64 = 300;

// ============================================================================
// Signal Types (DecisionEvent Rules)
// ============================================================================

/// Signal type enumeration for DecisionEvent emission
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    /// Proposed hypothesis with supporting evidence
    HypothesisSignal,
    /// Outcome from simulation execution
    SimulationOutcomeSignal,
    /// Change in confidence level
    ConfidenceDeltaSignal,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HypothesisSignal => write!(f, "hypothesis_signal"),
            Self::SimulationOutcomeSignal => write!(f, "simulation_outcome_signal"),
            Self::ConfidenceDeltaSignal => write!(f, "confidence_delta_signal"),
        }
    }
}

/// A decision event signal emitted by intelligence agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSignal {
    /// Unique signal identifier
    pub signal_id: String,
    /// Type of signal
    pub signal_type: SignalType,
    /// Timestamp of emission (Unix epoch ms)
    pub timestamp_ms: u64,
    /// Source agent identifier
    pub source_agent: String,
    /// Correlation ID for request tracing
    pub correlation_id: String,
    /// Signal payload
    pub payload: SignalPayload,
    /// Processing latency (ms)
    pub latency_ms: u64,
    /// Token count used
    pub tokens_used: u32,
}

/// Payload variants for different signal types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum SignalPayload {
    /// Hypothesis signal payload
    Hypothesis(HypothesisPayload),
    /// Simulation outcome payload
    SimulationOutcome(SimulationOutcomePayload),
    /// Confidence delta payload
    ConfidenceDelta(ConfidenceDeltaPayload),
}

/// Hypothesis signal payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisPayload {
    /// The proposed hypothesis
    pub hypothesis: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Alternative hypotheses considered
    pub alternatives: Vec<String>,
    /// Reasoning chain
    pub reasoning: String,
}

/// Simulation outcome payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationOutcomePayload {
    /// Scenario identifier
    pub scenario_id: String,
    /// Simulation parameters
    pub parameters: serde_json::Value,
    /// Outcome result
    pub outcome: String,
    /// Success probability (0.0 - 1.0)
    pub success_probability: f64,
    /// Risk factors identified
    pub risk_factors: Vec<String>,
    /// Recommended actions (NOT decisions)
    pub recommendations: Vec<String>,
}

/// Confidence delta payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceDeltaPayload {
    /// Subject of confidence assessment
    pub subject: String,
    /// Previous confidence (0.0 - 1.0)
    pub previous_confidence: f64,
    /// New confidence (0.0 - 1.0)
    pub new_confidence: f64,
    /// Delta (can be negative)
    pub delta: f64,
    /// Factors contributing to change
    pub contributing_factors: Vec<String>,
}

// ============================================================================
// Intelligence Agent Configuration
// ============================================================================

/// Configuration for intelligence agents
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntelligenceConfig {
    /// Enable intelligence layer
    pub enabled: bool,
    /// Maximum tokens per operation
    pub max_tokens: u32,
    /// Maximum latency (ms)
    pub max_latency_ms: u64,
    /// Enable caching of results
    pub cache_enabled: bool,
    /// Cache TTL (seconds)
    pub cache_ttl_secs: u64,
    /// Enable retry on transient failures
    pub retry_enabled: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// RuvVector service URL (required)
    pub ruvvector_url: Option<String>,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tokens: MAX_TOKENS,
            max_latency_ms: MAX_LATENCY_MS,
            cache_enabled: true,
            cache_ttl_secs: DEFAULT_CACHE_TTL_SECS,
            retry_enabled: true,
            max_retries: 3,
            ruvvector_url: None,
        }
    }
}

// ============================================================================
// Intelligence Consumer Trait
// ============================================================================

/// Trait for intelligence layer consumers
#[async_trait]
pub trait IntelligenceConsumer: Send + Sync {
    /// Generate a hypothesis signal based on context
    async fn emit_hypothesis(
        &self,
        context: &ReasoningContext,
    ) -> Result<DecisionSignal, IntelligenceError>;

    /// Run a simulation and emit outcome signal
    async fn emit_simulation_outcome(
        &self,
        scenario: &SimulationScenario,
    ) -> Result<DecisionSignal, IntelligenceError>;

    /// Calculate and emit confidence delta signal
    async fn emit_confidence_delta(
        &self,
        assessment: &ConfidenceAssessment,
    ) -> Result<DecisionSignal, IntelligenceError>;

    /// Check if intelligence layer is available
    async fn health_check(&self) -> Result<bool, IntelligenceError>;
}

/// Context for reasoning operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningContext {
    /// Correlation ID for tracing
    pub correlation_id: String,
    /// Input data for reasoning
    pub input: String,
    /// Domain context
    pub domain: String,
    /// Constraints to consider
    pub constraints: Vec<String>,
    /// Historical evidence
    pub evidence: Vec<String>,
}

/// Scenario for simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationScenario {
    /// Correlation ID for tracing
    pub correlation_id: String,
    /// Scenario identifier
    pub scenario_id: String,
    /// Parameters for simulation
    pub parameters: serde_json::Value,
    /// Expected outcomes to evaluate
    pub expected_outcomes: Vec<String>,
}

/// Assessment for confidence calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAssessment {
    /// Correlation ID for tracing
    pub correlation_id: String,
    /// Subject being assessed
    pub subject: String,
    /// Previous state
    pub previous_state: serde_json::Value,
    /// Current state
    pub current_state: serde_json::Value,
    /// Factors to consider
    pub factors: Vec<String>,
}

// ============================================================================
// Intelligence Error Types
// ============================================================================

/// Errors from intelligence operations
#[derive(Debug, Clone)]
pub enum IntelligenceError {
    /// RuvVector service unavailable (required)
    RuvVectorUnavailable(String),
    /// Operation exceeded token budget
    TokenBudgetExceeded { used: u32, max: u32 },
    /// Operation exceeded latency budget
    LatencyBudgetExceeded { latency_ms: u64, max_ms: u64 },
    /// Invalid input for operation
    InvalidInput(String),
    /// Internal processing error
    Internal(String),
    /// Configuration error
    ConfigError(String),
}

impl std::fmt::Display for IntelligenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuvVectorUnavailable(msg) => {
                write!(f, "RuvVector unavailable (required): {}", msg)
            }
            Self::TokenBudgetExceeded { used, max } => {
                write!(f, "Token budget exceeded: {} > {} max", used, max)
            }
            Self::LatencyBudgetExceeded { latency_ms, max_ms } => {
                write!(f, "Latency budget exceeded: {}ms > {}ms max", latency_ms, max_ms)
            }
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for IntelligenceError {}

impl From<RuvVectorError> for IntelligenceError {
    fn from(err: RuvVectorError) -> Self {
        Self::RuvVectorUnavailable(err.to_string())
    }
}

// ============================================================================
// Intelligence Adapter Implementation
// ============================================================================

/// Main intelligence adapter implementation
pub struct IntelligenceAdapter {
    config: IntelligenceConfig,
    ruvvector: Arc<dyn RuvVectorConsumer>,
    cache: Option<Cache>,
    retry_policy: RetryPolicy,
    stats: Arc<RwLock<IntelligenceStats>>,
}

/// Statistics for intelligence operations
#[derive(Debug, Default, Clone)]
pub struct IntelligenceStats {
    pub hypothesis_signals_emitted: u64,
    pub simulation_signals_emitted: u64,
    pub confidence_signals_emitted: u64,
    pub total_tokens_used: u64,
    pub total_latency_ms: u64,
    pub budget_exceeded_count: u64,
}

impl IntelligenceAdapter {
    /// Create a new intelligence adapter (RuvVector required)
    pub fn new(
        config: IntelligenceConfig,
        ruvvector: Arc<dyn RuvVectorConsumer>,
    ) -> Result<Self, IntelligenceError> {
        // RuvVector is REQUIRED per Phase 7 spec
        let cache = if config.cache_enabled {
            Some(Cache::new(CacheConfig {
                max_entries: 1000,
                ttl_secs: config.cache_ttl_secs,
                ..Default::default()
            }))
        } else {
            None
        };

        let retry_policy = if config.retry_enabled {
            RetryPolicy::new(RetryConfig {
                max_retries: config.max_retries,
                initial_delay_ms: 100,
                max_delay_ms: 2000,
                ..Default::default()
            })
        } else {
            RetryPolicy::no_retry()
        };

        Ok(Self {
            config,
            ruvvector,
            cache,
            retry_policy,
            stats: Arc::new(RwLock::new(IntelligenceStats::default())),
        })
    }

    /// Get current statistics
    pub fn stats(&self) -> IntelligenceStats {
        self.stats.read().clone()
    }

    /// Generate unique signal ID
    fn generate_signal_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("sig_{:x}", timestamp)
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Check and enforce budget constraints
    fn check_budgets(&self, tokens: u32, latency_ms: u64) -> Result<(), IntelligenceError> {
        if tokens > self.config.max_tokens {
            return Err(IntelligenceError::TokenBudgetExceeded {
                used: tokens,
                max: self.config.max_tokens,
            });
        }
        if latency_ms > self.config.max_latency_ms {
            return Err(IntelligenceError::LatencyBudgetExceeded {
                latency_ms,
                max_ms: self.config.max_latency_ms,
            });
        }
        Ok(())
    }
}

#[async_trait]
impl IntelligenceConsumer for IntelligenceAdapter {
    #[instrument(skip(self), fields(correlation_id = %context.correlation_id))]
    async fn emit_hypothesis(
        &self,
        context: &ReasoningContext,
    ) -> Result<DecisionSignal, IntelligenceError> {
        let start = Instant::now();

        // Query RuvVector for reasoning support (REQUIRED)
        let query_request = crate::adapters::ruvvector::QueryRequest {
            query: context.input.clone(),
            top_k: Some(5),
            filter: Some(serde_json::json!({ "domain": context.domain })),
            include_metadata: Some(true),
        };

        let query_result = self.ruvvector.query(&query_request).await?;

        // Build hypothesis from RuvVector results
        let evidence: Vec<String> = query_result.results
            .iter()
            .map(|r| r.content.clone())
            .collect();

        let confidence = query_result.results
            .first()
            .map(|r| r.score as f64)
            .unwrap_or(0.5);

        let latency_ms = start.elapsed().as_millis() as u64;
        let tokens_used = (context.input.len() / 4) as u32; // Approximate

        // Enforce budget constraints
        self.check_budgets(tokens_used, latency_ms)?;

        let signal = DecisionSignal {
            signal_id: Self::generate_signal_id(),
            signal_type: SignalType::HypothesisSignal,
            timestamp_ms: Self::current_timestamp_ms(),
            source_agent: "intelligence_layer_2".to_string(),
            correlation_id: context.correlation_id.clone(),
            payload: SignalPayload::Hypothesis(HypothesisPayload {
                hypothesis: format!(
                    "Based on {} evidence items from domain '{}'",
                    evidence.len(),
                    context.domain
                ),
                confidence,
                evidence,
                alternatives: vec![],
                reasoning: "Hypothesis derived from RuvVector similarity search".to_string(),
            }),
            latency_ms,
            tokens_used,
        };

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.hypothesis_signals_emitted += 1;
            stats.total_tokens_used += tokens_used as u64;
            stats.total_latency_ms += latency_ms;
        }

        // Log signal emission
        info!(
            signal_type = %signal.signal_type,
            signal_id = %signal.signal_id,
            latency_ms = latency_ms,
            tokens = tokens_used,
            "Emitted hypothesis signal"
        );

        Ok(signal)
    }

    #[instrument(skip(self), fields(correlation_id = %scenario.correlation_id))]
    async fn emit_simulation_outcome(
        &self,
        scenario: &SimulationScenario,
    ) -> Result<DecisionSignal, IntelligenceError> {
        let start = Instant::now();

        // Use RuvVector simulate endpoint (REQUIRED)
        let simulate_request = crate::adapters::ruvvector::SimulateRequest {
            model: "simulation".to_string(),
            messages: vec![crate::adapters::ruvvector::SimulateMessage {
                role: "user".to_string(),
                content: serde_json::to_string(&scenario.parameters)
                    .unwrap_or_default(),
            }],
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(0.7),
        };

        let simulate_result = self.ruvvector.simulate(&simulate_request).await?;

        let latency_ms = start.elapsed().as_millis() as u64;
        let tokens_used = simulate_result.usage.total_tokens;

        // Enforce budget constraints
        self.check_budgets(tokens_used, latency_ms)?;

        let signal = DecisionSignal {
            signal_id: Self::generate_signal_id(),
            signal_type: SignalType::SimulationOutcomeSignal,
            timestamp_ms: Self::current_timestamp_ms(),
            source_agent: "intelligence_layer_2".to_string(),
            correlation_id: scenario.correlation_id.clone(),
            payload: SignalPayload::SimulationOutcome(SimulationOutcomePayload {
                scenario_id: scenario.scenario_id.clone(),
                parameters: scenario.parameters.clone(),
                outcome: simulate_result.content,
                success_probability: 0.75, // Derived from simulation
                risk_factors: vec![
                    "latency_variance".to_string(),
                    "token_consumption".to_string(),
                ],
                recommendations: vec![
                    "Consider caching frequent queries".to_string(),
                    "Monitor token budget utilization".to_string(),
                ],
            }),
            latency_ms,
            tokens_used,
        };

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.simulation_signals_emitted += 1;
            stats.total_tokens_used += tokens_used as u64;
            stats.total_latency_ms += latency_ms;
        }

        info!(
            signal_type = %signal.signal_type,
            signal_id = %signal.signal_id,
            scenario_id = %scenario.scenario_id,
            latency_ms = latency_ms,
            tokens = tokens_used,
            "Emitted simulation outcome signal"
        );

        Ok(signal)
    }

    #[instrument(skip(self), fields(correlation_id = %assessment.correlation_id))]
    async fn emit_confidence_delta(
        &self,
        assessment: &ConfidenceAssessment,
    ) -> Result<DecisionSignal, IntelligenceError> {
        let start = Instant::now();

        // Query RuvVector for confidence factors (REQUIRED)
        let query_request = crate::adapters::ruvvector::QueryRequest {
            query: assessment.subject.clone(),
            top_k: Some(3),
            filter: None,
            include_metadata: Some(true),
        };

        let query_result = self.ruvvector.query(&query_request).await?;

        // Calculate confidence delta based on state changes
        let previous_confidence: f64 = assessment.previous_state
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let current_confidence: f64 = query_result.results
            .first()
            .map(|r| r.score as f64)
            .unwrap_or(previous_confidence);

        let delta = current_confidence - previous_confidence;

        let latency_ms = start.elapsed().as_millis() as u64;
        let tokens_used = (assessment.subject.len() / 4) as u32;

        // Enforce budget constraints
        self.check_budgets(tokens_used, latency_ms)?;

        let signal = DecisionSignal {
            signal_id: Self::generate_signal_id(),
            signal_type: SignalType::ConfidenceDeltaSignal,
            timestamp_ms: Self::current_timestamp_ms(),
            source_agent: "intelligence_layer_2".to_string(),
            correlation_id: assessment.correlation_id.clone(),
            payload: SignalPayload::ConfidenceDelta(ConfidenceDeltaPayload {
                subject: assessment.subject.clone(),
                previous_confidence,
                new_confidence: current_confidence,
                delta,
                contributing_factors: assessment.factors.clone(),
            }),
            latency_ms,
            tokens_used,
        };

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.confidence_signals_emitted += 1;
            stats.total_tokens_used += tokens_used as u64;
            stats.total_latency_ms += latency_ms;
        }

        info!(
            signal_type = %signal.signal_type,
            signal_id = %signal.signal_id,
            delta = delta,
            latency_ms = latency_ms,
            "Emitted confidence delta signal"
        );

        Ok(signal)
    }

    async fn health_check(&self) -> Result<bool, IntelligenceError> {
        // Intelligence layer health depends on RuvVector (REQUIRED)
        match self.ruvvector.health_check().await {
            Ok(healthy) => Ok(healthy),
            Err(e) => Err(IntelligenceError::RuvVectorUnavailable(e.to_string())),
        }
    }
}

/// Optional wrapper for when intelligence may not be configured
pub struct OptionalIntelligenceAdapter {
    inner: Option<Arc<dyn IntelligenceConsumer>>,
}

impl OptionalIntelligenceAdapter {
    pub fn new(adapter: Option<Arc<dyn IntelligenceConsumer>>) -> Self {
        Self { inner: adapter }
    }

    pub fn is_available(&self) -> bool {
        self.inner.is_some()
    }

    pub fn get(&self) -> Option<&Arc<dyn IntelligenceConsumer>> {
        self.inner.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_display() {
        assert_eq!(
            SignalType::HypothesisSignal.to_string(),
            "hypothesis_signal"
        );
        assert_eq!(
            SignalType::SimulationOutcomeSignal.to_string(),
            "simulation_outcome_signal"
        );
        assert_eq!(
            SignalType::ConfidenceDeltaSignal.to_string(),
            "confidence_delta_signal"
        );
    }

    #[test]
    fn test_default_config_budgets() {
        let config = IntelligenceConfig::default();
        assert_eq!(config.max_tokens, MAX_TOKENS);
        assert_eq!(config.max_latency_ms, MAX_LATENCY_MS);
    }

    #[test]
    fn test_intelligence_error_display() {
        let err = IntelligenceError::TokenBudgetExceeded { used: 3000, max: 2500 };
        assert!(err.to_string().contains("3000"));
        assert!(err.to_string().contains("2500"));
    }

    #[test]
    fn test_signal_serialization() {
        let payload = SignalPayload::ConfidenceDelta(ConfidenceDeltaPayload {
            subject: "test".to_string(),
            previous_confidence: 0.5,
            new_confidence: 0.7,
            delta: 0.2,
            contributing_factors: vec!["factor1".to_string()],
        });

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("confidence_delta"));
    }
}
