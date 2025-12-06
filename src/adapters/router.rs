//! Router Adapter (Layer 2 Module)
//!
//! Thin runtime consumer for routing decisions and conditional branching logic.
//! This adapter integrates with Axum's Router to provide advanced routing
//! capabilities consumed from external routing rules.
//!
//! Note: "Router (Layer 2)" refers to the application-layer routing logic,
//! not network layer 2. This uses Axum's built-in Router type extended
//! with external rule consumption.
//!
//! This adapter provides:
//! - Consumption of external routing decisions
//! - Conditional branching based on request attributes
//! - Model selection routing
//! - Provider fallback chain consumption
//!
//! ## Usage
//!
//! ```rust,ignore
//! use llm_simulator::adapters::router::{RouterConsumer, RouterAdapter};
//!
//! let adapter = RouterAdapter::new();
//! let decision = adapter.consume_routing_decision(&request).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Routing decision consumed from external routing service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedRoutingDecision {
    /// Decision identifier
    pub id: String,
    /// Selected provider
    pub provider: String,
    /// Selected model
    pub model: String,
    /// Routing reason
    pub reason: RoutingReason,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Alternative routes (fallback chain)
    pub alternatives: Vec<AlternativeRoute>,
    /// Decision timestamp
    pub timestamp: u64,
    /// Time to make decision
    pub decision_latency_ms: f64,
}

/// Reason for routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingReason {
    /// Based on cost optimization
    CostOptimization,
    /// Based on latency requirements
    LatencyOptimization,
    /// Based on capability matching
    CapabilityMatch,
    /// Based on load balancing
    LoadBalancing,
    /// Based on geographic routing
    Geographic,
    /// Based on explicit user preference
    UserPreference,
    /// Based on A/B testing rules
    ABTest(String),
    /// Based on custom rules
    CustomRule(String),
    /// Default fallback
    Default,
}

/// Alternative route in fallback chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeRoute {
    /// Provider name
    pub provider: String,
    /// Model name
    pub model: String,
    /// Priority in fallback chain (lower = higher priority)
    pub priority: u32,
    /// Why this is an alternative (not primary)
    pub reason: String,
}

/// Conditional branch consumed from routing rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedConditionalBranch {
    /// Branch identifier
    pub id: String,
    /// Branch name
    pub name: String,
    /// Condition that triggered this branch
    pub condition: BranchCondition,
    /// Target endpoint or handler
    pub target: String,
    /// Branch priority
    pub priority: u32,
    /// Whether branch is currently active
    pub active: bool,
}

/// Condition for branch evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BranchCondition {
    /// Match on model name pattern
    ModelMatch(String),
    /// Match on provider
    ProviderMatch(String),
    /// Match on request header
    HeaderMatch { name: String, pattern: String },
    /// Match on token count threshold
    TokenThreshold { min: Option<u32>, max: Option<u32> },
    /// Match on streaming requirement
    StreamingRequired(bool),
    /// Match on capability requirement
    CapabilityRequired(String),
    /// Percentage-based routing (A/B testing)
    PercentageSplit { percentage: f64, group: String },
    /// Time-based routing
    TimeWindow { start_hour: u8, end_hour: u8 },
    /// Custom condition expression
    Custom(String),
    /// Always match (default route)
    Always,
}

/// Routing rule set consumed from external source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedRoutingRules {
    /// Rule set version
    pub version: String,
    /// Rule set name
    pub name: String,
    /// Individual routing rules
    pub rules: Vec<RoutingRule>,
    /// Default route if no rules match
    pub default_route: DefaultRoute,
    /// Last update timestamp
    pub updated_at: u64,
}

/// Individual routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Conditions to match
    pub conditions: Vec<BranchCondition>,
    /// Action to take when matched
    pub action: RoutingAction,
    /// Rule priority (lower = higher priority)
    pub priority: u32,
    /// Whether rule is enabled
    pub enabled: bool,
}

/// Action to take when routing rule matches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAction {
    /// Route to specific provider/model
    RouteToModel { provider: String, model: String },
    /// Route to model pool (load balanced)
    RouteToPool { pool_name: String },
    /// Apply rate limiting
    RateLimit { requests_per_minute: u32 },
    /// Reject request
    Reject { reason: String },
    /// Transform request before routing
    Transform { transformation: String },
    /// Continue to next rule
    Continue,
}

/// Default route configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultRoute {
    /// Default provider
    pub provider: String,
    /// Default model
    pub model: String,
}

/// Request context for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingContext {
    /// Request path
    pub path: String,
    /// HTTP method
    pub method: String,
    /// Target model (from request)
    pub model: Option<String>,
    /// Target provider
    pub provider: Option<String>,
    /// Estimated input tokens
    pub input_tokens: Option<u32>,
    /// Whether streaming is requested
    pub streaming: bool,
    /// Request headers (filtered)
    pub headers: HashMap<String, String>,
    /// Client identifier
    pub client_id: Option<String>,
}

/// Consumer trait for Router integration
///
/// Implementors consume routing decisions and branching logic
/// without modifying any existing simulator APIs.
#[async_trait]
pub trait RouterConsumer: Send + Sync {
    /// Consume a routing decision for a request
    async fn consume_routing_decision(
        &self,
        context: &RoutingContext,
    ) -> Result<Option<ConsumedRoutingDecision>, AdapterError>;

    /// Consume conditional branches for a path
    async fn consume_conditional_branches(
        &self,
        path: &str,
    ) -> Result<Vec<ConsumedConditionalBranch>, AdapterError>;

    /// Consume current routing rules
    async fn consume_routing_rules(&self) -> Result<Option<ConsumedRoutingRules>, AdapterError>;

    /// Evaluate a specific rule against context
    async fn evaluate_rule(
        &self,
        rule_id: &str,
        context: &RoutingContext,
    ) -> Result<bool, AdapterError>;

    /// Check if the consumer is connected and healthy
    async fn health_check(&self) -> Result<bool, AdapterError>;
}

/// Error type for adapter operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum AdapterError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Rule not found: {0}")]
    RuleNotFound(String),

    #[error("Invalid rule configuration: {0}")]
    InvalidRule(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Upstream service error: {0}")]
    UpstreamError(String),
}

/// Default implementation for Router adapter
///
/// This adapter provides routing decision consumption that integrates
/// with Axum's Router without modifying the existing routing structure.
pub struct RouterAdapter {
    /// Cached routing rules
    rules_cache: Arc<parking_lot::RwLock<Option<ConsumedRoutingRules>>>,
    /// Cache TTL
    cache_ttl: Duration,
    /// Default provider
    default_provider: String,
    /// Default model
    default_model: String,
}

impl RouterAdapter {
    /// Create a new adapter with default settings
    pub fn new() -> Self {
        Self {
            rules_cache: Arc::new(parking_lot::RwLock::new(None)),
            cache_ttl: Duration::from_secs(60),
            default_provider: "openai".to_string(),
            default_model: "gpt-4".to_string(),
        }
    }

    /// Create adapter with custom defaults
    pub fn with_defaults(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            rules_cache: Arc::new(parking_lot::RwLock::new(None)),
            cache_ttl: Duration::from_secs(60),
            default_provider: provider.into(),
            default_model: model.into(),
        }
    }

    /// Create adapter with custom cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Evaluate a condition against context
    fn evaluate_condition(condition: &BranchCondition, context: &RoutingContext) -> bool {
        match condition {
            BranchCondition::ModelMatch(pattern) => {
                context.model.as_ref().map_or(false, |m| m.contains(pattern))
            }
            BranchCondition::ProviderMatch(provider) => {
                context.provider.as_ref().map_or(false, |p| p == provider)
            }
            BranchCondition::HeaderMatch { name, pattern } => {
                context.headers.get(name).map_or(false, |v| v.contains(pattern))
            }
            BranchCondition::TokenThreshold { min, max } => {
                let tokens = context.input_tokens.unwrap_or(0);
                let above_min = min.map_or(true, |m| tokens >= m);
                let below_max = max.map_or(true, |m| tokens <= m);
                above_min && below_max
            }
            BranchCondition::StreamingRequired(required) => context.streaming == *required,
            BranchCondition::CapabilityRequired(_) => true, // Simplified
            BranchCondition::PercentageSplit { percentage, .. } => {
                // Simplified: use hash of client_id for deterministic split
                let hash = context.client_id.as_ref()
                    .map(|id| {
                        let mut h: u64 = 0;
                        for b in id.bytes() {
                            h = h.wrapping_mul(31).wrapping_add(b as u64);
                        }
                        h
                    })
                    .unwrap_or(0);
                (hash % 100) < (*percentage * 100.0) as u64
            }
            BranchCondition::TimeWindow { start_hour, end_hour } => {
                let current_hour = (chrono::Utc::now().timestamp() / 3600 % 24) as u8;
                current_hour >= *start_hour && current_hour < *end_hour
            }
            BranchCondition::Custom(_) => true, // Custom conditions default to true
            BranchCondition::Always => true,
        }
    }
}

impl Default for RouterAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RouterConsumer for RouterAdapter {
    async fn consume_routing_decision(
        &self,
        context: &RoutingContext,
    ) -> Result<Option<ConsumedRoutingDecision>, AdapterError> {
        // Check if we have cached rules
        let rules = self.rules_cache.read().clone();

        if let Some(rules) = rules {
            // Evaluate rules in priority order
            let mut matched_rules: Vec<_> = rules.rules.iter()
                .filter(|r| r.enabled)
                .filter(|r| {
                    r.conditions.iter().all(|c| Self::evaluate_condition(c, context))
                })
                .collect();

            matched_rules.sort_by_key(|r| r.priority);

            if let Some(rule) = matched_rules.first() {
                if let RoutingAction::RouteToModel { provider, model } = &rule.action {
                    return Ok(Some(ConsumedRoutingDecision {
                        id: uuid::Uuid::new_v4().to_string(),
                        provider: provider.clone(),
                        model: model.clone(),
                        reason: RoutingReason::CustomRule(rule.name.clone()),
                        confidence: 1.0,
                        alternatives: Vec::new(),
                        timestamp: chrono::Utc::now().timestamp_millis() as u64,
                        decision_latency_ms: 0.1,
                    }));
                }
            }
        }

        // No external routing decision - simulator uses its own routing
        Ok(None)
    }

    async fn consume_conditional_branches(
        &self,
        _path: &str,
    ) -> Result<Vec<ConsumedConditionalBranch>, AdapterError> {
        // Return empty - simulator uses its own conditional logic
        Ok(Vec::new())
    }

    async fn consume_routing_rules(&self) -> Result<Option<ConsumedRoutingRules>, AdapterError> {
        // Return cached rules if available
        Ok(self.rules_cache.read().clone())
    }

    async fn evaluate_rule(
        &self,
        rule_id: &str,
        context: &RoutingContext,
    ) -> Result<bool, AdapterError> {
        let rules = self.rules_cache.read();

        if let Some(rules) = rules.as_ref() {
            if let Some(rule) = rules.rules.iter().find(|r| r.id == rule_id) {
                let all_match = rule.conditions.iter()
                    .all(|c| Self::evaluate_condition(c, context));
                return Ok(all_match);
            }
        }

        Err(AdapterError::RuleNotFound(rule_id.to_string()))
    }

    async fn health_check(&self) -> Result<bool, AdapterError> {
        // Adapter is always "healthy" as a thin consumption layer
        Ok(true)
    }
}

/// Utility for building routing context from HTTP request
impl RoutingContext {
    /// Create a new routing context
    pub fn new(path: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            method: method.into(),
            model: None,
            provider: None,
            input_tokens: None,
            streaming: false,
            headers: HashMap::new(),
            client_id: None,
        }
    }

    /// Set the target model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the target provider
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set estimated input tokens
    pub fn with_input_tokens(mut self, tokens: u32) -> Self {
        self.input_tokens = Some(tokens);
        self
    }

    /// Set streaming flag
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Add a header
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Set client ID
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = RouterAdapter::new();
        assert!(adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_routing_decision_returns_none() {
        let adapter = RouterAdapter::new();
        let context = RoutingContext::new("/v1/chat/completions", "POST")
            .with_model("gpt-4")
            .with_streaming(false);

        let result = adapter.consume_routing_decision(&context).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_condition_evaluation() {
        let context = RoutingContext::new("/v1/chat/completions", "POST")
            .with_model("gpt-4")
            .with_provider("openai")
            .with_input_tokens(500)
            .with_streaming(true);

        // Model match
        assert!(RouterAdapter::evaluate_condition(
            &BranchCondition::ModelMatch("gpt".to_string()),
            &context
        ));

        // Provider match
        assert!(RouterAdapter::evaluate_condition(
            &BranchCondition::ProviderMatch("openai".to_string()),
            &context
        ));

        // Token threshold
        assert!(RouterAdapter::evaluate_condition(
            &BranchCondition::TokenThreshold { min: Some(100), max: Some(1000) },
            &context
        ));

        // Streaming required
        assert!(RouterAdapter::evaluate_condition(
            &BranchCondition::StreamingRequired(true),
            &context
        ));

        // Always
        assert!(RouterAdapter::evaluate_condition(
            &BranchCondition::Always,
            &context
        ));
    }

    #[test]
    fn test_routing_context_builder() {
        let context = RoutingContext::new("/v1/chat/completions", "POST")
            .with_model("gpt-4")
            .with_provider("openai")
            .with_input_tokens(500)
            .with_streaming(true)
            .with_header("X-Request-ID", "req-123")
            .with_client_id("client-456");

        assert_eq!(context.path, "/v1/chat/completions");
        assert_eq!(context.method, "POST");
        assert_eq!(context.model, Some("gpt-4".to_string()));
        assert_eq!(context.provider, Some("openai".to_string()));
        assert_eq!(context.input_tokens, Some(500));
        assert!(context.streaming);
        assert_eq!(context.headers.get("X-Request-ID"), Some(&"req-123".to_string()));
        assert_eq!(context.client_id, Some("client-456".to_string()));
    }

    #[test]
    fn test_routing_reason_serialization() {
        let reason = RoutingReason::ABTest("experiment-1".to_string());
        let json = serde_json::to_string(&reason).unwrap();
        assert!(json.contains("ABTest"));
        assert!(json.contains("experiment-1"));
    }
}
