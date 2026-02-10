//! LLM-Memory-Graph Adapter
//!
//! Thin runtime consumer for lineage tracking and graph-based context states
//! from the LLM-Memory-Graph service.
//!
//! This adapter provides:
//! - Consumption of prompt lineage data
//! - Graph-based context state retrieval
//! - Session and conversation tracking
//! - Agent and tool invocation lineage
//!
//! ## Usage
//!
//! ```rust,ignore
//! use llm_simulator::adapters::memory_graph::{MemoryGraphConsumer, MemoryGraphAdapter};
//!
//! let adapter = MemoryGraphAdapter::new();
//! let lineage = adapter.consume_prompt_lineage("prompt-123").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Prompt lineage data consumed from LLM-Memory-Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedPromptLineage {
    /// Prompt node identifier
    pub id: String,
    /// Session this prompt belongs to
    pub session_id: String,
    /// Parent prompt ID (if derived)
    pub parent_id: Option<String>,
    /// Prompt text content
    pub content: String,
    /// Prompt metadata
    pub metadata: PromptMetadata,
    /// Child prompts derived from this one
    pub children: Vec<String>,
    /// Associated response ID
    pub response_id: Option<String>,
    /// Creation timestamp
    pub created_at: u64,
}

/// Metadata for a prompt node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptMetadata {
    /// Source of the prompt (user, system, template, etc.)
    pub source: String,
    /// Template ID if derived from template
    pub template_id: Option<String>,
    /// Variables used in template expansion
    pub variables: HashMap<String, String>,
    /// Token count
    pub token_count: Option<u32>,
    /// Custom tags
    pub tags: Vec<String>,
}

/// Response lineage data consumed from LLM-Memory-Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedResponseLineage {
    /// Response node identifier
    pub id: String,
    /// Associated prompt ID
    pub prompt_id: String,
    /// Session ID
    pub session_id: String,
    /// Response content
    pub content: String,
    /// Response metadata
    pub metadata: ResponseMetadata,
    /// Model that generated this response
    pub model: String,
    /// Provider
    pub provider: String,
    /// Creation timestamp
    pub created_at: u64,
}

/// Metadata for a response node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Token usage
    pub token_usage: Option<TokenUsage>,
    /// Latency in milliseconds
    pub latency_ms: Option<f64>,
    /// Cost in USD
    pub cost_usd: Option<f64>,
    /// Tool calls made
    pub tool_calls: Vec<String>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Session/conversation context consumed from LLM-Memory-Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedSessionContext {
    /// Session identifier
    pub id: String,
    /// Session name/title
    pub name: Option<String>,
    /// Session state
    pub state: SessionState,
    /// Root prompt IDs in this session
    pub root_prompts: Vec<String>,
    /// Total message count
    pub message_count: u32,
    /// Total token count
    pub total_tokens: u64,
    /// Session metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_activity_at: u64,
}

/// Session state enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionState {
    /// Session is active
    Active,
    /// Session is paused
    Paused,
    /// Session is completed
    Completed,
    /// Session is archived
    Archived,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::Active
    }
}

/// Graph context state consumed from LLM-Memory-Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedGraphContext {
    /// Context identifier
    pub id: String,
    /// Session ID
    pub session_id: String,
    /// Nodes in this context window
    pub nodes: Vec<ContextNode>,
    /// Edges connecting nodes
    pub edges: Vec<ContextEdge>,
    /// Context summary
    pub summary: Option<String>,
    /// Total context tokens
    pub token_count: u32,
    /// Context retrieval timestamp
    pub retrieved_at: u64,
}

/// Node in the context graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextNode {
    /// Node identifier
    pub id: String,
    /// Node type
    pub node_type: NodeType,
    /// Node content (truncated)
    pub content: String,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f64,
    /// Position in context
    pub position: u32,
}

/// Type of context node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeType {
    /// User prompt
    Prompt,
    /// Assistant response
    Response,
    /// System message
    System,
    /// Tool invocation
    Tool,
    /// Agent action
    Agent,
    /// Retrieved document
    Document,
    /// Custom node type
    Custom(String),
}

impl Default for NodeType {
    fn default() -> Self {
        Self::Prompt
    }
}

/// Edge in the context graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge weight/strength
    pub weight: f64,
}

/// Type of context edge
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EdgeType {
    /// Prompt to response
    PromptResponse,
    /// Response to follow-up prompt
    FollowUp,
    /// Derivation (template expansion, etc.)
    Derived,
    /// Reference (citation, link)
    References,
    /// Tool invocation
    Invokes,
    /// Context transfer
    ContextTransfer,
    /// Custom edge type
    Custom(String),
}

impl Default for EdgeType {
    fn default() -> Self {
        Self::PromptResponse
    }
}

/// Agent lineage consumed from LLM-Memory-Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedAgentLineage {
    /// Agent identifier
    pub agent_id: String,
    /// Agent name
    pub name: String,
    /// Agent configuration
    pub config: AgentConfig,
    /// Agent status
    pub status: AgentStatus,
    /// Actions taken by this agent
    pub actions: Vec<AgentAction>,
    /// Tool invocations
    pub tool_invocations: Vec<ToolInvocation>,
    /// Metrics
    pub metrics: AgentMetrics,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentConfig {
    /// Agent type
    pub agent_type: String,
    /// Model used
    pub model: String,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Available tools
    pub tools: Vec<String>,
    /// Max iterations
    pub max_iterations: Option<u32>,
}

/// Agent execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentStatus {
    Idle,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// Action taken by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    /// Action identifier
    pub id: String,
    /// Action type
    pub action_type: String,
    /// Action input
    pub input: String,
    /// Action output
    pub output: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Tool invocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    /// Invocation identifier
    pub id: String,
    /// Tool name
    pub tool_name: String,
    /// Tool input
    pub input: String,
    /// Tool output
    pub output: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Success status
    pub success: bool,
    /// Timestamp
    pub timestamp: u64,
}

/// Agent execution metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentMetrics {
    /// Total iterations
    pub iterations: u32,
    /// Total tool calls
    pub tool_calls: u32,
    /// Total tokens used
    pub total_tokens: u64,
    /// Total execution time in milliseconds
    pub execution_time_ms: f64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
}

/// Consumer trait for LLM-Memory-Graph integration
///
/// Implementors consume lineage and context data from the Memory-Graph service
/// without modifying any existing simulator APIs.
#[async_trait]
pub trait MemoryGraphConsumer: Send + Sync {
    /// Consume prompt lineage data
    async fn consume_prompt_lineage(
        &self,
        prompt_id: &str,
    ) -> Result<Option<ConsumedPromptLineage>, AdapterError>;

    /// Consume response lineage data
    async fn consume_response_lineage(
        &self,
        response_id: &str,
    ) -> Result<Option<ConsumedResponseLineage>, AdapterError>;

    /// Consume session context
    async fn consume_session_context(
        &self,
        session_id: &str,
    ) -> Result<Option<ConsumedSessionContext>, AdapterError>;

    /// Consume graph context for a session
    async fn consume_graph_context(
        &self,
        session_id: &str,
        max_tokens: Option<u32>,
    ) -> Result<Option<ConsumedGraphContext>, AdapterError>;

    /// Consume agent lineage
    async fn consume_agent_lineage(
        &self,
        agent_id: &str,
    ) -> Result<Option<ConsumedAgentLineage>, AdapterError>;

    /// List sessions
    async fn list_sessions(
        &self,
        limit: usize,
    ) -> Result<Vec<ConsumedSessionContext>, AdapterError>;

    /// Check if the consumer is connected and healthy
    async fn health_check(&self) -> Result<bool, AdapterError>;
}

/// Error type for adapter operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum AdapterError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Node not found: {0}")]
    NotFound(String),

    #[error("Invalid graph structure: {0}")]
    InvalidStructure(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Upstream service error: {0}")]
    UpstreamError(String),
}

/// Default implementation that bridges to llm-memory-graph types
pub struct MemoryGraphAdapter {
    /// Cached sessions
    session_cache: Arc<parking_lot::RwLock<HashMap<String, ConsumedSessionContext>>>,
    /// Cache TTL
    cache_ttl: Duration,
}

impl MemoryGraphAdapter {
    /// Create a new adapter with default settings
    pub fn new() -> Self {
        Self {
            session_cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            cache_ttl: Duration::from_secs(300), // 5 minute cache
        }
    }

    /// Create adapter with custom cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Convert from llm-memory-graph types to consumed types
    ///
    /// This bridge function is prepared for when upstream compilation is fixed.
    /// Currently uses placeholder implementation.
    #[allow(dead_code)]
    fn convert_from_upstream_node(
        id: &str,
        session_id: &str,
        content: &str,
        parent_id: Option<&str>,
    ) -> ConsumedPromptLineage {
        // Bridge conversion - ready for upstream types when available
        ConsumedPromptLineage {
            id: id.to_string(),
            session_id: session_id.to_string(),
            parent_id: parent_id.map(|s| s.to_string()),
            content: content.to_string(),
            metadata: PromptMetadata::default(),
            children: Vec::new(),
            response_id: None,
            created_at: chrono::Utc::now().timestamp_millis() as u64,
        }
    }
}

impl Default for MemoryGraphAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryGraphConsumer for MemoryGraphAdapter {
    async fn consume_prompt_lineage(
        &self,
        _prompt_id: &str,
    ) -> Result<Option<ConsumedPromptLineage>, AdapterError> {
        // In full implementation, query llm-memory-graph
        // Return None - simulator generates its own prompts
        Ok(None)
    }

    async fn consume_response_lineage(
        &self,
        _response_id: &str,
    ) -> Result<Option<ConsumedResponseLineage>, AdapterError> {
        // Return None - simulator generates its own responses
        Ok(None)
    }

    async fn consume_session_context(
        &self,
        session_id: &str,
    ) -> Result<Option<ConsumedSessionContext>, AdapterError> {
        // Check cache first
        if let Some(cached) = self.session_cache.read().get(session_id) {
            return Ok(Some(cached.clone()));
        }

        // Return None - no external session data
        Ok(None)
    }

    async fn consume_graph_context(
        &self,
        _session_id: &str,
        _max_tokens: Option<u32>,
    ) -> Result<Option<ConsumedGraphContext>, AdapterError> {
        // Return None - simulator doesn't use external context
        Ok(None)
    }

    async fn consume_agent_lineage(
        &self,
        _agent_id: &str,
    ) -> Result<Option<ConsumedAgentLineage>, AdapterError> {
        // Return None - simulator generates its own agent data
        Ok(None)
    }

    async fn list_sessions(
        &self,
        limit: usize,
    ) -> Result<Vec<ConsumedSessionContext>, AdapterError> {
        // Return cached sessions up to limit
        let sessions: Vec<_> = self.session_cache.read()
            .values()
            .take(limit)
            .cloned()
            .collect();
        Ok(sessions)
    }

    async fn health_check(&self) -> Result<bool, AdapterError> {
        // Adapter is always "healthy" as a thin consumption layer
        Ok(true)
    }
}

/// Utility implementations for consumed data
impl ConsumedPromptLineage {
    /// Check if this is a root prompt (no parent)
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Check if this prompt has children
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Get the depth in the lineage tree (0 for root)
    pub fn depth(&self) -> u32 {
        // Simplified - would need full tree traversal
        if self.parent_id.is_none() { 0 } else { 1 }
    }
}

impl ConsumedGraphContext {
    /// Get nodes of a specific type
    pub fn nodes_of_type(&self, node_type: &NodeType) -> Vec<&ContextNode> {
        self.nodes.iter()
            .filter(|n| &n.node_type == node_type)
            .collect()
    }

    /// Get edges of a specific type
    pub fn edges_of_type(&self, edge_type: &EdgeType) -> Vec<&ContextEdge> {
        self.edges.iter()
            .filter(|e| &e.edge_type == edge_type)
            .collect()
    }

    /// Get nodes connected to a given node
    pub fn connected_nodes(&self, node_id: &str) -> Vec<&str> {
        self.edges.iter()
            .filter(|e| e.source == node_id || e.target == node_id)
            .map(|e| {
                if e.source == node_id { e.target.as_str() } else { e.source.as_str() }
            })
            .collect()
    }
}

// ============================================================================
// FEU Extension Trait
// ============================================================================

use crate::adapters::observatory::SpanStatus;
use crate::telemetry::tracing_ext::{FeuSpanCollector, SpanArtifact};

/// Extension trait for FEU-aware memory graph operations.
/// Wraps MemoryGraphConsumer methods with span lifecycle management.
#[async_trait]
pub trait FeuMemoryGraphConsumer: MemoryGraphConsumer {
    /// Consume agent lineage with FEU span tracking.
    async fn consume_agent_lineage_traced(
        &self,
        agent_id: &str,
        collector: &mut FeuSpanCollector,
    ) -> Result<Option<ConsumedAgentLineage>, AdapterError>;
}

#[async_trait]
impl FeuMemoryGraphConsumer for MemoryGraphAdapter {
    async fn consume_agent_lineage_traced(
        &self,
        agent_id: &str,
        collector: &mut FeuSpanCollector,
    ) -> Result<Option<ConsumedAgentLineage>, AdapterError> {
        let span_id = collector.begin_agent_span("memory_graph")
            .map_err(|e: crate::telemetry::tracing_ext::FeuValidationError| AdapterError::UpstreamError(e.to_string()))?;

        match self.consume_agent_lineage(agent_id).await {
            Ok(Some(lineage)) => {
                let artifact = SpanArtifact {
                    kind: "agent_lineage".to_string(),
                    payload: serde_json::to_value(&lineage).unwrap_or(serde_json::Value::Null),
                    created_at: chrono::Utc::now().timestamp_millis() as u64,
                };
                let _ = collector.attach_artifact(&span_id, artifact);

                // Map AgentStatus to SpanStatus
                let span_status = match lineage.status {
                    AgentStatus::Completed => SpanStatus::Ok,
                    AgentStatus::Failed => SpanStatus::Failed,
                    AgentStatus::Cancelled => SpanStatus::Error,
                    _ => SpanStatus::Unset,
                };
                let _ = collector.end_agent_span(&span_id, span_status);
                Ok(Some(lineage))
            }
            Ok(None) => {
                let _ = collector.end_agent_span(&span_id, SpanStatus::Ok);
                Ok(None)
            }
            Err(e) => {
                let _ = collector.fail_agent_span(&span_id, &e.to_string());
                Err(e)
            }
        }
    }
}

impl ConsumedAgentLineage {
    /// Check if agent completed successfully
    pub fn is_successful(&self) -> bool {
        self.status == AgentStatus::Completed
    }

    /// Get total actions count
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Get successful tool invocations
    pub fn successful_tools(&self) -> Vec<&ToolInvocation> {
        self.tool_invocations.iter()
            .filter(|t| t.success)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = MemoryGraphAdapter::new();
        assert!(adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_consume_prompt_returns_none() {
        let adapter = MemoryGraphAdapter::new();
        let result = adapter.consume_prompt_lineage("prompt-123").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_consume_session_returns_none() {
        let adapter = MemoryGraphAdapter::new();
        let result = adapter.consume_session_context("session-123").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list_sessions_empty() {
        let adapter = MemoryGraphAdapter::new();
        let result = adapter.list_sessions(10).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_prompt_lineage_utilities() {
        let lineage = ConsumedPromptLineage {
            id: "prompt-1".to_string(),
            session_id: "session-1".to_string(),
            parent_id: None,
            content: "Hello".to_string(),
            metadata: PromptMetadata::default(),
            children: vec!["prompt-2".to_string()],
            response_id: Some("response-1".to_string()),
            created_at: 0,
        };

        assert!(lineage.is_root());
        assert!(lineage.has_children());
        assert_eq!(lineage.depth(), 0);
    }

    #[test]
    fn test_graph_context_utilities() {
        let context = ConsumedGraphContext {
            id: "ctx-1".to_string(),
            session_id: "session-1".to_string(),
            nodes: vec![
                ContextNode {
                    id: "node-1".to_string(),
                    node_type: NodeType::Prompt,
                    content: "Hello".to_string(),
                    relevance: 1.0,
                    position: 0,
                },
                ContextNode {
                    id: "node-2".to_string(),
                    node_type: NodeType::Response,
                    content: "Hi there".to_string(),
                    relevance: 0.9,
                    position: 1,
                },
            ],
            edges: vec![
                ContextEdge {
                    source: "node-1".to_string(),
                    target: "node-2".to_string(),
                    edge_type: EdgeType::PromptResponse,
                    weight: 1.0,
                },
            ],
            summary: None,
            token_count: 10,
            retrieved_at: 0,
        };

        assert_eq!(context.nodes_of_type(&NodeType::Prompt).len(), 1);
        assert_eq!(context.nodes_of_type(&NodeType::Response).len(), 1);
        assert_eq!(context.edges_of_type(&EdgeType::PromptResponse).len(), 1);
        assert_eq!(context.connected_nodes("node-1").len(), 1);
    }

    #[test]
    fn test_agent_lineage_utilities() {
        let lineage = ConsumedAgentLineage {
            agent_id: "agent-1".to_string(),
            name: "Test Agent".to_string(),
            config: AgentConfig::default(),
            status: AgentStatus::Completed,
            actions: vec![
                AgentAction {
                    id: "action-1".to_string(),
                    action_type: "search".to_string(),
                    input: "query".to_string(),
                    output: Some("results".to_string()),
                    timestamp: 0,
                },
            ],
            tool_invocations: vec![
                ToolInvocation {
                    id: "tool-1".to_string(),
                    tool_name: "search".to_string(),
                    input: "{}".to_string(),
                    output: Some("{}".to_string()),
                    execution_time_ms: 100.0,
                    success: true,
                    timestamp: 0,
                },
            ],
            metrics: AgentMetrics::default(),
        };

        assert!(lineage.is_successful());
        assert_eq!(lineage.action_count(), 1);
        assert_eq!(lineage.successful_tools().len(), 1);
    }

    #[test]
    fn test_node_type_default() {
        let node_type = NodeType::default();
        assert_eq!(node_type, NodeType::Prompt);
    }

    #[test]
    fn test_edge_type_default() {
        let edge_type = EdgeType::default();
        assert_eq!(edge_type, EdgeType::PromptResponse);
    }

    #[test]
    fn test_session_state_default() {
        let state = SessionState::default();
        assert_eq!(state, SessionState::Active);
    }

    #[test]
    fn test_lineage_as_artifact_payload() {
        let lineage = ConsumedAgentLineage {
            agent_id: "agent-feu".to_string(),
            name: "FEU Agent".to_string(),
            config: AgentConfig::default(),
            status: AgentStatus::Completed,
            actions: vec![],
            tool_invocations: vec![],
            metrics: AgentMetrics::default(),
        };

        let artifact = SpanArtifact {
            kind: "agent_lineage".to_string(),
            payload: serde_json::to_value(&lineage).unwrap(),
            created_at: 0,
        };

        assert_eq!(artifact.kind, "agent_lineage");
        assert!(artifact.payload.get("agent_id").is_some());
    }

    #[test]
    fn test_feu_memory_graph_collector_integration() {
        let mut collector = FeuSpanCollector::new(None);
        let span_id = collector.begin_agent_span("memory_graph").unwrap();
        collector.end_agent_span(&span_id, SpanStatus::Ok).unwrap();

        let trace = collector.finalize().unwrap();
        assert_eq!(trace.agent_spans.len(), 1);
        assert_eq!(trace.agent_spans[0].name, "agent:memory_graph");
    }
}
