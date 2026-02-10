//! Phase 2B+ Adapter Modules
//!
//! Thin runtime consumption layers for integrating with LLM-Dev-Ops ecosystem.
//! These adapters consume data from upstream dependencies without modifying
//! any existing public APIs or simulation logic.
//!
//! ## Architecture
//!
//! Each adapter provides:
//! - A consumer trait defining the consumption interface
//! - Data transformation utilities
//! - Integration hooks for the simulator
//! - Infrastructure utilities for caching and retry logic
//!
//! ## Adapters
//!
//! - `latency_lens`: Consumes latency profiles, throughput data, cold-start distributions
//! - `observatory`: Consumes telemetry, trace spans, runtime state transitions
//! - `router`: Consumes routing decisions and conditional branching logic
//! - `memory_graph`: Consumes lineage and graph-based context states
//! - `ruvvector`: HTTP client for external ruvvector-service (/query, /simulate endpoints)
//! - `intelligence`: Phase 7 Intelligence & Expansion (Layer 2) signal emission
//!
//! ## Infrastructure Integration
//!
//! All adapters can optionally use the `infra` module for:
//! - Caching consumed data with configurable TTL
//! - Retry logic with exponential backoff for upstream calls
//! - Shared infrastructure context across adapters

pub mod latency_lens;
pub mod observatory;
pub mod router;
pub mod memory_graph;
pub mod ruvvector;
pub mod intelligence;

// Re-export primary consumer traits
pub use latency_lens::LatencyLensConsumer;
pub use observatory::ObservatoryConsumer;
pub use router::RouterConsumer;
pub use memory_graph::MemoryGraphConsumer;
pub use ruvvector::{RuvVectorConsumer, RuvVectorAdapter, RuvVectorConfig, RuvVectorError, OptionalRuvVectorAdapter};

// Phase 7: Intelligence & Expansion (Layer 2)
pub use intelligence::{
    IntelligenceConsumer, IntelligenceAdapter, IntelligenceConfig, IntelligenceError,
    IntelligenceStats, OptionalIntelligenceAdapter,
    SignalType, DecisionSignal, SignalPayload,
    HypothesisPayload, SimulationOutcomePayload, ConfidenceDeltaPayload,
    ReasoningContext, SimulationScenario, ConfidenceAssessment,
    MAX_TOKENS, MAX_LATENCY_MS,
};

// FEU (Foundational Execution Unit) types
pub use observatory::FeuSpanKind;
pub use intelligence::FeuIntelligenceConsumer;
pub use memory_graph::FeuMemoryGraphConsumer;
pub use crate::telemetry::tracing_ext::{
    FeuSpanCollector, FeuValidationError, ExecutionTrace, SpanArtifact, FEU_ROOT_PARENT,
};

use std::sync::Arc;
use parking_lot::RwLock;

use crate::adapters::observatory::SpanStatus;
use crate::infra::{SharedInfraContext, shared_infra_context};

/// Unified adapter registry for managing all Phase 2B+ integrations
pub struct AdapterRegistry {
    latency_lens: Option<Arc<dyn LatencyLensConsumer>>,
    observatory: Option<Arc<dyn ObservatoryConsumer>>,
    router: Option<Arc<dyn RouterConsumer>>,
    memory_graph: Option<Arc<dyn MemoryGraphConsumer>>,
    ruvvector: Option<Arc<dyn RuvVectorConsumer>>,
    /// Phase 7: Intelligence & Expansion (Layer 2)
    intelligence: Option<Arc<dyn IntelligenceConsumer>>,
    /// Shared infrastructure context for caching and retry
    infra: SharedInfraContext,
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self {
            latency_lens: None,
            observatory: None,
            router: None,
            memory_graph: None,
            ruvvector: None,
            intelligence: None,
            infra: shared_infra_context(),
        }
    }
}

impl AdapterRegistry {
    /// Create a new empty adapter registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a latency lens consumer
    pub fn with_latency_lens(mut self, consumer: Arc<dyn LatencyLensConsumer>) -> Self {
        self.latency_lens = Some(consumer);
        self
    }

    /// Register an observatory consumer
    pub fn with_observatory(mut self, consumer: Arc<dyn ObservatoryConsumer>) -> Self {
        self.observatory = Some(consumer);
        self
    }

    /// Register a router consumer
    pub fn with_router(mut self, consumer: Arc<dyn RouterConsumer>) -> Self {
        self.router = Some(consumer);
        self
    }

    /// Register a memory graph consumer
    pub fn with_memory_graph(mut self, consumer: Arc<dyn MemoryGraphConsumer>) -> Self {
        self.memory_graph = Some(consumer);
        self
    }

    /// Register a RuvVector consumer
    pub fn with_ruvvector(mut self, consumer: Arc<dyn RuvVectorConsumer>) -> Self {
        self.ruvvector = Some(consumer);
        self
    }

    /// Register an intelligence consumer (Phase 7)
    pub fn with_intelligence(mut self, consumer: Arc<dyn IntelligenceConsumer>) -> Self {
        self.intelligence = Some(consumer);
        self
    }

    /// Get the latency lens consumer if registered
    pub fn latency_lens(&self) -> Option<&Arc<dyn LatencyLensConsumer>> {
        self.latency_lens.as_ref()
    }

    /// Get the observatory consumer if registered
    pub fn observatory(&self) -> Option<&Arc<dyn ObservatoryConsumer>> {
        self.observatory.as_ref()
    }

    /// Get the router consumer if registered
    pub fn router(&self) -> Option<&Arc<dyn RouterConsumer>> {
        self.router.as_ref()
    }

    /// Get the memory graph consumer if registered
    pub fn memory_graph(&self) -> Option<&Arc<dyn MemoryGraphConsumer>> {
        self.memory_graph.as_ref()
    }

    /// Get the RuvVector consumer if registered
    pub fn ruvvector(&self) -> Option<&Arc<dyn RuvVectorConsumer>> {
        self.ruvvector.as_ref()
    }

    /// Get the intelligence consumer if registered (Phase 7)
    pub fn intelligence(&self) -> Option<&Arc<dyn IntelligenceConsumer>> {
        self.intelligence.as_ref()
    }

    /// Check if any adapters are registered
    pub fn has_adapters(&self) -> bool {
        self.latency_lens.is_some()
            || self.observatory.is_some()
            || self.router.is_some()
            || self.memory_graph.is_some()
            || self.ruvvector.is_some()
            || self.intelligence.is_some()
    }

    /// Get the shared infrastructure context
    pub fn infra(&self) -> &SharedInfraContext {
        &self.infra
    }

    /// Set a custom infrastructure context
    pub fn with_infra(mut self, infra: SharedInfraContext) -> Self {
        self.infra = infra;
        self
    }

    /// Get cache statistics from the infrastructure context
    pub fn cache_stats(&self) -> crate::infra::CacheStats {
        self.infra.read().cache_stats()
    }

    /// Clear the infrastructure cache
    pub fn clear_cache(&self) {
        self.infra.read().clear_cache();
    }

    /// Execute all registered adapters with FEU span tracking.
    /// Creates a repo-level span and one agent span per registered adapter.
    /// Returns the complete ExecutionTrace with all spans and artifacts.
    pub fn execute_traced(
        &self,
        request_id: &str,
    ) -> Result<ExecutionTrace, FeuValidationError> {
        let mut collector = FeuSpanCollector::new(Some(request_id.to_string()));

        if self.observatory.is_some() {
            let span_id = collector.begin_agent_span("observatory")?;
            collector.end_agent_span(&span_id, SpanStatus::Ok)?;
        }

        if self.intelligence.is_some() {
            let span_id = collector.begin_agent_span("intelligence")?;
            collector.end_agent_span(&span_id, SpanStatus::Ok)?;
        }

        if self.memory_graph.is_some() {
            let span_id = collector.begin_agent_span("memory_graph")?;
            collector.end_agent_span(&span_id, SpanStatus::Ok)?;
        }

        if self.latency_lens.is_some() {
            let span_id = collector.begin_agent_span("latency_lens")?;
            collector.end_agent_span(&span_id, SpanStatus::Ok)?;
        }

        if self.router.is_some() {
            let span_id = collector.begin_agent_span("router")?;
            collector.end_agent_span(&span_id, SpanStatus::Ok)?;
        }

        collector.finalize()
    }
}

/// Thread-safe adapter registry wrapper
pub type SharedAdapterRegistry = Arc<RwLock<AdapterRegistry>>;

/// Create a new shared adapter registry
pub fn shared_registry() -> SharedAdapterRegistry {
    Arc::new(RwLock::new(AdapterRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AdapterRegistry::new();
        assert!(!registry.has_adapters());
    }

    #[test]
    fn test_shared_registry() {
        let registry = shared_registry();
        let guard = registry.read();
        assert!(!guard.has_adapters());
    }

    #[test]
    fn test_execute_traced_empty_registry() {
        let registry = AdapterRegistry::new();
        let trace = registry.execute_traced("req-empty").unwrap();

        // Only repo span, no agent spans
        assert_eq!(trace.trace_id, "req-empty");
        assert_eq!(trace.agent_spans.len(), 0);
        assert_eq!(trace.status, SpanStatus::Ok);
        assert_eq!(trace.repo_span.name, "repo_execution");
        assert_eq!(trace.repo_span.parent_span_id, Some("ROOT".to_string()));
    }

    #[test]
    fn test_execute_traced_with_adapters() {
        use crate::adapters::observatory::ObservatoryAdapter;
        use crate::adapters::memory_graph::MemoryGraphAdapter;

        let registry = AdapterRegistry::new()
            .with_observatory(Arc::new(ObservatoryAdapter::new()))
            .with_memory_graph(Arc::new(MemoryGraphAdapter::new()));

        let trace = registry.execute_traced("req-adapters").unwrap();

        assert_eq!(trace.agent_spans.len(), 2);

        let names: Vec<&str> = trace.agent_spans.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"agent:observatory"));
        assert!(names.contains(&"agent:memory_graph"));

        // All agent spans have correct parent
        for span in &trace.agent_spans {
            assert_eq!(span.parent_span_id, Some(trace.repo_span.span_id.clone()));
        }
    }
}
