//! Phase 2B Adapter Modules
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
//!
//! ## Adapters
//!
//! - `latency_lens`: Consumes latency profiles, throughput data, cold-start distributions
//! - `observatory`: Consumes telemetry, trace spans, runtime state transitions
//! - `router`: Consumes routing decisions and conditional branching logic
//! - `memory_graph`: Consumes lineage and graph-based context states

pub mod latency_lens;
pub mod observatory;
pub mod router;
pub mod memory_graph;

// Re-export primary consumer traits
pub use latency_lens::LatencyLensConsumer;
pub use observatory::ObservatoryConsumer;
pub use router::RouterConsumer;
pub use memory_graph::MemoryGraphConsumer;

use std::sync::Arc;
use parking_lot::RwLock;

/// Unified adapter registry for managing all Phase 2B integrations
#[derive(Default)]
pub struct AdapterRegistry {
    latency_lens: Option<Arc<dyn LatencyLensConsumer>>,
    observatory: Option<Arc<dyn ObservatoryConsumer>>,
    router: Option<Arc<dyn RouterConsumer>>,
    memory_graph: Option<Arc<dyn MemoryGraphConsumer>>,
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

    /// Check if any adapters are registered
    pub fn has_adapters(&self) -> bool {
        self.latency_lens.is_some()
            || self.observatory.is_some()
            || self.router.is_some()
            || self.memory_graph.is_some()
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
}
