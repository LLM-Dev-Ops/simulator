//! # LLM-Simulator
//!
//! Enterprise-grade offline LLM API simulator for testing and development.
//!
//! LLM-Simulator provides a drop-in replacement for production LLM APIs,
//! enabling cost-effective, deterministic, and comprehensive testing of
//! LLM-powered applications.
//!
//! ## Features
//!
//! - **Multi-Provider Support**: OpenAI, Anthropic, Google, Azure compatible APIs
//! - **Realistic Latency Simulation**: Statistical models for TTFT and ITL
//! - **Error Injection**: Chaos engineering for resilience testing
//! - **Deterministic Execution**: Reproducible tests with seed-based RNG
//! - **OpenTelemetry Integration**: Full observability support
//! - **High Performance**: 10,000+ RPS with <5ms overhead
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use llm_simulator::{SimulatorConfig, SimulationEngine, run_server};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = SimulatorConfig::default();
//!     run_server(config).await
//! }
//! ```

pub mod adapters;
pub mod benchmarks;
pub mod cli;
pub mod config;
pub mod engine;
pub mod error;
pub mod infra;
pub mod latency;
pub mod providers;
pub mod sdk;
pub mod security;
pub mod server;
pub mod telemetry;
pub mod types;

pub use config::SimulatorConfig;
pub use engine::SimulationEngine;
pub use error::{SimulationError, SimulatorResult};
pub use infra::{Cache, CacheConfig, InfraContext, RetryPolicy, RetryConfig};
pub use server::run_server;

// Re-export RuvVector adapter types for easy integration
pub use adapters::ruvvector::{
    RuvVectorAdapter, RuvVectorConfig, RuvVectorError, RuvVectorConsumer,
    OptionalRuvVectorAdapter, QueryRequest, QueryResponse, QueryResult,
    SimulateRequest, SimulateResponse, RUVVECTOR_SERVICE_URL_ENV,
};

// Re-export FEU (Foundational Execution Unit) types
pub use telemetry::{
    FeuSpanCollector, FeuValidationError, ExecutionTrace, SpanArtifact, FEU_ROOT_PARENT,
};
pub use adapters::observatory::FeuSpanKind;

// Re-export Phase 7 Intelligence & Expansion (Layer 2) types
pub use adapters::intelligence::{
    IntelligenceAdapter, IntelligenceConfig, IntelligenceError, IntelligenceConsumer,
    IntelligenceStats, OptionalIntelligenceAdapter,
    SignalType, DecisionSignal, SignalPayload,
    HypothesisPayload, SimulationOutcomePayload, ConfidenceDeltaPayload,
    ReasoningContext, SimulationScenario, ConfidenceAssessment,
    MAX_TOKENS, MAX_LATENCY_MS,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default server port
pub const DEFAULT_PORT: u16 = 8080;

/// Default maximum concurrent requests
pub const DEFAULT_MAX_CONCURRENT: usize = 10_000;
