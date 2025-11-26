# LLM-Simulator: Completion Specification

> **Document Type:** SPARC Phase 5 - Completion
> **Version:** 1.0.0
> **Status:** Production-Ready
> **Date:** 2025-11-26
> **Classification:** LLM DevOps Platform - Core Testing Module

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Implementation Guidelines](#2-implementation-guidelines)
3. [Coding Standards](#3-coding-standards)
4. [API Reference](#4-api-reference)
5. [SDK Integration Guide](#5-sdk-integration-guide)
6. [Deployment Procedures](#6-deployment-procedures)
7. [Operational Runbooks](#7-operational-runbooks)
8. [Monitoring and Alerting](#8-monitoring-and-alerting)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Maintenance Procedures](#10-maintenance-procedures)
11. [Release Management](#11-release-management)
12. [Production Readiness Checklist](#12-production-readiness-checklist)
13. [Support and Escalation](#13-support-and-escalation)
14. [Training and Onboarding](#14-training-and-onboarding)

---

## 1. Executive Summary

### 1.1 Purpose

This Completion document represents SPARC Phase 5, the final phase of the LLM-Simulator specification. It provides comprehensive implementation guidance, deployment procedures, operational runbooks, and production readiness criteria required to successfully build, deploy, and operate LLM-Simulator in enterprise environments.

### 1.2 Document Scope

| Area | Coverage |
|------|----------|
| Implementation | Coding standards, patterns, best practices |
| API | Complete endpoint reference, schemas, examples |
| Deployment | Docker, Kubernetes, Helm, CI/CD |
| Operations | Runbooks, monitoring, alerting, maintenance |
| Support | Troubleshooting, escalation, training |

### 1.3 Prerequisites

Before using this document, ensure completion of:
- [x] SPARC Phase 1: Specification (`LLM-Simulator-Specification.md`)
- [x] SPARC Phase 2: Pseudocode (`LLM-Simulator-Pseudocode.md`)
- [x] SPARC Phase 3: Architecture (`LLM-Simulator-Architecture.md`)
- [x] SPARC Phase 4: Refinement (`LLM-Simulator-Refinement.md`)

### 1.4 Quick Reference

| Task | Section | Time |
|------|---------|------|
| Start development | [Section 2](#2-implementation-guidelines) | 5 min |
| Understand API | [Section 4](#4-api-reference) | 10 min |
| Deploy to Kubernetes | [Section 6.3](#63-kubernetes-deployment) | 15 min |
| Configure monitoring | [Section 8](#8-monitoring-and-alerting) | 20 min |
| Troubleshoot issues | [Section 9](#9-troubleshooting-guide) | As needed |

---

## 2. Implementation Guidelines

### 2.1 Development Environment Setup

#### 2.1.1 Required Tools

```bash
# Rust toolchain (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup component add rustfmt clippy

# Additional tools
cargo install cargo-watch cargo-audit cargo-deny cargo-tarpaulin

# Optional: IDE setup
# VSCode with rust-analyzer extension
# IntelliJ with Rust plugin
```

#### 2.1.2 Repository Structure

```
llm-simulator/
├── Cargo.toml                 # Workspace manifest
├── Cargo.lock                 # Dependency lock
├── rust-toolchain.toml        # Rust version pin
├── .cargo/
│   └── config.toml            # Cargo configuration
├── src/
│   ├── main.rs                # Binary entry point
│   ├── lib.rs                 # Library root
│   └── [modules]/             # Source modules
├── tests/
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
├── benches/                   # Benchmarks
├── config/
│   ├── default.yaml           # Default configuration
│   └── profiles/              # Provider profiles
├── deploy/
│   ├── docker/                # Docker files
│   ├── kubernetes/            # K8s manifests
│   └── helm/                  # Helm charts
├── docs/                      # Documentation
├── scripts/                   # Build/deploy scripts
└── plans/                     # SPARC documentation
```

#### 2.1.3 Build Commands

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- serve

# Watch mode for development
cargo watch -x 'run -- serve'

# Format code
cargo fmt

# Lint code
cargo clippy -- -D warnings

# Security audit
cargo audit

# Generate documentation
cargo doc --open
```

### 2.2 Module Implementation Order

Based on the dependency graph from the Refinement document:

```
Phase 1: Foundation
├── 1. src/config/schema.rs        # Configuration schema
├── 2. src/config/loader.rs        # Configuration loading
├── 3. src/engine/rng.rs           # Deterministic RNG
└── 4. src/engine/simulation.rs    # Core engine (skeleton)

Phase 2: Providers
├── 5. src/providers/traits.rs     # Provider trait
├── 6. src/providers/openai.rs     # OpenAI implementation
├── 7. src/providers/anthropic.rs  # Anthropic implementation
└── 8. src/providers/registry.rs   # Provider registry

Phase 3: Simulation
├── 9. src/latency/distributions.rs # Statistical distributions
├── 10. src/latency/profiles.rs     # Latency profiles
├── 11. src/latency/streaming.rs    # Streaming timing
├── 12. src/errors/injection.rs     # Error injection
├── 13. src/errors/patterns.rs      # Error patterns
└── 14. src/errors/responses.rs     # Error responses

Phase 4: Server
├── 15. src/server/routes.rs        # Route definitions
├── 16. src/server/handlers.rs      # Request handlers
├── 17. src/server/middleware.rs    # Middleware stack
└── 18. src/server/streaming.rs     # SSE streaming

Phase 5: Observability
├── 19. src/telemetry/metrics.rs    # Metrics collection
├── 20. src/telemetry/tracing.rs    # Distributed tracing
└── 21. src/telemetry/logging.rs    # Structured logging

Phase 6: Integration
├── 22. src/engine/session.rs       # Session management
├── 23. src/load/patterns.rs        # Load patterns
└── 24. src/main.rs                 # Entry point integration
```

### 2.3 Key Implementation Patterns

#### 2.3.1 Error Handling Pattern

```rust
// Use thiserror for error definitions
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimulationError {
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded {
        retry_after: Duration,
    },

    #[error("Request timeout after {0:?}")]
    Timeout(Duration),

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

// Convert to HTTP response
impl IntoResponse for SimulationError {
    fn into_response(self) -> Response {
        let (status, error_response) = match &self {
            SimulationError::ProviderNotFound(_) => (
                StatusCode::NOT_FOUND,
                ErrorResponse::new("model_not_found", &self.to_string()),
            ),
            SimulationError::RateLimitExceeded { retry_after } => (
                StatusCode::TOO_MANY_REQUESTS,
                ErrorResponse::rate_limit(*retry_after),
            ),
            SimulationError::Timeout(_) => (
                StatusCode::GATEWAY_TIMEOUT,
                ErrorResponse::new("timeout", &self.to_string()),
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ErrorResponse::new("internal_error", "An internal error occurred"),
            ),
        };

        (status, Json(error_response)).into_response()
    }
}
```

#### 2.3.2 Async Handler Pattern

```rust
// Handler with proper error handling and tracing
#[tracing::instrument(
    skip(state, request),
    fields(
        request_id = %request.id,
        model = %request.model,
    )
)]
pub async fn chat_completion_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, SimulationError> {
    // Validate request
    request.validate()?;

    // Check if streaming
    if request.stream.unwrap_or(false) {
        let stream = state.engine
            .process_chat_completion_stream(request)
            .await?;

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let response = state.engine
            .process_chat_completion(request)
            .await?;

        Ok(Json(response).into_response())
    }
}
```

#### 2.3.3 Configuration Loading Pattern

```rust
// Hierarchical configuration with validation
pub fn load_config() -> Result<SimulatorConfig, ConfigError> {
    let config = Config::builder()
        // Start with defaults
        .add_source(config::File::from_str(
            include_str!("../config/default.yaml"),
            FileFormat::Yaml,
        ))
        // Layer local config if exists
        .add_source(
            config::File::with_name("simulator")
                .required(false)
        )
        // Layer environment variables
        .add_source(
            config::Environment::with_prefix("LLM_SIM")
                .separator("__")
        )
        .build()?;

    // Deserialize and validate
    let config: SimulatorConfig = config.try_deserialize()?;
    config.validate()?;

    Ok(config)
}
```

#### 2.3.4 Graceful Shutdown Pattern

```rust
pub async fn run_server(config: SimulatorConfig) -> Result<(), ServerError> {
    let engine = Arc::new(SimulationEngine::new(config.clone()).await?);
    let app = create_router(engine.clone());

    let addr = SocketAddr::from((config.server.host, config.server.port));
    let listener = TcpListener::bind(addr).await?;

    tracing::info!("Server listening on {}", addr);

    // Graceful shutdown handling
    let shutdown_signal = async {
        let ctrl_c = async {
            signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        tracing::info!("Shutdown signal received");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    // Cleanup
    engine.shutdown().await;

    Ok(())
}
```

### 2.4 Testing Patterns

#### 2.4.1 Unit Test Pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng_reproducibility() {
        let seed = 12345u64;

        let mut rng1 = DeterministicRng::new(seed);
        let mut rng2 = DeterministicRng::new(seed);

        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_normal_distribution_parameters() {
        let mut rng = DeterministicRng::new(42);
        let mean = 100.0;
        let std_dev = 10.0;

        let samples: Vec<f64> = (0..10000)
            .map(|_| rng.next_normal(mean, std_dev))
            .collect();

        let actual_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let actual_var = samples.iter()
            .map(|x| (x - actual_mean).powi(2))
            .sum::<f64>() / samples.len() as f64;

        assert!((actual_mean - mean).abs() < 1.0);
        assert!((actual_var.sqrt() - std_dev).abs() < 1.0);
    }
}
```

#### 2.4.2 Integration Test Pattern

```rust
// tests/integration/api_test.rs
use axum_test::TestServer;
use llm_simulator::{create_app, Config};

#[tokio::test]
async fn test_chat_completion_endpoint() {
    let config = Config::default();
    let app = create_app(config).await;
    let server = TestServer::new(app).unwrap();

    let response = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }))
        .await;

    response.assert_status_ok();

    let body: ChatCompletionResponse = response.json();
    assert_eq!(body.object, "chat.completion");
    assert!(!body.choices.is_empty());
}

#[tokio::test]
async fn test_streaming_response() {
    let config = Config::default();
    let app = create_app(config).await;
    let server = TestServer::new(app).unwrap();

    let response = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .await;

    response.assert_status_ok();
    response.assert_header("content-type", "text/event-stream");

    let events: Vec<_> = response.sse_events().collect();
    assert!(events.len() > 1);
    assert!(events.last().unwrap().data.contains("[DONE]"));
}
```

---

## 3. Coding Standards

### 3.1 Rust Style Guide

#### 3.1.1 Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `latency_model` |
| Types | PascalCase | `SimulationEngine` |
| Functions | snake_case | `process_request` |
| Constants | SCREAMING_SNAKE | `MAX_TOKENS` |
| Variables | snake_case | `request_count` |
| Traits | PascalCase | `Provider` |
| Type Parameters | Single uppercase | `T`, `E` |
| Lifetimes | Short lowercase | `'a`, `'ctx` |

#### 3.1.2 File Organization

```rust
// Standard file organization

// 1. Module documentation
//! Module-level documentation describing purpose and usage.
//!
//! # Examples
//!
//! ```rust
//! use llm_simulator::engine::SimulationEngine;
//! ```

// 2. Imports (grouped and sorted)
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{info, instrument};

use crate::config::Config;
use crate::errors::SimulationError;

// 3. Constants
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CONCURRENT: usize = 10_000;

// 4. Type definitions
type Result<T> = std::result::Result<T, SimulationError>;

// 5. Traits
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(&self, request: &Request) -> Result<Response>;
}

// 6. Structs
pub struct SimulationEngine {
    config: Arc<RwLock<Config>>,
    // ...
}

// 7. Implementations
impl SimulationEngine {
    pub fn new(config: Config) -> Self {
        // ...
    }
}

// 8. Private helper functions
fn validate_request(request: &Request) -> Result<()> {
    // ...
}

// 9. Tests (in same file or tests/ directory)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        // ...
    }
}
```

#### 3.1.3 Documentation Standards

```rust
/// Processes a chat completion request through the simulation engine.
///
/// This function handles both streaming and non-streaming requests,
/// applying configured latency profiles and error injection rules.
///
/// # Arguments
///
/// * `request` - The chat completion request to process
///
/// # Returns
///
/// Returns a `ChatCompletionResponse` on success, or a `SimulationError`
/// if the request cannot be processed.
///
/// # Errors
///
/// This function will return an error if:
/// - The requested model is not configured
/// - The request exceeds configured limits
/// - Error injection is triggered
///
/// # Examples
///
/// ```rust
/// use llm_simulator::engine::SimulationEngine;
///
/// let engine = SimulationEngine::new(config).await?;
/// let response = engine.process_chat_completion(request).await?;
/// println!("Response: {:?}", response);
/// ```
///
/// # Performance
///
/// This function has O(n) complexity where n is the number of tokens
/// in the response. Memory usage is bounded by the configured
/// `max_tokens` parameter.
pub async fn process_chat_completion(
    &self,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse> {
    // Implementation
}
```

### 3.2 Error Handling Standards

#### 3.2.1 Error Categories

```rust
// Define clear error hierarchy
#[derive(Error, Debug)]
pub enum SimulationError {
    // Configuration errors (4xx - client fixable)
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    // Validation errors (400)
    #[error("Validation error: {0}")]
    Validation(String),

    // Not found errors (404)
    #[error("Resource not found: {0}")]
    NotFound(String),

    // Rate limiting (429)
    #[error("Rate limit exceeded")]
    RateLimit { retry_after: Duration },

    // Timeout errors (504)
    #[error("Request timeout")]
    Timeout,

    // Internal errors (500)
    #[error("Internal error: {0}")]
    Internal(String),

    // Injected errors (for simulation)
    #[error("Simulated error: {0}")]
    Injected(InjectedError),
}

// Always provide context
impl SimulationError {
    pub fn context(self, ctx: &str) -> Self {
        match self {
            Self::Internal(msg) => Self::Internal(format!("{}: {}", ctx, msg)),
            other => other,
        }
    }
}
```

#### 3.2.2 Error Propagation

```rust
// Use ? operator with context
async fn process_request(&self, req: Request) -> Result<Response> {
    let config = self.load_config()
        .await
        .map_err(|e| e.context("loading config"))?;

    let provider = self.get_provider(&req.model)
        .ok_or_else(|| SimulationError::NotFound(
            format!("model: {}", req.model)
        ))?;

    provider.generate(&req)
        .await
        .map_err(|e| e.context("generating response"))
}
```

### 3.3 Logging Standards

#### 3.3.1 Log Levels

| Level | Usage | Example |
|-------|-------|---------|
| ERROR | Errors requiring attention | Failed request, invalid config |
| WARN | Unexpected but handled | Retry, fallback used |
| INFO | Significant events | Server started, config reloaded |
| DEBUG | Development details | Request details, state changes |
| TRACE | Fine-grained details | Function entry/exit, values |

#### 3.3.2 Structured Logging

```rust
use tracing::{info, warn, error, instrument, Span};

#[instrument(
    skip(self, request),
    fields(
        request_id = %request.id,
        model = %request.model,
        tokens = tracing::field::Empty,
    )
)]
pub async fn process(&self, request: Request) -> Result<Response> {
    info!("Processing request");

    let response = match self.generate(&request).await {
        Ok(resp) => {
            // Record computed field
            Span::current().record("tokens", resp.usage.total_tokens);
            resp
        }
        Err(e) => {
            error!(error = %e, "Request failed");
            return Err(e);
        }
    };

    info!(
        tokens = response.usage.total_tokens,
        latency_ms = ?start.elapsed().as_millis(),
        "Request completed"
    );

    Ok(response)
}
```

### 3.4 Performance Standards

#### 3.4.1 Allocation Guidelines

```rust
// DO: Reuse buffers
pub struct RequestProcessor {
    buffer_pool: BufferPool,
}

impl RequestProcessor {
    pub fn process(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buffer = self.buffer_pool.acquire();
        // Use buffer...
        let result = buffer.to_vec();
        self.buffer_pool.release(buffer);
        Ok(result)
    }
}

// DON'T: Allocate in hot path
pub fn process_slow(&self, data: &[u8]) -> Result<Vec<u8>> {
    let buffer = Vec::with_capacity(data.len()); // Allocation every call
    // ...
}
```

#### 3.4.2 Async Guidelines

```rust
// DO: Use non-blocking operations
async fn fetch_config(&self) -> Result<Config> {
    let config = tokio::fs::read_to_string(&self.path).await?;
    Ok(serde_yaml::from_str(&config)?)
}

// DON'T: Block the async runtime
async fn fetch_config_blocking(&self) -> Result<Config> {
    let config = std::fs::read_to_string(&self.path)?; // Blocks!
    Ok(serde_yaml::from_str(&config)?)
}

// DO: Use spawn_blocking for CPU-intensive work
async fn compute_heavy(&self, data: &[u8]) -> Result<Output> {
    let data = data.to_vec();
    tokio::task::spawn_blocking(move || {
        heavy_computation(&data)
    }).await?
}
```

---

## 4. API Reference

### 4.1 OpenAI-Compatible Endpoints

#### 4.1.1 Chat Completions

**Endpoint:** `POST /v1/chat/completions`

**Request:**
```json
{
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "n": 1,
  "stop": ["\n\n"],
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "user": "user-123"
}
```

**Response (Non-Streaming):**
```json
{
  "id": "chatcmpl-sim-abc123",
  "object": "chat.completion",
  "created": 1699900000,
  "model": "gpt-4-turbo-simulated",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 18,
    "total_tokens": 43
  },
  "system_fingerprint": "sim-12345"
}
```

**Response (Streaming):**
```
data: {"id":"chatcmpl-sim-abc123","object":"chat.completion.chunk","created":1699900000,"model":"gpt-4-turbo-simulated","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-sim-abc123","object":"chat.completion.chunk","created":1699900000,"model":"gpt-4-turbo-simulated","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-sim-abc123","object":"chat.completion.chunk","created":1699900000,"model":"gpt-4-turbo-simulated","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### 4.1.2 Completions (Legacy)

**Endpoint:** `POST /v1/completions`

**Request:**
```json
{
  "model": "gpt-3.5-turbo-instruct",
  "prompt": "Say hello in French:",
  "max_tokens": 50,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "cmpl-sim-xyz789",
  "object": "text_completion",
  "created": 1699900000,
  "model": "gpt-3.5-turbo-instruct-simulated",
  "choices": [
    {
      "text": " Bonjour!",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 3,
    "total_tokens": 8
  }
}
```

#### 4.1.3 Embeddings

**Endpoint:** `POST /v1/embeddings`

**Request:**
```json
{
  "model": "text-embedding-3-small",
  "input": "The quick brown fox",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0145, 0.0089, ...]
    }
  ],
  "model": "text-embedding-3-small-simulated",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

#### 4.1.4 Models

**Endpoint:** `GET /v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4-turbo",
      "object": "model",
      "created": 1699900000,
      "owned_by": "llm-simulator"
    },
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1699900000,
      "owned_by": "llm-simulator"
    }
  ]
}
```

### 4.2 Anthropic-Compatible Endpoints

#### 4.2.1 Messages

**Endpoint:** `POST /v1/messages`

**Request:**
```json
{
  "model": "claude-3-opus-20240229",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Hello, Claude!"
    }
  ]
}
```

**Response:**
```json
{
  "id": "msg-sim-abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I assist you today?"
    }
  ],
  "model": "claude-3-opus-20240229-simulated",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 12
  }
}
```

### 4.3 Admin Endpoints

#### 4.3.1 Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "checks": {
    "config": "ok",
    "providers": "ok",
    "metrics": "ok"
  }
}
```

#### 4.3.2 Readiness Probe

**Endpoint:** `GET /ready`

**Response:** `200 OK` or `503 Service Unavailable`

#### 4.3.3 Liveness Probe

**Endpoint:** `GET /live`

**Response:** `200 OK`

#### 4.3.4 Metrics

**Endpoint:** `GET /metrics`

**Response:** Prometheus text format
```
# HELP llm_simulator_requests_total Total number of requests
# TYPE llm_simulator_requests_total counter
llm_simulator_requests_total{model="gpt-4-turbo",status="success"} 1234

# HELP llm_simulator_request_duration_seconds Request duration
# TYPE llm_simulator_request_duration_seconds histogram
llm_simulator_request_duration_seconds_bucket{le="0.1"} 100
llm_simulator_request_duration_seconds_bucket{le="0.5"} 500
llm_simulator_request_duration_seconds_bucket{le="1.0"} 800
```

#### 4.3.5 Configuration

**Endpoint:** `GET /admin/config`

**Response:**
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "providers": ["gpt-4-turbo", "claude-3-opus"],
  "simulation": {
    "deterministic": true,
    "seed": 12345
  }
}
```

**Endpoint:** `POST /admin/config/reload`

**Response:**
```json
{
  "status": "reloaded",
  "timestamp": "2025-11-26T10:00:00Z"
}
```

#### 4.3.6 Scenarios

**Endpoint:** `POST /admin/scenarios/{name}/activate`

**Request:**
```json
{
  "duration_seconds": 300,
  "parameters": {
    "error_rate": 0.1
  }
}
```

**Response:**
```json
{
  "scenario": "degraded_service",
  "status": "active",
  "expires_at": "2025-11-26T10:05:00Z"
}
```

### 4.4 Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "error_type",
    "code": "error_code",
    "param": "affected_parameter"
  }
}
```

**Error Types:**

| HTTP Status | Type | Description |
|-------------|------|-------------|
| 400 | `invalid_request_error` | Malformed request |
| 401 | `authentication_error` | Invalid API key |
| 403 | `permission_error` | Insufficient permissions |
| 404 | `not_found_error` | Resource not found |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `server_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

---

## 5. SDK Integration Guide

### 5.1 OpenAI SDK

#### 5.1.1 Python

```python
from openai import OpenAI

# Point to LLM-Simulator
client = OpenAI(
    api_key="sk-simulated-key",  # Any valid format
    base_url="http://localhost:8080/v1"
)

# Use exactly like production
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### 5.1.2 Node.js/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'sk-simulated-key',
  baseURL: 'http://localhost:8080/v1',
});

async function main() {
  const response = await client.chat.completions.create({
    model: 'gpt-4-turbo',
    messages: [{ role: 'user', content: 'Hello!' }],
  });

  console.log(response.choices[0].message.content);
}

// Streaming
async function streamExample() {
  const stream = await client.chat.completions.create({
    model: 'gpt-4-turbo',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: true,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
}
```

### 5.2 Anthropic SDK

#### 5.2.1 Python

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-ant-simulated",
    base_url="http://localhost:8080"
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(message.content[0].text)
```

### 5.3 curl Examples

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-simulated-key" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-simulated-key" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

### 5.4 Environment Configuration

```bash
# Switch between simulator and production
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.openai.com/v1}"

# In development/testing
export LLM_BASE_URL="http://localhost:8080/v1"

# In production
export LLM_BASE_URL="https://api.openai.com/v1"
```

---

## 6. Deployment Procedures

### 6.1 Binary Deployment

#### 6.1.1 Download and Install

```bash
# Download latest release
curl -LO https://github.com/llm-devops/llm-simulator/releases/latest/download/llm-simulator-linux-x86_64.tar.gz

# Extract
tar -xzf llm-simulator-linux-x86_64.tar.gz

# Make executable
chmod +x llm-simulator

# Verify
./llm-simulator --version

# Run
./llm-simulator serve --config config.yaml
```

#### 6.1.2 Systemd Service

```ini
# /etc/systemd/system/llm-simulator.service
[Unit]
Description=LLM Simulator
After=network.target

[Service]
Type=simple
User=llmsim
Group=llmsim
WorkingDirectory=/opt/llm-simulator
ExecStart=/opt/llm-simulator/llm-simulator serve --config /etc/llm-simulator/config.yaml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadOnlyPaths=/

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable llm-simulator
sudo systemctl start llm-simulator

# Check status
sudo systemctl status llm-simulator

# View logs
sudo journalctl -u llm-simulator -f
```

### 6.2 Docker Deployment

#### 6.2.1 Basic Usage

```bash
# Pull image
docker pull ghcr.io/llm-devops/llm-simulator:latest

# Run with default config
docker run -p 8080:8080 ghcr.io/llm-devops/llm-simulator:latest

# Run with custom config
docker run -p 8080:8080 \
  -v $(pwd)/config.yaml:/app/config/simulator.yaml:ro \
  ghcr.io/llm-devops/llm-simulator:latest

# Run with environment variables
docker run -p 8080:8080 \
  -e LLM_SIM__SERVER__PORT=8080 \
  -e LLM_SIM__SIMULATION__SEED=12345 \
  ghcr.io/llm-devops/llm-simulator:latest
```

#### 6.2.2 Docker Compose

```yaml
# docker-compose.yaml
version: '3.8'

services:
  llm-simulator:
    image: ghcr.io/llm-devops/llm-simulator:latest
    ports:
      - "8080:8080"
      - "9090:9090"  # Metrics
    volumes:
      - ./config:/app/config:ro
      - ./profiles:/app/profiles:ro
    environment:
      - RUST_LOG=info
      - LLM_SIM__SERVER__HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yaml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f llm-simulator

# Stop
docker-compose down
```

### 6.3 Kubernetes Deployment

#### 6.3.1 Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-simulator
  labels:
    name: llm-simulator
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-simulator-config
  namespace: llm-simulator
data:
  simulator.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
    simulation:
      deterministic: true
      seed: 12345
    providers:
      gpt-4-turbo:
        latency:
          ttft:
            distribution: log_normal
            p50_ms: 800
            p95_ms: 1500
    telemetry:
      metrics:
        enabled: true
        prometheus_port: 9090
```

#### 6.3.2 Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-simulator
  namespace: llm-simulator
  labels:
    app: llm-simulator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-simulator
  template:
    metadata:
      labels:
        app: llm-simulator
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: llm-simulator
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: llm-simulator
          image: ghcr.io/llm-devops/llm-simulator:1.0.0
          ports:
            - name: http
              containerPort: 8080
            - name: metrics
              containerPort: 9090
          env:
            - name: RUST_LOG
              value: "info"
          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /live
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
      volumes:
        - name: config
          configMap:
            name: llm-simulator-config
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: llm-simulator
                topologyKey: topology.kubernetes.io/zone
```

#### 6.3.3 Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-simulator
  namespace: llm-simulator
spec:
  selector:
    app: llm-simulator
  ports:
    - name: http
      port: 80
      targetPort: 8080
    - name: metrics
      port: 9090
      targetPort: 9090
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-simulator
  namespace: llm-simulator
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - llm-simulator.example.com
      secretName: llm-simulator-tls
  rules:
    - host: llm-simulator.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: llm-simulator
                port:
                  number: 80
```

#### 6.3.4 HPA and PDB

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-simulator
  namespace: llm-simulator
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-simulator
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
---
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: llm-simulator
  namespace: llm-simulator
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: llm-simulator
```

### 6.4 Helm Deployment

```bash
# Add repository
helm repo add llm-devops https://charts.llm-devops.io
helm repo update

# Install with defaults
helm install llm-simulator llm-devops/llm-simulator \
  --namespace llm-simulator \
  --create-namespace

# Install with custom values
helm install llm-simulator llm-devops/llm-simulator \
  --namespace llm-simulator \
  --create-namespace \
  --values values.yaml

# Upgrade
helm upgrade llm-simulator llm-devops/llm-simulator \
  --namespace llm-simulator \
  --values values.yaml

# Rollback
helm rollback llm-simulator 1 --namespace llm-simulator
```

**values.yaml:**
```yaml
replicaCount: 3

image:
  repository: ghcr.io/llm-devops/llm-simulator
  tag: "1.0.0"
  pullPolicy: IfNotPresent

config:
  simulation:
    deterministic: true
    seed: 12345
  providers:
    gpt-4-turbo:
      enabled: true

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: llm-simulator.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: llm-simulator-tls
      hosts:
        - llm-simulator.example.com

serviceMonitor:
  enabled: true
```

---

## 7. Operational Runbooks

### 7.1 Runbook: Service Startup

**Objective:** Start LLM-Simulator service

**Prerequisites:**
- Configuration file prepared
- Network access verified
- Dependencies available

**Procedure:**

1. **Verify configuration:**
   ```bash
   llm-simulator validate --config config.yaml
   ```

2. **Start service:**
   ```bash
   # Binary
   llm-simulator serve --config config.yaml

   # Docker
   docker-compose up -d

   # Kubernetes
   kubectl apply -f deploy/kubernetes/
   ```

3. **Verify startup:**
   ```bash
   # Check health
   curl http://localhost:8080/health

   # Check logs
   # Binary: view stdout
   # Docker: docker-compose logs llm-simulator
   # K8s: kubectl logs -l app=llm-simulator
   ```

4. **Verify functionality:**
   ```bash
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4-turbo","messages":[{"role":"user","content":"test"}]}'
   ```

**Expected Outcome:**
- Health endpoint returns 200
- Test request returns valid response
- Metrics endpoint accessible

### 7.2 Runbook: Service Shutdown

**Objective:** Gracefully shutdown LLM-Simulator

**Procedure:**

1. **Initiate graceful shutdown:**
   ```bash
   # Binary: Send SIGTERM
   kill -TERM $(pgrep llm-simulator)

   # Docker
   docker-compose stop llm-simulator

   # Kubernetes
   kubectl scale deployment llm-simulator --replicas=0
   ```

2. **Monitor drain:**
   ```bash
   # Watch active connections
   watch 'curl -s http://localhost:8080/metrics | grep active_connections'
   ```

3. **Verify shutdown:**
   ```bash
   # Confirm process stopped
   pgrep llm-simulator  # Should return nothing
   ```

**Timeout:** 30 seconds for graceful shutdown before force kill

### 7.3 Runbook: Configuration Update

**Objective:** Update configuration without restart

**Procedure:**

1. **Validate new configuration:**
   ```bash
   llm-simulator validate --config new-config.yaml
   ```

2. **Backup current configuration:**
   ```bash
   cp config.yaml config.yaml.backup
   ```

3. **Deploy new configuration:**
   ```bash
   # File-based hot reload
   cp new-config.yaml config.yaml

   # API-based reload
   curl -X POST http://localhost:8080/admin/config/reload
   ```

4. **Verify reload:**
   ```bash
   curl http://localhost:8080/admin/config
   ```

**Rollback:**
```bash
cp config.yaml.backup config.yaml
curl -X POST http://localhost:8080/admin/config/reload
```

### 7.4 Runbook: Scaling

**Objective:** Scale service capacity

**Horizontal Scaling (Kubernetes):**

```bash
# Manual scaling
kubectl scale deployment llm-simulator --replicas=10

# Verify scaling
kubectl get pods -l app=llm-simulator

# Check HPA status
kubectl get hpa llm-simulator
```

**Vertical Scaling (Kubernetes):**

```bash
# Update resource limits
kubectl patch deployment llm-simulator -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "llm-simulator",
          "resources": {
            "limits": {
              "cpu": "4000m",
              "memory": "4Gi"
            }
          }
        }]
      }
    }
  }
}'
```

### 7.5 Runbook: Incident Response

**Objective:** Respond to service incidents

**Severity Classification:**

| Severity | Impact | Response Time | Escalation |
|----------|--------|---------------|------------|
| P1 | Service down | Immediate | Page on-call |
| P2 | Degraded performance | 15 min | Page on-call |
| P3 | Minor issue | 1 hour | Slack notification |
| P4 | Improvement | Next business day | Ticket |

**P1 Response Procedure:**

1. **Acknowledge alert** (within 5 minutes)

2. **Assess impact:**
   ```bash
   # Check service health
   curl http://localhost:8080/health

   # Check error rates
   curl -s http://localhost:8080/metrics | grep error

   # Check pod status
   kubectl get pods -l app=llm-simulator
   ```

3. **Attempt quick recovery:**
   ```bash
   # Restart pods
   kubectl rollout restart deployment llm-simulator

   # Or rollback
   kubectl rollout undo deployment llm-simulator
   ```

4. **Escalate if not resolved** (within 15 minutes)

5. **Document incident** after resolution

---

## 8. Monitoring and Alerting

### 8.1 Key Metrics

| Metric | Description | Warning | Critical |
|--------|-------------|---------|----------|
| `llm_simulator_requests_total` | Total requests | - | - |
| `llm_simulator_request_duration_seconds` | Request latency | p99 > 3s | p99 > 5s |
| `llm_simulator_errors_total` | Error count | rate > 1% | rate > 5% |
| `llm_simulator_active_requests` | In-flight requests | > 80% capacity | > 95% capacity |
| `llm_simulator_queue_depth` | Queued requests | > 100 | > 500 |

### 8.2 Prometheus Configuration

```yaml
# prometheus.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/rules/*.yaml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'llm-simulator'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['llm-simulator']
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: llm-simulator
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        target_label: __address__
        regex: (.+)
        replacement: ${1}:9090
```

### 8.3 Alert Rules

```yaml
# alerts.yaml
groups:
  - name: llm-simulator
    rules:
      - alert: LLMSimulatorDown
        expr: up{job="llm-simulator"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM Simulator is down"
          description: "Instance {{ $labels.instance }} has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: |
          sum(rate(llm_simulator_errors_total[5m]))
          / sum(rate(llm_simulator_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(llm_simulator_request_duration_seconds_bucket[5m])) by (le)
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P99 latency is {{ $value }}s"

      - alert: HighQueueDepth
        expr: llm_simulator_queue_depth > 500
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High queue depth"
          description: "Queue depth is {{ $value }}"
```

### 8.4 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "LLM Simulator",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(llm_simulator_requests_total[1m]))",
            "legendFormat": "Requests/s"
          }
        ]
      },
      {
        "title": "Latency Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sum(rate(llm_simulator_request_duration_seconds_bucket[1m])) by (le)",
            "format": "heatmap"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "sum(rate(llm_simulator_errors_total[5m])) / sum(rate(llm_simulator_requests_total[5m]))"
          }
        ],
        "thresholds": {
          "mode": "absolute",
          "steps": [
            {"color": "green", "value": null},
            {"color": "yellow", "value": 0.01},
            {"color": "red", "value": 0.05}
          ]
        }
      },
      {
        "title": "Active Requests",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(llm_simulator_active_requests)"
          }
        ]
      }
    ]
  }
}
```

---

## 9. Troubleshooting Guide

### 9.1 Common Issues

#### Issue: Service Won't Start

**Symptoms:** Process exits immediately, health check fails

**Diagnosis:**
```bash
# Check logs
journalctl -u llm-simulator -n 100

# Validate config
llm-simulator validate --config config.yaml

# Check port availability
ss -tlnp | grep 8080
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Invalid config | Fix configuration errors |
| Port in use | Change port or stop conflicting service |
| Missing permissions | Check file/directory permissions |
| Missing dependencies | Install required system libraries |

#### Issue: High Latency

**Symptoms:** Request duration exceeds SLA

**Diagnosis:**
```bash
# Check system resources
top -p $(pgrep llm-simulator)

# Check metrics
curl -s localhost:9090/metrics | grep duration

# Check for contention
curl -s localhost:9090/metrics | grep queue
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| CPU saturation | Scale horizontally or vertically |
| Memory pressure | Increase memory limits |
| Queue backlog | Increase concurrency limits |
| Slow clients | Implement client timeouts |

#### Issue: High Error Rate

**Symptoms:** Error rate exceeds threshold

**Diagnosis:**
```bash
# Check error types
curl -s localhost:9090/metrics | grep error

# Check logs for errors
grep -i error /var/log/llm-simulator.log | tail -100

# Check configuration
curl localhost:8080/admin/config
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Error injection active | Disable or adjust injection rate |
| Invalid requests | Check client request format |
| Resource exhaustion | Scale service |
| Configuration error | Fix and reload configuration |

#### Issue: Memory Growth

**Symptoms:** Memory usage increases over time

**Diagnosis:**
```bash
# Check memory usage
ps aux | grep llm-simulator

# Check for leaks
curl -s localhost:9090/metrics | grep memory

# Profile if needed
RUST_LOG=debug llm-simulator serve
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Session accumulation | Reduce session TTL |
| Buffer accumulation | Restart service (short-term) |
| Actual memory leak | Report bug, apply patch |

### 9.2 Diagnostic Commands

```bash
# Health check
curl http://localhost:8080/health | jq

# Detailed metrics
curl http://localhost:8080/metrics

# Current configuration
curl http://localhost:8080/admin/config | jq

# Runtime statistics
curl http://localhost:8080/admin/stats | jq

# Active sessions
curl http://localhost:8080/admin/sessions | jq

# Test request
curl -w "@curl-format.txt" -o /dev/null -s \
  http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4-turbo","messages":[{"role":"user","content":"test"}]}'
```

### 9.3 Log Analysis

```bash
# Find errors
grep -E "(ERROR|WARN)" /var/log/llm-simulator.log

# Find slow requests
grep "duration_ms" /var/log/llm-simulator.log | awk -F'duration_ms=' '{print $2}' | sort -n | tail -20

# Request distribution by model
grep "model=" /var/log/llm-simulator.log | grep -oP 'model=\K[^ ]+' | sort | uniq -c

# Error distribution
grep "error" /var/log/llm-simulator.log | grep -oP 'type=\K[^ ]+' | sort | uniq -c
```

---

## 10. Maintenance Procedures

### 10.1 Routine Maintenance

#### Daily
- [ ] Review error rates and latency metrics
- [ ] Check disk space for logs
- [ ] Verify backup completion (if applicable)

#### Weekly
- [ ] Review and rotate logs
- [ ] Check for security updates
- [ ] Review capacity trends
- [ ] Update documentation if needed

#### Monthly
- [ ] Review and optimize configuration
- [ ] Conduct performance benchmarks
- [ ] Review and update alerts
- [ ] Capacity planning review

### 10.2 Upgrade Procedure

**Pre-Upgrade:**
```bash
# 1. Review release notes
# 2. Test in staging environment
# 3. Backup current configuration
cp config.yaml config.yaml.backup

# 4. Note current version
llm-simulator --version
```

**Upgrade (Kubernetes):**
```bash
# 1. Update image tag
kubectl set image deployment/llm-simulator \
  llm-simulator=ghcr.io/llm-devops/llm-simulator:1.1.0

# 2. Monitor rollout
kubectl rollout status deployment/llm-simulator

# 3. Verify health
kubectl get pods -l app=llm-simulator
curl http://llm-simulator/health
```

**Rollback (if needed):**
```bash
kubectl rollout undo deployment/llm-simulator
```

### 10.3 Backup and Recovery

**Configuration Backup:**
```bash
# Backup to Git (recommended)
git add config/
git commit -m "Configuration backup $(date +%Y%m%d)"
git push

# Manual backup
tar -czf backup-$(date +%Y%m%d).tar.gz config/
```

**Recovery:**
```bash
# From Git
git checkout <commit-hash> -- config/
curl -X POST http://localhost:8080/admin/config/reload

# From archive
tar -xzf backup-YYYYMMDD.tar.gz
curl -X POST http://localhost:8080/admin/config/reload
```

---

## 11. Release Management

### 11.1 Version Strategy

**Semantic Versioning:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking API changes
- **MINOR:** New features, backward compatible
- **PATCH:** Bug fixes, backward compatible

**Version Examples:**
- `1.0.0` → Initial release
- `1.1.0` → New provider support
- `1.1.1` → Bug fix
- `2.0.0` → Breaking API change

### 11.2 Release Process

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Develop   │───▶│   Staging   │───▶│     RC      │───▶│   Release   │
│   Branch    │    │   Deploy    │    │   Testing   │    │   Tag       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │   Publish   │◀───│   Build     │◀───│   Artifacts │
                   │   Release   │    │   Binaries  │    │   Docker    │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

### 11.3 Release Checklist

**Pre-Release:**
- [ ] All tests pass
- [ ] Changelog updated
- [ ] Documentation updated
- [ ] Security scan clean
- [ ] Performance benchmarks pass
- [ ] Staging validation complete

**Release:**
- [ ] Tag created: `git tag v1.0.0`
- [ ] GitHub release created
- [ ] Docker images pushed
- [ ] Helm chart updated
- [ ] Documentation deployed

**Post-Release:**
- [ ] Monitor error rates
- [ ] Monitor performance
- [ ] Collect feedback
- [ ] Plan next iteration

### 11.4 Changelog Format

```markdown
# Changelog

## [1.1.0] - 2025-12-15

### Added
- Google Gemini provider support
- Batch API endpoint
- Custom metric labels configuration

### Changed
- Improved streaming performance by 20%
- Updated default latency profiles

### Fixed
- Memory leak in long-running streams
- Race condition in session cleanup

### Security
- Updated dependencies to patch CVE-XXXX-XXXX

## [1.0.0] - 2025-11-26

### Added
- Initial release
- OpenAI API compatibility
- Anthropic API compatibility
- Latency simulation
- Error injection
- OpenTelemetry integration
```

---

## 12. Production Readiness Checklist

### 12.1 Pre-Production Checklist

#### Infrastructure
- [ ] Kubernetes cluster provisioned
- [ ] Namespaces created
- [ ] RBAC configured
- [ ] Network policies applied
- [ ] Ingress controller configured
- [ ] TLS certificates provisioned
- [ ] DNS configured

#### Application
- [ ] Configuration validated
- [ ] Secrets management configured
- [ ] Resource limits set
- [ ] Health probes configured
- [ ] HPA configured
- [ ] PDB configured

#### Observability
- [ ] Prometheus configured
- [ ] Grafana dashboards deployed
- [ ] Alertmanager configured
- [ ] Alert rules deployed
- [ ] Log aggregation configured
- [ ] Tracing configured

#### Security
- [ ] Container image scanned
- [ ] Pod security policies applied
- [ ] Network policies tested
- [ ] Secrets encrypted
- [ ] RBAC least privilege verified

#### Operations
- [ ] Runbooks documented
- [ ] On-call rotation defined
- [ ] Escalation paths documented
- [ ] Backup procedures tested
- [ ] Recovery procedures tested

### 12.2 Go-Live Checklist

**T-7 Days:**
- [ ] Final staging validation
- [ ] Load testing complete
- [ ] Security review complete
- [ ] Documentation complete

**T-1 Day:**
- [ ] Stakeholder notification
- [ ] On-call briefing
- [ ] Rollback plan verified
- [ ] Monitoring dashboards open

**Go-Live:**
- [ ] Deploy to production
- [ ] Verify health checks
- [ ] Run smoke tests
- [ ] Monitor metrics
- [ ] Announce availability

**T+1 Day:**
- [ ] Review error rates
- [ ] Review latency
- [ ] Collect feedback
- [ ] Document lessons learned

### 12.3 Sign-Off

| Area | Approver | Date | Signature |
|------|----------|------|-----------|
| Engineering | | | |
| QA | | | |
| Security | | | |
| Operations | | | |
| Product | | | |

---

## 13. Support and Escalation

### 13.1 Support Tiers

| Tier | Scope | Response Time | Contact |
|------|-------|---------------|---------|
| L1 | Basic troubleshooting | 15 min | On-call engineer |
| L2 | Advanced debugging | 1 hour | Platform team |
| L3 | Code-level investigation | 4 hours | Core developers |

### 13.2 Escalation Matrix

| Severity | L1 Duration | L2 Duration | L3 Trigger |
|----------|-------------|-------------|------------|
| P1 | 15 min | 30 min | Immediately if no progress |
| P2 | 30 min | 2 hours | After L2 exhausted |
| P3 | 4 hours | Next day | As needed |

### 13.3 Contact Information

| Role | Contact | Availability |
|------|---------|--------------|
| On-Call | pagerduty://llm-simulator | 24/7 |
| Platform Team | #llm-platform (Slack) | Business hours |
| Security | security@example.com | Business hours |

---

## 14. Training and Onboarding

### 14.1 Training Modules

| Module | Duration | Audience | Materials |
|--------|----------|----------|-----------|
| Overview | 30 min | All | Slides, demo |
| API Usage | 1 hour | Developers | Tutorial, examples |
| Deployment | 2 hours | DevOps | Hands-on lab |
| Operations | 2 hours | SRE | Runbook walkthrough |
| Troubleshooting | 1 hour | Support | Case studies |

### 14.2 Onboarding Checklist

**Developer Onboarding:**
- [ ] Read specification document
- [ ] Complete API tutorial
- [ ] Set up local development
- [ ] Run integration tests
- [ ] Submit first PR

**Operations Onboarding:**
- [ ] Read architecture document
- [ ] Complete deployment lab
- [ ] Review runbooks
- [ ] Shadow on-call rotation
- [ ] Complete incident simulation

### 14.3 Knowledge Base

| Resource | Location | Purpose |
|----------|----------|---------|
| Specification | `plans/LLM-Simulator-Specification.md` | Requirements |
| Pseudocode | `plans/LLM-Simulator-Pseudocode.md` | Implementation guide |
| Architecture | `plans/LLM-Simulator-Architecture.md` | System design |
| Refinement | `plans/LLM-Simulator-Refinement.md` | Quality assurance |
| Completion | `plans/LLM-Simulator-Completion.md` | Operations guide |
| API Reference | `docs/api.md` | Endpoint documentation |
| FAQ | `docs/faq.md` | Common questions |

---

## Appendix A: Configuration Reference

### Complete Configuration Schema

```yaml
# LLM-Simulator Configuration Reference
# Version: 1.0.0

# Server configuration
server:
  host: "0.0.0.0"          # Bind address
  port: 8080               # HTTP port
  workers: 8               # Worker threads (0 = auto)
  max_connections: 10000   # Maximum concurrent connections
  request_timeout_secs: 300 # Request timeout
  shutdown_timeout_secs: 30 # Graceful shutdown timeout

# Simulation configuration
simulation:
  deterministic: true      # Enable deterministic mode
  seed: 12345              # Global RNG seed (null = random)

  concurrency:
    max_concurrent_requests: 10000
    queue_size: 1000
    backpressure_threshold: 0.8

  sessions:
    enabled: true
    max_sessions: 100000
    session_ttl_secs: 3600
    max_context_tokens: 128000

# Provider configurations
providers:
  gpt-4-turbo:
    enabled: true
    latency:
      ttft:
        distribution: "log_normal"
        p50_ms: 800
        p95_ms: 1500
        p99_ms: 2500
      itl:
        distribution: "normal"
        mean_ms: 20
        std_dev_ms: 5

    errors:
      enabled: true
      rate_limit:
        probability: 0.02
        retry_after_secs: 60
      timeout:
        probability: 0.01
        timeout_ms: 30000
      server_error:
        probability: 0.005

  claude-3-opus:
    enabled: true
    latency:
      ttft:
        distribution: "log_normal"
        p50_ms: 1000
        p95_ms: 2000
      itl:
        distribution: "normal"
        mean_ms: 25
        std_dev_ms: 8

# Telemetry configuration
telemetry:
  logging:
    level: "info"          # trace, debug, info, warn, error
    format: "json"         # json, pretty

  metrics:
    enabled: true
    prometheus_port: 9090

  tracing:
    enabled: true
    otlp_endpoint: "http://localhost:4317"
    sample_rate: 0.1

# Admin API configuration
admin:
  enabled: true
  api_key: null            # null = no auth required
  allowed_ips: []          # Empty = all allowed
```

---

## Appendix B: Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_SIM__SERVER__HOST` | Server bind address | `0.0.0.0` |
| `LLM_SIM__SERVER__PORT` | Server port | `8080` |
| `LLM_SIM__SIMULATION__SEED` | RNG seed | Random |
| `LLM_SIM__SIMULATION__DETERMINISTIC` | Deterministic mode | `true` |
| `LLM_SIM__TELEMETRY__LOGGING__LEVEL` | Log level | `info` |
| `RUST_LOG` | Rust logging filter | - |
| `RUST_BACKTRACE` | Enable backtraces | `0` |

---

## Document Metadata

- **Version:** 1.0.0
- **Status:** Production-Ready
- **License:** LLM Dev Ops Permanent Source-Available Commercial License v1.0
- **Copyright:** (c) 2025 Global Business Advisors Inc.
- **Classification:** Internal - LLM DevOps Platform Specification

---

**End of LLM-Simulator Completion Specification**

---

## SPARC Specification Complete

This document concludes the SPARC methodology for LLM-Simulator:

| Phase | Document | Status |
|-------|----------|--------|
| 1. Specification | `LLM-Simulator-Specification.md` | Complete |
| 2. Pseudocode | `LLM-Simulator-Pseudocode.md` | Complete |
| 3. Architecture | `LLM-Simulator-Architecture.md` | Complete |
| 4. Refinement | `LLM-Simulator-Refinement.md` | Complete |
| 5. Completion | `LLM-Simulator-Completion.md` | Complete |

The LLM-Simulator is now fully specified and ready for implementation.
