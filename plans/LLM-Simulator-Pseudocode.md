# LLM-Simulator Pseudocode Specification

> **Document Type:** SPARC Pseudocode Phase
> **Module:** LLM-Simulator
> **Version:** 1.0.0
> **Status:** Draft
> **Last Updated:** 2025-11-26
> **Language:** Rust (pseudocode)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Architecture](#2-module-architecture)
3. [Core Simulation Engine](#3-core-simulation-engine)
4. [Provider Abstraction Layer](#4-provider-abstraction-layer)
5. [Latency Modeling System](#5-latency-modeling-system)
6. [Error Injection Framework](#6-error-injection-framework)
7. [Configuration Management](#7-configuration-management)
8. [Telemetry & Observability](#8-telemetry--observability)
9. [HTTP Server & API Layer](#9-http-server--api-layer)
10. [Load Testing & Concurrency](#10-load-testing--concurrency)
11. [Integration Patterns](#11-integration-patterns)
12. [Production Checklist](#12-production-checklist)

---

## 1. Overview

This document provides enterprise-grade, production-ready pseudocode for the LLM-Simulator module. The design targets:

- **10,000+ requests/second** throughput
- **<5ms processing overhead** per request
- **100% deterministic** execution with seed-based reproducibility
- **Full API compatibility** with OpenAI and Anthropic
- **OpenTelemetry-native** observability

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Rust | Performance, safety, async |
| HTTP Server | Axum + Tower | High-performance, middleware |
| Async Runtime | Tokio | Industry standard |
| Serialization | Serde | Flexible formats |
| Metrics | OpenTelemetry + Prometheus | Standards compliance |
| Configuration | Figment | Multi-format support |

---

## 2. Module Architecture

```
llm-simulator/
├── src/
│   ├── main.rs                 # Entry point
│   ├── lib.rs                  # Library exports
│   │
│   ├── engine/                 # Core simulation engine
│   │   ├── mod.rs
│   │   ├── simulation.rs       # SimulationEngine
│   │   ├── request.rs          # Request processing
│   │   ├── session.rs          # Session management
│   │   └── rng.rs              # Deterministic RNG
│   │
│   ├── providers/              # Provider abstraction
│   │   ├── mod.rs
│   │   ├── traits.rs           # Provider trait
│   │   ├── openai.rs           # OpenAI implementation
│   │   ├── anthropic.rs        # Anthropic implementation
│   │   ├── google.rs           # Google/Gemini
│   │   ├── registry.rs         # Provider registry
│   │   └── transform.rs        # Request/response transforms
│   │
│   ├── latency/                # Latency modeling
│   │   ├── mod.rs
│   │   ├── distributions.rs    # Statistical distributions
│   │   ├── profiles.rs         # Provider profiles
│   │   ├── streaming.rs        # Token streaming timing
│   │   └── degradation.rs      # Load-dependent degradation
│   │
│   ├── errors/                 # Error injection
│   │   ├── mod.rs
│   │   ├── injection.rs        # Injection strategies
│   │   ├── patterns.rs         # Error patterns
│   │   ├── responses.rs        # Provider error formats
│   │   └── circuit_breaker.rs  # Circuit breaker
│   │
│   ├── config/                 # Configuration
│   │   ├── mod.rs
│   │   ├── schema.rs           # Configuration schema
│   │   ├── loader.rs           # Multi-format loader
│   │   ├── validation.rs       # Validation rules
│   │   └── hot_reload.rs       # File watching
│   │
│   ├── telemetry/              # Observability
│   │   ├── mod.rs
│   │   ├── tracing.rs          # Distributed tracing
│   │   ├── metrics.rs          # LLM metrics
│   │   ├── logging.rs          # Structured logging
│   │   └── export.rs           # Analytics Hub export
│   │
│   ├── server/                 # HTTP server
│   │   ├── mod.rs
│   │   ├── routes.rs           # Route definitions
│   │   ├── handlers.rs         # Request handlers
│   │   ├── middleware.rs       # Middleware stack
│   │   └── streaming.rs        # SSE streaming
│   │
│   └── load/                   # Load testing
│       ├── mod.rs
│       ├── patterns.rs         # Load patterns
│       ├── workers.rs          # Worker pool
│       ├── backpressure.rs     # Backpressure control
│       └── stats.rs            # Real-time statistics
│
├── config/
│   ├── simulator.yaml          # Default configuration
│   └── profiles/               # Provider profiles
│       ├── openai.yaml
│       └── anthropic.yaml
│
└── tests/
    ├── integration/
    └── benchmarks/
```

---

## 3. Core Simulation Engine

### 3.1 SimulationEngine Struct

```rust
// ============================================================================
// Module: engine/simulation.rs
// Purpose: Core orchestration of LLM simulation requests
// ============================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, Semaphore};
use dashmap::DashMap;

/// Main simulation engine - thread-safe, high-performance request processor
pub struct SimulationEngine {
    // Configuration (hot-reloadable)
    config: Arc<RwLock<SimulationConfig>>,

    // Provider registry for multi-provider support
    providers: Arc<ProviderRegistry>,

    // Latency model for realistic timing simulation
    latency_model: Arc<LatencyModel>,

    // Error injector for chaos engineering
    error_injector: Arc<ErrorInjector>,

    // Session store for conversation state
    sessions: Arc<SessionStore>,

    // Deterministic RNG for reproducibility
    rng: Arc<RwLock<DeterministicRng>>,

    // Concurrency control
    semaphore: Arc<Semaphore>,

    // Metrics collector
    metrics: Arc<MetricsCollector>,

    // Request counter for unique IDs
    request_counter: AtomicU64,

    // Graceful shutdown signal
    shutdown: Arc<tokio::sync::Notify>,
}

impl SimulationEngine {
    /// Create new simulation engine with configuration
    pub async fn new(config: SimulationConfig) -> Result<Self, EngineError> {
        // Validate configuration
        config.validate()?;

        // Initialize deterministic RNG with seed
        let rng = DeterministicRng::new(config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        }));

        // Initialize provider registry with configured providers
        let providers = ProviderRegistry::new();
        for (name, profile) in &config.providers {
            providers.register(name, profile.clone())?;
        }

        // Initialize latency model
        let latency_model = LatencyModel::from_config(&config.latency)?;

        // Initialize error injector
        let error_injector = ErrorInjector::from_config(&config.errors)?;

        // Initialize session store
        let sessions = SessionStore::new(SessionStoreConfig {
            max_sessions: config.sessions.max_sessions,
            session_ttl: config.sessions.session_ttl,
            max_context_tokens: config.sessions.max_context_tokens,
        });

        // Initialize metrics
        let metrics = MetricsCollector::new(&config.telemetry)?;

        Ok(Self {
            config: Arc::new(RwLock::new(config.clone())),
            providers: Arc::new(providers),
            latency_model: Arc::new(latency_model),
            error_injector: Arc::new(error_injector),
            sessions: Arc::new(sessions),
            rng: Arc::new(RwLock::new(rng)),
            semaphore: Arc::new(Semaphore::new(config.concurrency.max_concurrent_requests)),
            metrics: Arc::new(metrics),
            request_counter: AtomicU64::new(0),
            shutdown: Arc::new(tokio::sync::Notify::new()),
        })
    }

    /// Process a chat completion request
    pub async fn process_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, SimulationError> {
        // Acquire concurrency permit
        let _permit = self.semaphore
            .acquire()
            .await
            .map_err(|_| SimulationError::ServiceUnavailable)?;

        // Generate unique request ID
        let request_id = self.generate_request_id();

        // Start timing
        let start = std::time::Instant::now();

        // Create tracing span
        let span = tracing::info_span!(
            "chat_completion",
            request_id = %request_id,
            model = %request.model,
            provider = %self.get_provider_for_model(&request.model),
        );
        let _guard = span.enter();

        // Check for error injection
        if let Some(error) = self.error_injector
            .should_inject(&request.model, "chat_completion")
            .await
        {
            self.metrics.record_error(&error);
            return Err(error.into());
        }

        // Get provider and latency profile
        let provider = self.providers.get_for_model(&request.model)?;
        let profile = self.latency_model.get_profile(&request.model)?;

        // Calculate token counts
        let prompt_tokens = self.count_tokens(&request.messages)?;
        let completion_tokens = self.estimate_completion_tokens(&request)?;

        // Generate simulated response
        let response_content = self.generate_response_content(
            &request,
            &provider,
        ).await?;

        // Simulate latency
        let timing = profile.simulate_request(
            prompt_tokens,
            completion_tokens,
            &mut *self.rng.write().await,
        )?;

        tokio::time::sleep(timing.total_duration).await;

        // Build response
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-sim-{}", request_id),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: format!("{}-simulated", request.model),
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(response_content),
                    tool_calls: None,
                    name: None,
                },
                finish_reason: FinishReason::Stop,
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: prompt_tokens as u32,
                completion_tokens: completion_tokens as u32,
                total_tokens: (prompt_tokens + completion_tokens) as u32,
            },
            system_fingerprint: Some(format!("sim-{}", self.config.read().await.seed.unwrap_or(0))),
        };

        // Record metrics
        let duration = start.elapsed();
        self.metrics.record_request(
            &request.model,
            duration,
            prompt_tokens,
            completion_tokens,
            true,
        );

        Ok(response)
    }

    /// Process streaming chat completion
    pub async fn process_chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunk, SimulationError>>, SimulationError> {
        // Similar setup to non-streaming...
        let request_id = self.generate_request_id();
        let profile = self.latency_model.get_profile(&request.model)?;

        // Calculate timing
        let completion_tokens = self.estimate_completion_tokens(&request)?;
        let token_timings = profile.generate_streaming_timings(
            completion_tokens,
            &mut *self.rng.write().await,
        )?;

        // Generate response content
        let response_content = self.generate_response_content(&request, &provider).await?;
        let tokens = self.tokenize(&response_content)?;

        // Create stream
        let stream = async_stream::stream! {
            // Simulate TTFT
            tokio::time::sleep(token_timings.ttft).await;

            // Stream tokens
            for (i, (token, timing)) in tokens.iter().zip(token_timings.itls.iter()).enumerate() {
                yield Ok(ChatCompletionChunk {
                    id: format!("chatcmpl-sim-{}", request_id),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: if i == 0 { Some("assistant".to_string()) } else { None },
                            content: Some(token.clone()),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                });

                tokio::time::sleep(*timing).await;
            }

            // Final chunk with finish_reason
            yield Ok(ChatCompletionChunk {
                id: format!("chatcmpl-sim-{}", request_id),
                object: "chat.completion.chunk".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: request.model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                }],
            });
        };

        Ok(stream)
    }

    /// Generate unique request ID
    fn generate_request_id(&self) -> String {
        let counter = self.request_counter.fetch_add(1, Ordering::Relaxed);
        format!("{:016x}", counter)
    }

    /// Hot-reload configuration
    pub async fn reload_config(&self, new_config: SimulationConfig) -> Result<(), EngineError> {
        new_config.validate()?;

        // Update providers
        for (name, profile) in &new_config.providers {
            self.providers.update(name, profile.clone())?;
        }

        // Update config
        *self.config.write().await = new_config;

        tracing::info!("Configuration reloaded successfully");
        Ok(())
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) {
        tracing::info!("Initiating graceful shutdown");
        self.shutdown.notify_waiters();

        // Wait for in-flight requests
        let timeout = self.config.read().await.shutdown_timeout;
        tokio::time::timeout(timeout, async {
            while self.semaphore.available_permits()
                < self.config.read().await.concurrency.max_concurrent_requests
            {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }).await.ok();

        tracing::info!("Shutdown complete");
    }
}
```

### 3.2 Deterministic RNG

```rust
// ============================================================================
// Module: engine/rng.rs
// Purpose: Deterministic random number generation for reproducibility
// ============================================================================

/// XorShift64* PRNG - Fast, deterministic, excellent statistical properties
pub struct DeterministicRng {
    state: u64,
    initial_seed: u64,
    generations: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
            initial_seed: seed,
            generations: 0,
        }
    }

    /// Generate next u64
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        self.generations += 1;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Generate f64 in [0, 1)
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate normal distribution sample (Box-Muller)
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }

    /// Fork RNG for request-scoped operations
    pub fn fork(&mut self) -> DeterministicRng {
        let seed = self.next_u64();
        DeterministicRng::new(seed)
    }

    /// Checkpoint state for reproducibility
    pub fn checkpoint(&self) -> RngCheckpoint {
        RngCheckpoint {
            state: self.state,
            initial_seed: self.initial_seed,
            generations: self.generations,
        }
    }

    /// Restore from checkpoint
    pub fn restore(&mut self, checkpoint: RngCheckpoint) {
        self.state = checkpoint.state;
        self.initial_seed = checkpoint.initial_seed;
        self.generations = checkpoint.generations;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RngCheckpoint {
    pub state: u64,
    pub initial_seed: u64,
    pub generations: u64,
}
```

### 3.3 Session Management

```rust
// ============================================================================
// Module: engine/session.rs
// Purpose: Conversation state and session management
// ============================================================================

/// Thread-safe session store with TTL-based cleanup
pub struct SessionStore {
    sessions: DashMap<String, Session>,
    config: SessionStoreConfig,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub conversations: Vec<Conversation>,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: String,
    pub messages: Vec<Message>,
    pub total_tokens: usize,
    pub created_at: Instant,
}

impl SessionStore {
    pub fn new(config: SessionStoreConfig) -> Self {
        let store = Self {
            sessions: DashMap::new(),
            config,
        };

        // Spawn cleanup task
        store.spawn_cleanup_task();
        store
    }

    /// Get or create session
    pub fn get_or_create(&self, session_id: &str) -> Session {
        self.sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Session {
                id: session_id.to_string(),
                conversations: Vec::new(),
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                metadata: HashMap::new(),
            })
            .clone()
    }

    /// Update session with new message
    pub fn add_message(
        &self,
        session_id: &str,
        conversation_id: &str,
        message: Message,
        tokens: usize,
    ) -> Result<(), SessionError> {
        let mut session = self.sessions
            .get_mut(session_id)
            .ok_or(SessionError::NotFound)?;

        // Find or create conversation
        let conversation = session.conversations
            .iter_mut()
            .find(|c| c.id == conversation_id)
            .or_else(|| {
                session.conversations.push(Conversation {
                    id: conversation_id.to_string(),
                    messages: Vec::new(),
                    total_tokens: 0,
                    created_at: Instant::now(),
                });
                session.conversations.last_mut()
            })
            .unwrap();

        // Check context limit
        if conversation.total_tokens + tokens > self.config.max_context_tokens {
            // Trim oldest messages
            while conversation.total_tokens + tokens > self.config.max_context_tokens
                && !conversation.messages.is_empty()
            {
                conversation.messages.remove(0);
                // Would need token count per message for accurate trimming
            }
        }

        conversation.messages.push(message);
        conversation.total_tokens += tokens;
        session.last_accessed = Instant::now();

        Ok(())
    }

    fn spawn_cleanup_task(&self) {
        let sessions = self.sessions.clone();
        let ttl = self.config.session_ttl;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                let now = Instant::now();
                sessions.retain(|_, session| {
                    now.duration_since(session.last_accessed) < ttl
                });
            }
        });
    }
}
```

---

## 4. Provider Abstraction Layer

### 4.1 Provider Trait

```rust
// ============================================================================
// Module: providers/traits.rs
// Purpose: Core provider abstraction for multi-provider support
// ============================================================================

#[async_trait]
pub trait Provider: Send + Sync {
    /// Provider identification
    fn name(&self) -> &str;
    fn version(&self) -> ApiVersion;
    fn supported_features(&self) -> ProviderFeatures;

    /// Health and capability checks
    async fn health_check(&self) -> Result<HealthStatus, ProviderError>;

    /// Core LLM operations
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError>;

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunk, ProviderError>>, ProviderError>;

    async fn create_embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError>;

    /// Request transformation
    fn validate_request(&self, request: &dyn ValidatableRequest) -> Result<(), ProviderError>;
    fn transform_request(&self, request: UnifiedRequest) -> Result<ProviderRequest, ProviderError>;
    fn transform_response(&self, response: ProviderResponse) -> Result<UnifiedResponse, ProviderError>;

    /// Error mapping
    fn map_error(&self, error: ProviderSpecificError) -> ProviderError;
}

#[derive(Debug, Clone)]
pub struct ProviderFeatures {
    pub chat_completion: bool,
    pub streaming: bool,
    pub function_calling: bool,
    pub tool_use: bool,
    pub vision: bool,
    pub embeddings: bool,
    pub json_mode: bool,
    pub system_prompts: bool,
}
```

### 4.2 OpenAI Provider

```rust
// ============================================================================
// Module: providers/openai.rs
// Purpose: OpenAI API simulation with exact schema compatibility
// ============================================================================

pub struct OpenAIProvider {
    config: OpenAIConfig,
    latency_profile: Arc<LatencyProfile>,
    error_injector: Arc<ErrorInjector>,
    response_generator: Arc<ResponseGenerator>,
}

impl OpenAIProvider {
    /// Transform request to OpenAI format
    fn transform_chat_request(&self, req: &ChatCompletionRequest) -> serde_json::Value {
        json!({
            "model": req.model,
            "messages": self.transform_messages(&req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": req.stream,
            "tools": req.tools,
            "tool_choice": self.transform_tool_choice(&req.tool_choice),
            "response_format": req.response_format,
            "seed": req.seed,
        })
    }

    /// Generate OpenAI-compatible response
    fn generate_response(
        &self,
        request: &ChatCompletionRequest,
        content: String,
        usage: Usage,
    ) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(content),
                    tool_calls: None,
                    name: None,
                },
                finish_reason: FinishReason::Stop,
                logprobs: None,
            }],
            usage,
            system_fingerprint: Some("fp_sim_openai".to_string()),
        }
    }

    /// Generate OpenAI-compatible error response
    fn generate_error_response(&self, error: &InjectedError) -> serde_json::Value {
        json!({
            "error": {
                "message": error.message,
                "type": self.map_error_type(&error.error_type),
                "code": error.error_type.to_string(),
            }
        })
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str { "openai" }

    fn version(&self) -> ApiVersion {
        ApiVersion { major: 1, minor: 0, patch: 0, deprecated: false }
    }

    fn supported_features(&self) -> ProviderFeatures {
        ProviderFeatures {
            chat_completion: true,
            streaming: true,
            function_calling: true,
            tool_use: true,
            vision: true,
            embeddings: true,
            json_mode: true,
            system_prompts: true,
        }
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        // Validate request
        self.validate_request(&request)?;

        // Check error injection
        if let Some(error) = self.error_injector.should_inject("openai", "chat").await {
            return Err(error.into());
        }

        // Calculate tokens
        let prompt_tokens = self.count_tokens(&request.messages);
        let completion_tokens = self.estimate_completion_tokens(&request);

        // Generate response content
        let content = self.response_generator.generate(&request).await?;

        // Simulate latency
        let timing = self.latency_profile.simulate(prompt_tokens, completion_tokens)?;
        tokio::time::sleep(timing.total_duration).await;

        // Build response
        Ok(self.generate_response(
            &request,
            content,
            Usage {
                prompt_tokens: prompt_tokens as u32,
                completion_tokens: completion_tokens as u32,
                total_tokens: (prompt_tokens + completion_tokens) as u32,
            },
        ))
    }

    // ... additional implementations
}
```

### 4.3 Anthropic Provider

```rust
// ============================================================================
// Module: providers/anthropic.rs
// Purpose: Anthropic Messages API simulation
// ============================================================================

pub struct AnthropicProvider {
    config: AnthropicConfig,
    latency_profile: Arc<LatencyProfile>,
    error_injector: Arc<ErrorInjector>,
}

impl AnthropicProvider {
    /// Transform to Anthropic Messages format
    fn transform_to_anthropic(&self, req: &ChatCompletionRequest) -> Result<serde_json::Value, ProviderError> {
        // Anthropic separates system messages
        let (system, messages) = self.extract_system_and_messages(&req.messages)?;

        let mut anthropic_req = json!({
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens.unwrap_or(4096),
        });

        if let Some(sys) = system {
            anthropic_req["system"] = json!(sys);
        }

        if let Some(temp) = req.temperature {
            anthropic_req["temperature"] = json!(temp);
        }

        if req.stream {
            anthropic_req["stream"] = json!(true);
        }

        Ok(anthropic_req)
    }

    /// Generate Anthropic-compatible error
    fn generate_error_response(&self, error: &InjectedError) -> serde_json::Value {
        json!({
            "type": "error",
            "error": {
                "type": self.map_error_type(&error.error_type),
                "message": error.message,
            }
        })
    }

    /// Transform Anthropic response to unified format
    fn transform_response(&self, anthropic_resp: serde_json::Value) -> ChatCompletionResponse {
        // Extract content blocks
        let content = anthropic_resp["content"]
            .as_array()
            .map(|blocks| {
                blocks.iter()
                    .filter_map(|b| b["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        ChatCompletionResponse {
            id: anthropic_resp["id"].as_str().unwrap_or("").to_string(),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: anthropic_resp["model"].as_str().unwrap_or("").to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(content),
                    tool_calls: None,
                    name: None,
                },
                finish_reason: self.map_stop_reason(&anthropic_resp["stop_reason"]),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: anthropic_resp["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: anthropic_resp["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: 0, // Calculated
            },
            system_fingerprint: None,
        }
    }
}
```

### 4.4 Provider Registry

```rust
// ============================================================================
// Module: providers/registry.rs
// Purpose: Dynamic provider registration and lookup
// ============================================================================

pub struct ProviderRegistry {
    providers: DashMap<String, Arc<dyn Provider>>,
    model_mappings: DashMap<String, String>, // model -> provider
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: DashMap::new(),
            model_mappings: DashMap::new(),
        }
    }

    /// Register a provider
    pub fn register(&self, name: &str, config: ProviderConfig) -> Result<(), RegistryError> {
        let provider: Arc<dyn Provider> = match config.provider_type.as_str() {
            "openai" => Arc::new(OpenAIProvider::new(config)?),
            "anthropic" => Arc::new(AnthropicProvider::new(config)?),
            "google" => Arc::new(GoogleProvider::new(config)?),
            _ => return Err(RegistryError::UnknownProvider(config.provider_type)),
        };

        // Register model mappings
        for model in &config.models {
            self.model_mappings.insert(model.clone(), name.to_string());
        }

        self.providers.insert(name.to_string(), provider);
        Ok(())
    }

    /// Get provider for model
    pub fn get_for_model(&self, model: &str) -> Result<Arc<dyn Provider>, RegistryError> {
        let provider_name = self.model_mappings
            .get(model)
            .map(|r| r.clone())
            .or_else(|| self.infer_provider(model))
            .ok_or_else(|| RegistryError::ModelNotFound(model.to_string()))?;

        self.providers
            .get(&provider_name)
            .map(|r| Arc::clone(&r))
            .ok_or_else(|| RegistryError::ProviderNotFound(provider_name))
    }

    /// Infer provider from model name
    fn infer_provider(&self, model: &str) -> Option<String> {
        if model.starts_with("gpt-") || model.starts_with("o1") {
            Some("openai".to_string())
        } else if model.starts_with("claude-") {
            Some("anthropic".to_string())
        } else if model.starts_with("gemini-") {
            Some("google".to_string())
        } else {
            None
        }
    }
}
```

---

## 5. Latency Modeling System

### 5.1 Latency Distribution Trait

```rust
// ============================================================================
// Module: latency/distributions.rs
// Purpose: Statistical distributions for realistic latency simulation
// ============================================================================

/// Trait for latency distributions
pub trait LatencyDistribution: Send + Sync {
    /// Sample from distribution
    fn sample(&self, rng: &mut DeterministicRng) -> Duration;

    /// Get percentile value
    fn percentile(&self, p: f64) -> Duration;

    /// Distribution name
    fn name(&self) -> &str;
}

/// Log-normal distribution - realistic for LLM latencies
pub struct LogNormalDistribution {
    mu: f64,      // Location parameter (log-space)
    sigma: f64,   // Scale parameter (log-space)
}

impl LogNormalDistribution {
    /// Create from percentile targets
    pub fn from_percentiles(p50_ms: f64, p99_ms: f64) -> Self {
        // Solve for mu and sigma from p50 and p99
        let z_99 = 2.326; // 99th percentile z-score

        let mu = p50_ms.ln();
        let sigma = (p99_ms.ln() - mu) / z_99;

        Self { mu, sigma }
    }
}

impl LatencyDistribution for LogNormalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        let normal = rng.next_normal(0.0, 1.0);
        let value = (self.mu + self.sigma * normal).exp();
        Duration::from_secs_f64(value / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        let z = Self::inverse_normal_cdf(p);
        let value = (self.mu + self.sigma * z).exp();
        Duration::from_secs_f64(value / 1000.0)
    }

    fn name(&self) -> &str { "log_normal" }
}

/// Normal distribution
pub struct NormalDistribution {
    mean_ms: f64,
    std_dev_ms: f64,
}

impl LatencyDistribution for NormalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        let value = rng.next_normal(self.mean_ms, self.std_dev_ms).max(0.0);
        Duration::from_secs_f64(value / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        let z = Self::inverse_normal_cdf(p);
        let value = (self.mean_ms + self.std_dev_ms * z).max(0.0);
        Duration::from_secs_f64(value / 1000.0)
    }

    fn name(&self) -> &str { "normal" }
}

/// Empirical distribution from measurements
pub struct EmpiricalDistribution {
    sorted_samples: Vec<f64>,
}

impl EmpiricalDistribution {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { sorted_samples: samples }
    }
}

impl LatencyDistribution for EmpiricalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        let idx = (rng.next_f64() * self.sorted_samples.len() as f64) as usize;
        let value = self.sorted_samples[idx.min(self.sorted_samples.len() - 1)];
        Duration::from_secs_f64(value / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        let idx = (p * self.sorted_samples.len() as f64) as usize;
        let value = self.sorted_samples[idx.min(self.sorted_samples.len() - 1)];
        Duration::from_secs_f64(value / 1000.0)
    }

    fn name(&self) -> &str { "empirical" }
}
```

### 5.2 Latency Profile

```rust
// ============================================================================
// Module: latency/profiles.rs
// Purpose: Provider-specific latency profiles
// ============================================================================

/// Complete latency profile for a provider/model
pub struct LatencyProfile {
    pub name: String,
    pub provider: String,
    pub model: String,

    /// Time to first token distribution
    pub ttft: Box<dyn LatencyDistribution>,

    /// Inter-token latency distribution
    pub itl: Box<dyn LatencyDistribution>,

    /// Tokens per second (for batch estimation)
    pub tokens_per_second: f64,

    /// Load-dependent degradation model
    pub degradation: Option<DegradationModel>,
}

impl LatencyProfile {
    /// Create GPT-4 Turbo profile
    pub fn gpt4_turbo() -> Self {
        Self {
            name: "gpt-4-turbo".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4-turbo".to_string(),
            ttft: Box::new(LogNormalDistribution::from_percentiles(800.0, 2500.0)),
            itl: Box::new(NormalDistribution { mean_ms: 20.0, std_dev_ms: 5.0 }),
            tokens_per_second: 50.0,
            degradation: Some(DegradationModel::exponential(0.5, 10.0)),
        }
    }

    /// Create Claude 3 Opus profile
    pub fn claude3_opus() -> Self {
        Self {
            name: "claude-3-opus".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-3-opus-20240229".to_string(),
            ttft: Box::new(LogNormalDistribution::from_percentiles(1200.0, 3000.0)),
            itl: Box::new(NormalDistribution { mean_ms: 25.0, std_dev_ms: 6.0 }),
            tokens_per_second: 40.0,
            degradation: Some(DegradationModel::exponential(0.4, 8.0)),
        }
    }

    /// Simulate request timing
    pub fn simulate_request(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        rng: &mut DeterministicRng,
    ) -> RequestTiming {
        let ttft = self.ttft.sample(rng);

        let mut itls = Vec::with_capacity(completion_tokens);
        for _ in 0..completion_tokens {
            itls.push(self.itl.sample(rng));
        }

        let total_itl: Duration = itls.iter().sum();
        let total_duration = ttft + total_itl;

        RequestTiming {
            ttft,
            itls,
            total_duration,
            prompt_tokens,
            completion_tokens,
        }
    }

    /// Generate streaming token timings
    pub fn generate_streaming_timings(
        &self,
        completion_tokens: usize,
        rng: &mut DeterministicRng,
    ) -> StreamingTimings {
        let ttft = self.ttft.sample(rng);

        let itls: Vec<Duration> = (0..completion_tokens)
            .map(|_| self.itl.sample(rng))
            .collect();

        StreamingTimings { ttft, itls }
    }
}

#[derive(Debug, Clone)]
pub struct RequestTiming {
    pub ttft: Duration,
    pub itls: Vec<Duration>,
    pub total_duration: Duration,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct StreamingTimings {
    pub ttft: Duration,
    pub itls: Vec<Duration>,
}
```

### 5.3 Load-Dependent Degradation

```rust
// ============================================================================
// Module: latency/degradation.rs
// Purpose: Model latency degradation under load
// ============================================================================

/// Degradation model for load-dependent latency increase
pub enum DegradationModel {
    /// No degradation
    None,

    /// Linear: latency = base * (1 + alpha * load)
    Linear { alpha: f64 },

    /// Exponential: latency = base * exp(alpha * load)
    Exponential { alpha: f64, baseline_qps: f64 },

    /// M/M/1 queueing: latency = base / (1 - utilization)
    MM1Queue { service_rate: f64 },

    /// Piecewise linear with breakpoints
    Piecewise { breakpoints: Vec<(f64, f64)> },
}

impl DegradationModel {
    pub fn exponential(alpha: f64, baseline_qps: f64) -> Self {
        Self::Exponential { alpha, baseline_qps }
    }

    /// Calculate degradation multiplier
    pub fn calculate_multiplier(&self, current_qps: f64) -> f64 {
        match self {
            Self::None => 1.0,

            Self::Linear { alpha } => 1.0 + alpha * current_qps,

            Self::Exponential { alpha, baseline_qps } => {
                let normalized_load = current_qps / baseline_qps;
                (alpha * normalized_load).exp()
            }

            Self::MM1Queue { service_rate } => {
                let utilization = (current_qps / service_rate).min(0.99);
                1.0 / (1.0 - utilization)
            }

            Self::Piecewise { breakpoints } => {
                for window in breakpoints.windows(2) {
                    let (qps1, mult1) = window[0];
                    let (qps2, mult2) = window[1];

                    if current_qps >= qps1 && current_qps < qps2 {
                        let t = (current_qps - qps1) / (qps2 - qps1);
                        return mult1 + t * (mult2 - mult1);
                    }
                }
                breakpoints.last().map_or(1.0, |(_, m)| *m)
            }
        }
    }
}
```

---

## 6. Error Injection Framework

### 6.1 Error Injection Trait

```rust
// ============================================================================
// Module: errors/injection.rs
// Purpose: Configurable error injection for chaos engineering
// ============================================================================

/// Error injection orchestrator
pub struct ErrorInjector {
    strategies: Vec<Box<dyn ErrorInjectionStrategy>>,
    circuit_breaker: Option<CircuitBreaker>,
    formatters: HashMap<String, Box<dyn ErrorFormatter>>,
    metrics: Arc<InjectionMetrics>,
    config: ErrorInjectionConfig,
}

/// Strategy for determining when to inject errors
#[async_trait]
pub trait ErrorInjectionStrategy: Send + Sync {
    /// Determine if error should be injected
    async fn should_inject(&self, context: &RequestContext) -> Option<InjectedError>;

    /// Strategy name
    fn name(&self) -> &str;

    /// Reset state (e.g., after config reload)
    fn reset(&self);
}

/// Probabilistic error injection
pub struct ProbabilisticStrategy {
    error_type: ErrorType,
    probability: f64,
    rng: Arc<RwLock<DeterministicRng>>,
}

impl ProbabilisticStrategy {
    pub fn new(error_type: ErrorType, probability: f64, seed: u64) -> Self {
        Self {
            error_type,
            probability,
            rng: Arc::new(RwLock::new(DeterministicRng::new(seed))),
        }
    }
}

#[async_trait]
impl ErrorInjectionStrategy for ProbabilisticStrategy {
    async fn should_inject(&self, _context: &RequestContext) -> Option<InjectedError> {
        let roll = self.rng.write().await.next_f64();

        if roll < self.probability {
            Some(InjectedError {
                error_type: self.error_type.clone(),
                http_status: self.error_type.http_status(),
                message: self.error_type.default_message(),
                retry_after: self.error_type.retry_after(),
                headers: self.error_type.additional_headers(),
            })
        } else {
            None
        }
    }

    fn name(&self) -> &str { "probabilistic" }
    fn reset(&self) {}
}

/// Sequence-based error injection
pub struct SequenceStrategy {
    error_type: ErrorType,
    pattern: SequencePattern,
    counter: AtomicU64,
}

#[derive(Debug, Clone)]
pub enum SequencePattern {
    EveryNth(u64),
    Range { start: u64, end: u64 },
    Specific(Vec<u64>),
    Burst { failures: u64, successes: u64 },
}

#[async_trait]
impl ErrorInjectionStrategy for SequenceStrategy {
    async fn should_inject(&self, _context: &RequestContext) -> Option<InjectedError> {
        let count = self.counter.fetch_add(1, Ordering::Relaxed);

        let should_fail = match &self.pattern {
            SequencePattern::EveryNth(n) => count % n == 0,
            SequencePattern::Range { start, end } => count >= *start && count < *end,
            SequencePattern::Specific(nums) => nums.contains(&count),
            SequencePattern::Burst { failures, successes } => {
                let cycle = failures + successes;
                count % cycle < *failures
            }
        };

        if should_fail {
            Some(InjectedError {
                error_type: self.error_type.clone(),
                http_status: self.error_type.http_status(),
                message: self.error_type.default_message(),
                retry_after: self.error_type.retry_after(),
                headers: HashMap::new(),
            })
        } else {
            None
        }
    }

    fn name(&self) -> &str { "sequence" }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }
}
```

### 6.2 Error Types

```rust
// ============================================================================
// Module: errors/types.rs
// Purpose: Comprehensive error type definitions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    // Rate limiting
    RateLimit {
        limit_type: RateLimitType,
        retry_after_seconds: u64,
    },

    // Authentication
    AuthenticationError { message: String },
    PermissionDenied { message: String },

    // Request errors
    BadRequest { message: String },
    ContextLengthExceeded { max_tokens: u32, requested: u32 },
    ContentFilter { reason: String },

    // Server errors
    InternalServerError,
    ServiceUnavailable { retry_after_seconds: Option<u64> },
    BadGateway,

    // Timeout
    Timeout { timeout_ms: u64 },

    // Quota
    QuotaExceeded { reset_time: DateTime<Utc> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitType {
    RequestsPerMinute,
    TokensPerMinute,
    RequestsPerDay,
    ConcurrentRequests,
}

impl ErrorType {
    pub fn http_status(&self) -> u16 {
        match self {
            Self::RateLimit { .. } => 429,
            Self::AuthenticationError { .. } => 401,
            Self::PermissionDenied { .. } => 403,
            Self::BadRequest { .. } => 400,
            Self::ContextLengthExceeded { .. } => 400,
            Self::ContentFilter { .. } => 400,
            Self::InternalServerError => 500,
            Self::ServiceUnavailable { .. } => 503,
            Self::BadGateway => 502,
            Self::Timeout { .. } => 504,
            Self::QuotaExceeded { .. } => 429,
        }
    }

    pub fn default_message(&self) -> String {
        match self {
            Self::RateLimit { limit_type, .. } => {
                format!("Rate limit exceeded: {:?}", limit_type)
            }
            Self::AuthenticationError { message } => message.clone(),
            Self::PermissionDenied { message } => message.clone(),
            Self::BadRequest { message } => message.clone(),
            Self::ContextLengthExceeded { max_tokens, requested } => {
                format!("Context length {} exceeds maximum {}", requested, max_tokens)
            }
            Self::ContentFilter { reason } => {
                format!("Content filtered: {}", reason)
            }
            Self::InternalServerError => "Internal server error".to_string(),
            Self::ServiceUnavailable { .. } => "Service temporarily unavailable".to_string(),
            Self::BadGateway => "Bad gateway".to_string(),
            Self::Timeout { timeout_ms } => format!("Request timeout after {}ms", timeout_ms),
            Self::QuotaExceeded { reset_time } => {
                format!("Quota exceeded, resets at {}", reset_time)
            }
        }
    }

    pub fn retry_after(&self) -> Option<u64> {
        match self {
            Self::RateLimit { retry_after_seconds, .. } => Some(*retry_after_seconds),
            Self::ServiceUnavailable { retry_after_seconds } => *retry_after_seconds,
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InjectedError {
    pub error_type: ErrorType,
    pub http_status: u16,
    pub message: String,
    pub retry_after: Option<u64>,
    pub headers: HashMap<String, String>,
}
```

### 6.3 Circuit Breaker

```rust
// ============================================================================
// Module: errors/circuit_breaker.rs
// Purpose: Circuit breaker pattern for resilience simulation
// ============================================================================

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    config: CircuitBreakerConfig,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure: Arc<RwLock<Option<Instant>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,    // Normal operation
    Open,      // Rejecting requests
    HalfOpen,  // Testing recovery
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u64,
    pub success_threshold: u64,
    pub timeout: Duration,
    pub half_open_requests: u64,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if request should be allowed
    pub async fn allow_request(&self) -> bool {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last) = *self.last_failure.read().await {
                    if last.elapsed() > self.config.timeout {
                        *self.state.write().await = CircuitState::HalfOpen;
                        self.success_count.store(0, Ordering::Relaxed);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests
                self.success_count.load(Ordering::Relaxed) < self.config.half_open_requests
            }
        }
    }

    /// Record request success
    pub async fn record_success(&self) {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    *self.state.write().await = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record request failure
    pub async fn record_failure(&self) {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.config.failure_threshold {
                    *self.state.write().await = CircuitState::Open;
                    *self.last_failure.write().await = Some(Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                *self.state.write().await = CircuitState::Open;
                *self.last_failure.write().await = Some(Instant::now());
            }
            CircuitState::Open => {}
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }
}
```

---

## 7. Configuration Management

### 7.1 Configuration Schema

```rust
// ============================================================================
// Module: config/schema.rs
// Purpose: Complete configuration schema with validation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SimulatorConfig {
    /// Configuration version for migrations
    #[serde(default = "default_version")]
    pub version: String,

    /// Global simulation seed for determinism
    pub seed: Option<u64>,

    /// Server configuration
    #[validate(nested)]
    pub server: ServerConfig,

    /// Provider configurations
    #[validate(nested)]
    pub providers: HashMap<String, ProviderConfig>,

    /// Latency configuration
    #[validate(nested)]
    pub latency: LatencyConfig,

    /// Error injection configuration
    #[validate(nested)]
    pub errors: ErrorConfig,

    /// Session configuration
    #[validate(nested)]
    pub sessions: SessionConfig,

    /// Concurrency configuration
    #[validate(nested)]
    pub concurrency: ConcurrencyConfig,

    /// Telemetry configuration
    #[validate(nested)]
    pub telemetry: TelemetryConfig,

    /// Scenario definitions
    pub scenarios: HashMap<String, ScenarioConfig>,

    /// Graceful shutdown timeout
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[validate(range(min = 1, max = 65535))]
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default)]
    pub tls: Option<TlsConfig>,

    #[serde(default = "default_workers")]
    pub workers: usize,

    #[serde(default)]
    pub cors: CorsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ProviderConfig {
    pub name: String,
    pub provider: String, // "openai", "anthropic", "google"
    pub models: Vec<String>,
    pub enabled: bool,

    #[validate(nested)]
    pub latency: ProviderLatencyConfig,

    #[validate(nested)]
    pub errors: ProviderErrorConfig,

    pub response: ResponseConfig,
    pub cost: Option<CostConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ProviderLatencyConfig {
    pub ttft: DistributionConfig,
    pub itl: DistributionConfig,
    pub degradation: Option<DegradationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DistributionConfig {
    #[serde(rename = "normal")]
    Normal { mean_ms: f64, std_dev_ms: f64 },

    #[serde(rename = "log_normal")]
    LogNormal { p50_ms: f64, p99_ms: f64 },

    #[serde(rename = "exponential")]
    Exponential { rate: f64 },

    #[serde(rename = "constant")]
    Constant { value_ms: f64 },

    #[serde(rename = "empirical")]
    Empirical { samples: Vec<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConcurrencyConfig {
    #[validate(range(min = 1, max = 100000))]
    pub max_concurrent_requests: usize,

    #[validate(range(min = 1, max = 10000))]
    pub connection_pool_size: usize,

    pub request_timeout: Duration,
    pub connection_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TelemetryConfig {
    #[validate(nested)]
    pub logging: LoggingConfig,

    #[validate(nested)]
    pub metrics: MetricsConfig,

    #[validate(nested)]
    pub tracing: TracingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub prometheus: PrometheusConfig,
    pub otlp: Option<OtlpConfig>,
}
```

### 7.2 Configuration Loader

```rust
// ============================================================================
// Module: config/loader.rs
// Purpose: Multi-format configuration loading with hierarchy
// ============================================================================

pub struct ConfigLoader {
    figment: Figment,
}

impl ConfigLoader {
    /// Create loader with default configuration hierarchy
    pub fn new() -> Self {
        let figment = Figment::new()
            // 1. Built-in defaults
            .merge(Serialized::defaults(SimulatorConfig::default()))
            // 2. System-wide config
            .merge(Yaml::file("/etc/llm-simulator/config.yaml"))
            // 3. User config
            .merge(Yaml::file("~/.config/llm-simulator/config.yaml"))
            // 4. Project config
            .merge(Yaml::file("simulator.yaml"))
            .merge(Json::file("simulator.json"))
            .merge(Toml::file("simulator.toml"))
            // 5. Local overrides
            .merge(Yaml::file("simulator.local.yaml"))
            // 6. Environment variables
            .merge(Env::prefixed("LLM_SIM_").split("__"))
            ;

        Self { figment }
    }

    /// Add CLI argument overrides
    pub fn with_cli_overrides(mut self, overrides: HashMap<String, String>) -> Self {
        for (key, value) in overrides {
            self.figment = self.figment.merge(Serialized::default(&key, &value));
        }
        self
    }

    /// Load and validate configuration
    pub fn load(&self) -> Result<SimulatorConfig, ConfigError> {
        let config: SimulatorConfig = self.figment.extract()?;

        // Validate
        config.validate()
            .map_err(|e| ConfigError::ValidationError(format!("{:?}", e)))?;

        Ok(config)
    }

    /// Load with migration from older versions
    pub fn load_with_migration(&self) -> Result<SimulatorConfig, ConfigError> {
        let raw: serde_json::Value = self.figment.extract()?;

        let version = raw["version"].as_str().unwrap_or("1.0.0");
        let migrated = self.migrate(raw, version)?;

        let config: SimulatorConfig = serde_json::from_value(migrated)?;
        config.validate()?;

        Ok(config)
    }

    fn migrate(&self, mut config: serde_json::Value, from_version: &str) -> Result<serde_json::Value, ConfigError> {
        let migrations: Vec<(&str, fn(&mut serde_json::Value))> = vec![
            ("0.9.0", migrate_0_9_to_1_0),
            ("1.0.0", migrate_1_0_to_1_1),
        ];

        for (target_version, migration_fn) in migrations {
            if semver::Version::parse(from_version)? < semver::Version::parse(target_version)? {
                migration_fn(&mut config);
            }
        }

        Ok(config)
    }
}
```

### 7.3 Hot Reload

```rust
// ============================================================================
// Module: config/hot_reload.rs
// Purpose: File watching and configuration hot-reload
// ============================================================================

pub struct ConfigWatcher {
    config_path: PathBuf,
    current_config: Arc<RwLock<SimulatorConfig>>,
    loader: ConfigLoader,
    subscribers: Arc<RwLock<Vec<Box<dyn Fn(&SimulatorConfig) + Send + Sync>>>>,
}

impl ConfigWatcher {
    pub fn new(config_path: PathBuf, initial_config: SimulatorConfig, loader: ConfigLoader) -> Self {
        Self {
            config_path,
            current_config: Arc::new(RwLock::new(initial_config)),
            loader,
            subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start watching for config changes
    pub fn watch(self) -> ConfigWatchHandle {
        let (tx, mut rx) = mpsc::channel(1);

        let config_path = self.config_path.clone();
        let current_config = Arc::clone(&self.current_config);
        let loader = self.loader;
        let subscribers = Arc::clone(&self.subscribers);

        // Spawn file watcher
        let watch_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            let mut last_modified = std::fs::metadata(&config_path)
                .and_then(|m| m.modified())
                .ok();

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Check for file changes
                        if let Ok(metadata) = std::fs::metadata(&config_path) {
                            if let Ok(modified) = metadata.modified() {
                                if last_modified.map_or(true, |last| modified > last) {
                                    last_modified = Some(modified);

                                    // Reload configuration
                                    match loader.load() {
                                        Ok(new_config) => {
                                            *current_config.write().await = new_config.clone();

                                            // Notify subscribers
                                            for subscriber in subscribers.read().await.iter() {
                                                subscriber(&new_config);
                                            }

                                            tracing::info!("Configuration reloaded");
                                        }
                                        Err(e) => {
                                            tracing::error!(error = ?e, "Failed to reload configuration");
                                        }
                                    }
                                }
                            }
                        }
                    }

                    _ = rx.recv() => {
                        tracing::info!("Config watcher stopping");
                        break;
                    }
                }
            }
        });

        ConfigWatchHandle {
            shutdown_tx: tx,
            handle: watch_handle,
        }
    }

    /// Subscribe to configuration changes
    pub async fn subscribe<F>(&self, callback: F)
    where
        F: Fn(&SimulatorConfig) + Send + Sync + 'static,
    {
        self.subscribers.write().await.push(Box::new(callback));
    }

    /// Get current configuration
    pub async fn get_config(&self) -> SimulatorConfig {
        self.current_config.read().await.clone()
    }
}

pub struct ConfigWatchHandle {
    shutdown_tx: mpsc::Sender<()>,
    handle: tokio::task::JoinHandle<()>,
}

impl ConfigWatchHandle {
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(()).await;
        let _ = self.handle.await;
    }
}
```

---

## 8. Telemetry & Observability

### 8.1 Telemetry System

```rust
// ============================================================================
// Module: telemetry/mod.rs
// Purpose: Unified telemetry initialization and management
// ============================================================================

pub struct TelemetrySystem {
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    metrics: Arc<LLMMetrics>,
}

impl TelemetrySystem {
    pub async fn init(config: &TelemetryConfig) -> Result<Self, TelemetryError> {
        // Initialize OpenTelemetry resource
        let resource = Resource::new(vec![
            KeyValue::new("service.name", "llm-simulator"),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        ]);

        // Initialize tracer provider
        let tracer_provider = Self::init_tracer(&config.tracing, &resource)?;

        // Initialize meter provider
        let meter_provider = Self::init_meter(&config.metrics, &resource)?;

        // Initialize logger provider
        let logger_provider = Self::init_logger(&config.logging, &resource)?;

        // Create LLM-specific metrics
        let metrics = Arc::new(LLMMetrics::new(&meter_provider)?);

        Ok(Self {
            tracer_provider,
            meter_provider,
            logger_provider,
            metrics,
        })
    }

    fn init_tracer(config: &TracingConfig, resource: &Resource) -> Result<TracerProvider, TelemetryError> {
        let mut provider_builder = TracerProvider::builder()
            .with_resource(resource.clone());

        if config.enabled {
            if let Some(otlp) = &config.otlp {
                let exporter = opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(&otlp.endpoint)
                    .build_span_exporter()?;

                provider_builder = provider_builder.with_batch_exporter(
                    exporter,
                    opentelemetry_sdk::runtime::Tokio,
                );
            }
        }

        Ok(provider_builder.build())
    }

    fn init_meter(config: &MetricsConfig, resource: &Resource) -> Result<MeterProvider, TelemetryError> {
        let mut provider_builder = MeterProvider::builder()
            .with_resource(resource.clone());

        if config.enabled {
            // Prometheus exporter
            if config.prometheus.enabled {
                let exporter = opentelemetry_prometheus::exporter()
                    .with_registry(prometheus::default_registry().clone())
                    .build()?;

                provider_builder = provider_builder.with_reader(exporter);
            }

            // OTLP exporter
            if let Some(otlp) = &config.otlp {
                let exporter = opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(&otlp.endpoint)
                    .build_metrics_exporter()?;

                provider_builder = provider_builder.with_reader(
                    PeriodicReader::builder(exporter, opentelemetry_sdk::runtime::Tokio)
                        .with_interval(Duration::from_secs(10))
                        .build(),
                );
            }
        }

        Ok(provider_builder.build())
    }

    pub fn metrics(&self) -> Arc<LLMMetrics> {
        Arc::clone(&self.metrics)
    }

    pub async fn shutdown(&self) {
        self.tracer_provider.shutdown().ok();
        self.meter_provider.shutdown().ok();
    }
}
```

### 8.2 LLM Metrics

```rust
// ============================================================================
// Module: telemetry/metrics.rs
// Purpose: LLM-specific metrics collection
// ============================================================================

pub struct LLMMetrics {
    // Request metrics
    request_duration: Histogram<f64>,
    requests_total: Counter<u64>,
    active_requests: UpDownCounter<i64>,

    // Token metrics
    tokens_total: Counter<u64>,
    tokens_per_second: Gauge<f64>,

    // Latency metrics
    ttft_seconds: Histogram<f64>,

    // Error metrics
    errors_total: Counter<u64>,

    // Cost metrics
    cost_dollars: Counter<f64>,
}

impl LLMMetrics {
    pub fn new(meter_provider: &MeterProvider) -> Result<Self, MetricError> {
        let meter = meter_provider.meter("llm_simulator");

        Ok(Self {
            request_duration: meter
                .f64_histogram("llm_simulator_request_duration_seconds")
                .with_description("Request duration in seconds")
                .with_unit("s")
                .init(),

            requests_total: meter
                .u64_counter("llm_simulator_requests_total")
                .with_description("Total number of requests")
                .init(),

            active_requests: meter
                .i64_up_down_counter("llm_simulator_active_requests")
                .with_description("Number of active requests")
                .init(),

            tokens_total: meter
                .u64_counter("llm_simulator_tokens_total")
                .with_description("Total tokens processed")
                .init(),

            tokens_per_second: meter
                .f64_gauge("llm_simulator_tokens_per_second")
                .with_description("Token throughput")
                .init(),

            ttft_seconds: meter
                .f64_histogram("llm_simulator_ttft_seconds")
                .with_description("Time to first token")
                .with_unit("s")
                .init(),

            errors_total: meter
                .u64_counter("llm_simulator_errors_total")
                .with_description("Total errors")
                .init(),

            cost_dollars: meter
                .f64_counter("llm_simulator_cost_dollars")
                .with_description("Simulated cost in dollars")
                .init(),
        })
    }

    pub fn record_request(
        &self,
        provider: &str,
        model: &str,
        duration: Duration,
        prompt_tokens: u64,
        completion_tokens: u64,
        status: &str,
    ) {
        let attributes = [
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("status", status.to_string()),
        ];

        self.request_duration.record(duration.as_secs_f64(), &attributes);
        self.requests_total.add(1, &attributes);

        self.tokens_total.add(prompt_tokens, &[
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("type", "prompt"),
        ]);

        self.tokens_total.add(completion_tokens, &[
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("type", "completion"),
        ]);
    }

    pub fn record_ttft(&self, provider: &str, model: &str, ttft: Duration) {
        self.ttft_seconds.record(ttft.as_secs_f64(), &[
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
        ]);
    }

    pub fn record_error(&self, provider: &str, error_type: &str) {
        self.errors_total.add(1, &[
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("error_type", error_type.to_string()),
        ]);
    }

    pub fn record_cost(&self, provider: &str, model: &str, cost: f64) {
        self.cost_dollars.add(cost, &[
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
        ]);
    }

    pub fn increment_active_requests(&self, provider: &str) {
        self.active_requests.add(1, &[KeyValue::new("provider", provider.to_string())]);
    }

    pub fn decrement_active_requests(&self, provider: &str) {
        self.active_requests.add(-1, &[KeyValue::new("provider", provider.to_string())]);
    }
}
```

### 8.3 Request Tracing

```rust
// ============================================================================
// Module: telemetry/tracing.rs
// Purpose: Distributed tracing for request lifecycle
// ============================================================================

pub struct RequestSpan {
    span: tracing::Span,
    start: Instant,
    correlation_id: String,
}

impl RequestSpan {
    pub fn new(request_id: &str, model: &str, provider: &str) -> Self {
        let correlation_id = Uuid::new_v4().to_string();

        let span = tracing::info_span!(
            "llm_request",
            request_id = %request_id,
            model = %model,
            provider = %provider,
            correlation_id = %correlation_id,
            otel.kind = "server",
        );

        Self {
            span,
            start: Instant::now(),
            correlation_id,
        }
    }

    pub fn enter(&self) -> tracing::span::EnteredSpan {
        self.span.enter()
    }

    pub fn record_tokens(&self, prompt_tokens: u64, completion_tokens: u64) {
        self.span.record("prompt_tokens", prompt_tokens);
        self.span.record("completion_tokens", completion_tokens);
    }

    pub fn record_ttft(&self, ttft: Duration) {
        self.span.record("ttft_ms", ttft.as_millis() as u64);
    }

    pub fn record_error(&self, error: &str) {
        self.span.record("error", true);
        self.span.record("error.message", error);
    }

    pub fn child_span(&self, name: &str) -> tracing::Span {
        tracing::info_span!(
            parent: &self.span,
            "llm_operation",
            operation = %name,
        )
    }

    pub fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
```

---

## 9. HTTP Server & API Layer

### 9.1 Server Setup

```rust
// ============================================================================
// Module: server/mod.rs
// Purpose: High-performance HTTP server with middleware stack
// ============================================================================

pub struct SimulatorServer {
    config: ServerConfig,
    engine: Arc<SimulationEngine>,
    router: Router,
}

impl SimulatorServer {
    pub async fn new(config: ServerConfig, engine: Arc<SimulationEngine>) -> Result<Self, ServerError> {
        let router = Self::create_router(Arc::clone(&engine));

        Ok(Self {
            config,
            engine,
            router,
        })
    }

    fn create_router(engine: Arc<SimulationEngine>) -> Router {
        let state = AppState { engine };

        Router::new()
            // OpenAI-compatible routes
            .route("/v1/chat/completions", post(handlers::chat_completion))
            .route("/v1/completions", post(handlers::text_completion))
            .route("/v1/embeddings", post(handlers::embeddings))
            .route("/v1/models", get(handlers::list_models))
            .route("/v1/models/:model", get(handlers::get_model))

            // Anthropic-compatible routes
            .route("/v1/messages", post(handlers::anthropic_messages))

            // Health and admin routes
            .route("/health", get(handlers::health))
            .route("/ready", get(handlers::ready))
            .route("/metrics", get(handlers::metrics))
            .route("/admin/config", post(handlers::reload_config))
            .route("/admin/stats", get(handlers::stats))
            .route("/admin/scenarios/:name/activate", post(handlers::activate_scenario))

            // Apply middleware
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(TimeoutLayer::new(Duration::from_secs(300)))
                    .layer(CompressionLayer::new())
                    .layer(CorsLayer::permissive())
                    .layer(from_fn(middleware::auth_middleware))
                    .layer(from_fn(middleware::rate_limit_middleware))
                    .layer(from_fn(middleware::metrics_middleware))
            )
            .with_state(state)
    }

    pub async fn run(self) -> Result<(), ServerError> {
        let addr = format!("{}:{}", self.config.host, self.config.port);

        tracing::info!(address = %addr, "Starting LLM Simulator server");

        let listener = tokio::net::TcpListener::bind(&addr).await?;

        axum::serve(listener, self.router)
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<SimulationEngine>,
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received");
}
```

### 9.2 Request Handlers

```rust
// ============================================================================
// Module: server/handlers.rs
// Purpose: API endpoint handlers
// ============================================================================

/// POST /v1/chat/completions - OpenAI compatible
pub async fn chat_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // Validate request
    request.validate().map_err(|e| ApiError::BadRequest(e.to_string()))?;

    // Check if streaming
    if request.stream {
        // Return SSE stream
        let stream = state.engine
            .process_chat_completion_stream(request)
            .await
            .map_err(ApiError::from)?;

        Ok(Sse::new(stream.map(|result| {
            result.map(|chunk| {
                Event::default()
                    .data(serde_json::to_string(&chunk).unwrap())
            })
        }))
        .keep_alive(KeepAlive::default())
        .into_response())
    } else {
        // Return JSON response
        let response = state.engine
            .process_chat_completion(request)
            .await
            .map_err(ApiError::from)?;

        Ok(Json(response).into_response())
    }
}

/// POST /v1/messages - Anthropic compatible
pub async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<AnthropicMessagesRequest>,
) -> Result<Response, ApiError> {
    // Transform Anthropic request to unified format
    let unified = transform_anthropic_to_unified(request)?;

    if unified.stream {
        let stream = state.engine
            .process_chat_completion_stream(unified)
            .await?;

        // Transform stream to Anthropic format
        let anthropic_stream = stream.map(|result| {
            result.map(|chunk| transform_chunk_to_anthropic(chunk))
        });

        Ok(Sse::new(anthropic_stream.map(|result| {
            result.map(|event| Event::default().event(&event.event_type).data(event.data))
        })).into_response())
    } else {
        let response = state.engine.process_chat_completion(unified).await?;
        let anthropic_response = transform_response_to_anthropic(response);

        Ok(Json(anthropic_response).into_response())
    }
}

/// GET /health
pub async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "timestamp": Utc::now().to_rfc3339(),
    }))
}

/// GET /ready
pub async fn ready(State(state): State<AppState>) -> impl IntoResponse {
    let is_ready = state.engine.is_ready().await;

    if is_ready {
        (StatusCode::OK, Json(json!({ "status": "ready" })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "status": "not_ready" })))
    }
}

/// GET /metrics - Prometheus format
pub async fn metrics() -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();

    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            metrics,
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode metrics: {}", e),
        ).into_response(),
    }
}

/// GET /admin/stats
pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.engine.get_stats().await;
    Json(stats)
}

/// POST /admin/config
pub async fn reload_config(
    State(state): State<AppState>,
    Json(config): Json<SimulatorConfig>,
) -> Result<impl IntoResponse, ApiError> {
    state.engine
        .reload_config(config)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(Json(json!({ "status": "reloaded" })))
}
```

### 9.3 Middleware

```rust
// ============================================================================
// Module: server/middleware.rs
// Purpose: Request processing middleware
// ============================================================================

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // Skip auth for health endpoints
    if request.uri().path().starts_with("/health") || request.uri().path().starts_with("/metrics") {
        return Ok(next.run(request).await);
    }

    // Check Authorization header
    let auth_header = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(auth) if auth.starts_with("Bearer ") => {
            let token = &auth[7..];

            // In simulation mode, accept any token that looks valid
            if token.len() >= 20 {
                Ok(next.run(request).await)
            } else {
                Err(ApiError::Unauthorized("Invalid API key format".to_string()))
            }
        }
        Some(auth) if auth.starts_with("x-api-key ") => {
            // Anthropic style
            Ok(next.run(request).await)
        }
        _ => Err(ApiError::Unauthorized("Missing authorization header".to_string())),
    }
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let client_id = extract_client_id(&headers);

    if let Err(retry_after) = state.engine.check_rate_limit(&client_id).await {
        return Err(ApiError::RateLimited(retry_after));
    }

    Ok(next.run(request).await)
}

/// Metrics collection middleware
pub async fn metrics_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status().as_u16();

    state.engine.metrics().record_http_request(
        method.as_str(),
        &path,
        status,
        duration,
    );

    response
}

fn extract_client_id(headers: &HeaderMap) -> String {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "anonymous".to_string())
}
```

### 9.4 Streaming Response

```rust
// ============================================================================
// Module: server/streaming.rs
// Purpose: Server-Sent Events streaming implementation
// ============================================================================

pub struct StreamingResponse {
    request_id: String,
    model: String,
    tokens: Vec<String>,
    timings: StreamingTimings,
}

impl StreamingResponse {
    pub fn into_stream(self) -> impl Stream<Item = Result<Event, Infallible>> {
        async_stream::stream! {
            // Wait for TTFT
            tokio::time::sleep(self.timings.ttft).await;

            // Stream tokens
            for (i, (token, timing)) in self.tokens.iter()
                .zip(self.timings.itls.iter())
                .enumerate()
            {
                let chunk = ChatCompletionChunk {
                    id: self.request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: Utc::now().timestamp(),
                    model: self.model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: if i == 0 { Some("assistant".to_string()) } else { None },
                            content: Some(token.clone()),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };

                yield Ok(Event::default()
                    .data(serde_json::to_string(&chunk).unwrap()));

                tokio::time::sleep(*timing).await;
            }

            // Final chunk
            let final_chunk = ChatCompletionChunk {
                id: self.request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: Utc::now().timestamp(),
                model: self.model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                }],
            };

            yield Ok(Event::default()
                .data(serde_json::to_string(&final_chunk).unwrap()));

            yield Ok(Event::default().data("[DONE]"));
        }
    }
}
```

---

## 10. Load Testing & Concurrency

### 10.1 Load Pattern Trait

```rust
// ============================================================================
// Module: load/patterns.rs
// Purpose: Load pattern implementations for stress testing
// ============================================================================

pub trait LoadPattern: Send + Sync {
    /// Get current target RPS at elapsed time
    fn current_rate(&self, elapsed: Duration) -> f64;

    /// Check if pattern is complete
    fn is_complete(&self, elapsed: Duration) -> bool;

    /// Get total duration (None if infinite)
    fn duration(&self) -> Option<Duration>;

    /// Pattern name
    fn name(&self) -> &str;

    /// Clone for multi-threaded use
    fn clone_box(&self) -> Box<dyn LoadPattern>;
}

/// Steady state - constant RPS
pub struct SteadyPattern {
    pub target_rps: f64,
    pub duration: Option<Duration>,
}

impl LoadPattern for SteadyPattern {
    fn current_rate(&self, _elapsed: Duration) -> f64 {
        self.target_rps
    }

    fn is_complete(&self, elapsed: Duration) -> bool {
        self.duration.map_or(false, |d| elapsed >= d)
    }

    fn duration(&self) -> Option<Duration> { self.duration }
    fn name(&self) -> &str { "steady" }
    fn clone_box(&self) -> Box<dyn LoadPattern> { Box::new(self.clone()) }
}

/// Ramp up - linear increase
pub struct RampUpPattern {
    pub start_rps: f64,
    pub end_rps: f64,
    pub duration: Duration,
}

impl LoadPattern for RampUpPattern {
    fn current_rate(&self, elapsed: Duration) -> f64 {
        if elapsed >= self.duration {
            return self.end_rps;
        }

        let progress = elapsed.as_secs_f64() / self.duration.as_secs_f64();
        self.start_rps + (self.end_rps - self.start_rps) * progress
    }

    fn is_complete(&self, elapsed: Duration) -> bool {
        elapsed >= self.duration
    }

    fn duration(&self) -> Option<Duration> { Some(self.duration) }
    fn name(&self) -> &str { "ramp-up" }
    fn clone_box(&self) -> Box<dyn LoadPattern> { Box::new(self.clone()) }
}

/// Spike - sudden burst
pub struct SpikePattern {
    pub base_rps: f64,
    pub spike_rps: f64,
    pub spike_duration: Duration,
    pub spike_start: Duration,
    pub total_duration: Duration,
}

impl LoadPattern for SpikePattern {
    fn current_rate(&self, elapsed: Duration) -> f64 {
        let spike_end = self.spike_start + self.spike_duration;

        if elapsed >= self.spike_start && elapsed < spike_end {
            self.spike_rps
        } else {
            self.base_rps
        }
    }

    fn is_complete(&self, elapsed: Duration) -> bool {
        elapsed >= self.total_duration
    }

    fn duration(&self) -> Option<Duration> { Some(self.total_duration) }
    fn name(&self) -> &str { "spike" }
    fn clone_box(&self) -> Box<dyn LoadPattern> { Box::new(self.clone()) }
}

/// Wave - sinusoidal variation
pub struct WavePattern {
    pub base_rps: f64,
    pub amplitude: f64,
    pub period: Duration,
    pub duration: Option<Duration>,
}

impl LoadPattern for WavePattern {
    fn current_rate(&self, elapsed: Duration) -> f64 {
        use std::f64::consts::PI;

        let t = elapsed.as_secs_f64();
        let period_sec = self.period.as_secs_f64();
        let phase = (2.0 * PI * t) / period_sec;

        (self.base_rps + self.amplitude * phase.sin()).max(0.0)
    }

    fn is_complete(&self, elapsed: Duration) -> bool {
        self.duration.map_or(false, |d| elapsed >= d)
    }

    fn duration(&self) -> Option<Duration> { self.duration }
    fn name(&self) -> &str { "wave" }
    fn clone_box(&self) -> Box<dyn LoadPattern> { Box::new(self.clone()) }
}

/// Chaos - random variations
pub struct ChaosPattern {
    pub min_rps: f64,
    pub max_rps: f64,
    pub change_interval: Duration,
    pub duration: Option<Duration>,
    pub seed: u64,
}

impl LoadPattern for ChaosPattern {
    fn current_rate(&self, elapsed: Duration) -> f64 {
        // Deterministic pseudo-random based on interval
        let interval = elapsed.as_secs() / self.change_interval.as_secs();
        let mut state = self.seed.wrapping_add(interval);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let rand = (state as f64) / (u64::MAX as f64);
        self.min_rps + (self.max_rps - self.min_rps) * rand
    }

    fn is_complete(&self, elapsed: Duration) -> bool {
        self.duration.map_or(false, |d| elapsed >= d)
    }

    fn duration(&self) -> Option<Duration> { self.duration }
    fn name(&self) -> &str { "chaos" }
    fn clone_box(&self) -> Box<dyn LoadPattern> { Box::new(self.clone()) }
}
```

### 10.2 Load Statistics

```rust
// ============================================================================
// Module: load/stats.rs
// Purpose: Real-time load testing statistics
// ============================================================================

pub struct LoadStats {
    // Counters
    requests_sent: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    requests_timeout: AtomicU64,
    requests_shed: AtomicU64,

    // Latency histogram
    latency_histogram: Arc<RwLock<hdrhistogram::Histogram<u64>>>,

    // Throughput
    current_rps: Arc<RwLock<f64>>,
    peak_rps: AtomicU64,

    // Tokens
    tokens_sent: AtomicU64,
    tokens_received: AtomicU64,

    // Timing
    start_time: Instant,
}

impl LoadStats {
    pub fn new() -> Self {
        Self {
            requests_sent: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_timeout: AtomicU64::new(0),
            requests_shed: AtomicU64::new(0),
            latency_histogram: Arc::new(RwLock::new(
                hdrhistogram::Histogram::new_with_bounds(1, 60_000, 3).unwrap()
            )),
            current_rps: Arc::new(RwLock::new(0.0)),
            peak_rps: AtomicU64::new(0),
            tokens_sent: AtomicU64::new(0),
            tokens_received: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    #[inline]
    pub fn record_request_sent(&self) {
        self.requests_sent.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_request_completed(&self, latency_ms: u64) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut hist) = self.latency_histogram.try_write() {
            let _ = hist.record(latency_ms);
        }
    }

    #[inline]
    pub fn record_request_failed(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn snapshot(&self) -> LoadStatsSnapshot {
        let hist = self.latency_histogram.read().await;

        LoadStatsSnapshot {
            requests_sent: self.requests_sent.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.requests_failed.load(Ordering::Relaxed),
            requests_timeout: self.requests_timeout.load(Ordering::Relaxed),
            requests_shed: self.requests_shed.load(Ordering::Relaxed),

            latency_p50_ms: hist.value_at_percentile(50.0),
            latency_p95_ms: hist.value_at_percentile(95.0),
            latency_p99_ms: hist.value_at_percentile(99.0),
            latency_p999_ms: hist.value_at_percentile(99.9),
            latency_max_ms: hist.max(),
            latency_mean_ms: hist.mean(),

            current_rps: *self.current_rps.read().await,
            peak_rps: f64::from_bits(self.peak_rps.load(Ordering::Relaxed)),

            tokens_sent: self.tokens_sent.load(Ordering::Relaxed),
            tokens_received: self.tokens_received.load(Ordering::Relaxed),

            elapsed: self.start_time.elapsed(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadStatsSnapshot {
    pub requests_sent: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_timeout: u64,
    pub requests_shed: u64,

    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub latency_p99_ms: u64,
    pub latency_p999_ms: u64,
    pub latency_max_ms: u64,
    pub latency_mean_ms: f64,

    pub current_rps: f64,
    pub peak_rps: f64,

    pub tokens_sent: u64,
    pub tokens_received: u64,

    pub elapsed: Duration,
}

impl LoadStatsSnapshot {
    pub fn success_rate(&self) -> f64 {
        if self.requests_sent == 0 { return 0.0; }
        (self.requests_completed as f64 / self.requests_sent as f64) * 100.0
    }

    pub fn error_rate(&self) -> f64 {
        if self.requests_sent == 0 { return 0.0; }
        (self.requests_failed as f64 / self.requests_sent as f64) * 100.0
    }
}
```

### 10.3 Backpressure Controller

```rust
// ============================================================================
// Module: load/backpressure.rs
// Purpose: Adaptive backpressure and load shedding
// ============================================================================

pub struct BackpressureController {
    config: BackpressureConfig,
    stats: Arc<LoadStats>,
    current_pressure: Arc<RwLock<f64>>,
    circuit_state: Arc<RwLock<CircuitState>>,
}

#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    pub queue_depth_threshold: usize,
    pub latency_p99_threshold_ms: u64,
    pub error_rate_threshold: f64,
    pub circuit_breaker_enabled: bool,
    pub load_shedding_enabled: bool,
    pub shed_priority_threshold: u8,
}

impl BackpressureController {
    pub fn new(config: BackpressureConfig, stats: Arc<LoadStats>) -> Self {
        Self {
            config,
            stats,
            current_pressure: Arc::new(RwLock::new(0.0)),
            circuit_state: Arc::new(RwLock::new(CircuitState::Closed)),
        }
    }

    /// Evaluate current backpressure level (0.0-1.0)
    pub async fn evaluate_pressure(&self, queue_depth: usize, snapshot: &LoadStatsSnapshot) -> f64 {
        let mut pressure = 0.0;

        // Queue depth contribution (40%)
        let queue_pressure = (queue_depth as f64 / self.config.queue_depth_threshold as f64).min(1.0);
        pressure += queue_pressure * 0.4;

        // Latency contribution (40%)
        let latency_pressure = (snapshot.latency_p99_ms as f64 / self.config.latency_p99_threshold_ms as f64).min(1.0);
        pressure += latency_pressure * 0.4;

        // Error rate contribution (20%)
        let error_pressure = (snapshot.error_rate() / 100.0 / self.config.error_rate_threshold).min(1.0);
        pressure += error_pressure * 0.2;

        *self.current_pressure.write().await = pressure;
        pressure
    }

    /// Should this request be shed?
    pub async fn should_shed_request(&self, priority: u8) -> bool {
        if !self.config.load_shedding_enabled {
            return false;
        }

        // Check circuit breaker
        if *self.circuit_state.read().await == CircuitState::Open {
            return true;
        }

        let pressure = *self.current_pressure.read().await;

        if pressure < 0.5 {
            return false;
        }

        // Priority-based shedding
        if priority < self.config.shed_priority_threshold {
            let priority_factor = priority as f64 / 255.0;
            let shed_prob = (pressure - 0.5) * 2.0 * (1.0 - priority_factor);

            use rand::Rng;
            rand::thread_rng().gen::<f64>() < shed_prob
        } else {
            false
        }
    }
}
```

---

## 11. Integration Patterns

### 11.1 Module Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LLM-Simulator                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐  │
│  │   HTTP      │────▶│  Simulation │────▶│     Provider            │  │
│  │   Server    │     │   Engine    │     │     Registry            │  │
│  │   (Axum)    │     │             │     │                         │  │
│  └──────┬──────┘     └──────┬──────┘     └───────────┬─────────────┘  │
│         │                   │                         │                │
│         │                   ▼                         ▼                │
│         │           ┌─────────────┐     ┌─────────────────────────┐  │
│         │           │   Latency   │     │  OpenAI  │  Anthropic   │  │
│         │           │   Model     │     │ Provider │   Provider   │  │
│         │           └──────┬──────┘     └───────────────────────────┘  │
│         │                  │                                           │
│         │                  ▼                                           │
│         │           ┌─────────────┐     ┌─────────────────────────┐  │
│         │           │   Error     │────▶│   Telemetry             │  │
│         │           │   Injector  │     │   System                │  │
│         │           └─────────────┘     └───────────┬─────────────┘  │
│         │                                           │                │
│         ▼                                           ▼                │
│  ┌─────────────┐                         ┌─────────────────────────┐  │
│  │Configuration│                         │  Prometheus │  OTLP     │  │
│  │   Manager   │                         │  Exporter   │  Exporter │  │
│  └─────────────┘                         └─────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     External Integrations                                │
├──────────────┬───────────────┬────────────────┬────────────────────────┤
│LLM-Orchestrator│LLM-Edge-Agent│LLM-Analytics-Hub│    LLM-Gateway       │
└──────────────┴───────────────┴────────────────┴────────────────────────┘
```

### 11.2 Request Flow

```
1. HTTP Request arrives at Axum server
2. Middleware stack processes:
   - Authentication validation
   - Rate limit check
   - Metrics recording
3. Handler routes to appropriate provider
4. Simulation Engine:
   a. Acquires concurrency permit
   b. Checks error injection
   c. Gets latency profile
   d. Generates response content
   e. Simulates latency (TTFT + ITL)
   f. Records metrics and traces
5. Response returned (JSON or SSE stream)
```

### 11.3 Configuration Flow

```
1. Load hierarchy:
   - Built-in defaults
   - System config (/etc/llm-simulator/)
   - User config (~/.config/llm-simulator/)
   - Project config (simulator.yaml)
   - Local overrides (simulator.local.yaml)
   - Environment variables (LLM_SIM_*)
   - CLI arguments

2. Validation:
   - Schema validation
   - Cross-field validation
   - Provider profile validation

3. Hot-reload:
   - File watcher monitors config files
   - On change: reload, validate, apply
   - Notify subscribers
```

---

## 12. Production Checklist

### 12.1 Performance Targets

| Metric | Target | Implementation |
|--------|--------|----------------|
| Throughput | 10,000+ RPS | Work-stealing, lock-free stats |
| Latency Overhead | <5ms | Async processing, connection pooling |
| Memory | <500MB | Bounded queues, session TTL |
| Startup | <1s | Lazy initialization |

### 12.2 Reliability Requirements

- [ ] Graceful shutdown with request draining
- [ ] Circuit breaker for cascading failure prevention
- [ ] Backpressure handling with load shedding
- [ ] Hot-reload without restart
- [ ] Health and readiness probes

### 12.3 Observability Requirements

- [ ] OpenTelemetry traces for all requests
- [ ] Prometheus metrics endpoint
- [ ] Structured JSON logging
- [ ] Correlation IDs across all operations
- [ ] LLM-specific metrics (tokens, TTFT, cost)

### 12.4 Security Requirements

- [ ] API key validation (simulated)
- [ ] Rate limiting per client
- [ ] Input validation
- [ ] TLS support
- [ ] Admin endpoint protection

### 12.5 Testing Requirements

- [ ] Unit tests for all modules
- [ ] Integration tests for API compatibility
- [ ] Load tests for throughput validation
- [ ] Chaos tests for resilience
- [ ] Latency accuracy validation

---

## Appendix A: Type Definitions

### Core Types

```rust
// Request/Response types
pub struct ChatCompletionRequest { /* ... */ }
pub struct ChatCompletionResponse { /* ... */ }
pub struct ChatCompletionChunk { /* ... */ }
pub struct Message { /* ... */ }
pub struct Choice { /* ... */ }
pub struct Usage { /* ... */ }

// Error types
pub enum SimulationError { /* ... */ }
pub enum ProviderError { /* ... */ }
pub enum ConfigError { /* ... */ }
pub enum ApiError { /* ... */ }
```

### Configuration Types

```rust
// Full configuration hierarchy
pub struct SimulatorConfig { /* ... */ }
pub struct ServerConfig { /* ... */ }
pub struct ProviderConfig { /* ... */ }
pub struct LatencyConfig { /* ... */ }
pub struct ErrorConfig { /* ... */ }
pub struct TelemetryConfig { /* ... */ }
```

---

## Appendix B: Related Documents

- [LLM-Simulator-Specification.md](./LLM-Simulator-Specification.md) - SPARC Phase 1
- LLM-Simulator-Architecture.md - SPARC Phase 3 (Upcoming)
- LLM-Simulator-Refinement.md - SPARC Phase 4 (Upcoming)
- LLM-Simulator-Completion.md - SPARC Phase 5 (Upcoming)

---

*Document generated as part of the SPARC methodology for LLM-Simulator within the LLM DevOps ecosystem.*
