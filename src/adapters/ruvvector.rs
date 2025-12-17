//! RuvVector Service Adapter
//!
//! Thin HTTP client adapter for integrating with the external ruvvector-service.
//! This adapter discovers the service via `RUVVECTOR_SERVICE_URL` environment variable
//! and provides access to `/query` and `/simulate` endpoints.
//!
//! ## Design Principles
//!
//! - Does NOT own schema, vector logic, or business rules
//! - Acts purely as a transport layer between simulator and ruvvector-service
//! - Uses existing infra utilities for caching and retry
//! - Preserves simulator behavior when service is unavailable (graceful degradation)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use llm_simulator::adapters::ruvvector::{RuvVectorAdapter, RuvVectorConfig};
//!
//! // Adapter discovers service URL from RUVVECTOR_SERVICE_URL env var
//! let adapter = RuvVectorAdapter::from_env()?;
//!
//! // Or configure explicitly
//! let adapter = RuvVectorAdapter::new(RuvVectorConfig {
//!     service_url: "http://localhost:8081".to_string(),
//!     ..Default::default()
//! });
//!
//! // Query vectors
//! let vectors = adapter.query(&query_request).await?;
//!
//! // Run simulation
//! let result = adapter.simulate(&simulate_request).await?;
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::infra::{Cache, CacheConfig, RetryPolicy};

/// Environment variable name for service URL discovery
pub const RUVVECTOR_SERVICE_URL_ENV: &str = "RUVVECTOR_SERVICE_URL";

/// Default timeout for HTTP requests to ruvvector-service
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default cache TTL for query results
pub const DEFAULT_CACHE_TTL_SECS: u64 = 300;

/// Configuration for the RuvVector adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuvVectorConfig {
    /// Base URL of the ruvvector-service (e.g., "http://localhost:8081")
    pub service_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Enable request caching
    pub cache_enabled: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Enable retry on transient failures
    pub retry_enabled: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub retry_base_delay_ms: u64,
    /// Enable health checks before requests
    pub health_check_enabled: bool,
    /// Connect timeout in seconds
    pub connect_timeout_secs: u64,
}

impl Default for RuvVectorConfig {
    fn default() -> Self {
        Self {
            service_url: String::new(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            cache_enabled: true,
            cache_ttl_secs: DEFAULT_CACHE_TTL_SECS,
            retry_enabled: true,
            max_retries: 3,
            retry_base_delay_ms: 100,
            health_check_enabled: false,
            connect_timeout_secs: 5,
        }
    }
}

impl RuvVectorConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Result<Self, RuvVectorError> {
        let service_url = std::env::var(RUVVECTOR_SERVICE_URL_ENV)
            .map_err(|_| RuvVectorError::ConfigError(
                format!("Environment variable {} not set", RUVVECTOR_SERVICE_URL_ENV)
            ))?;

        if service_url.is_empty() {
            return Err(RuvVectorError::ConfigError(
                format!("{} is empty", RUVVECTOR_SERVICE_URL_ENV)
            ));
        }

        let mut config = Self::default();
        config.service_url = service_url;

        // Optional environment overrides
        if let Ok(timeout) = std::env::var("RUVVECTOR_TIMEOUT_SECS") {
            if let Ok(secs) = timeout.parse() {
                config.timeout_secs = secs;
            }
        }

        if let Ok(cache) = std::env::var("RUVVECTOR_CACHE_ENABLED") {
            config.cache_enabled = cache.parse().unwrap_or(true);
        }

        if let Ok(retries) = std::env::var("RUVVECTOR_MAX_RETRIES") {
            if let Ok(n) = retries.parse() {
                config.max_retries = n;
            }
        }

        Ok(config)
    }

    /// Check if the configuration has a valid service URL
    pub fn is_configured(&self) -> bool {
        !self.service_url.is_empty()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), RuvVectorError> {
        if self.service_url.is_empty() {
            return Err(RuvVectorError::ConfigError(
                "service_url is required".to_string()
            ));
        }

        // Validate URL format
        url::Url::parse(&self.service_url)
            .map_err(|e| RuvVectorError::ConfigError(
                format!("Invalid service_url: {}", e)
            ))?;

        Ok(())
    }
}

/// Error type for RuvVector adapter operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum RuvVectorError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Request failed: {status} - {message}")]
    RequestFailed {
        status: u16,
        message: String,
    },

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Not configured: ruvvector-service URL not set")]
    NotConfigured,

    #[error("Mock generators are disabled and RuvVector is unavailable")]
    MocksDisabled,
}

impl RuvVectorError {
    /// Check if the error is a service availability issue
    ///
    /// Returns true for errors that indicate the service is temporarily unavailable
    /// and requests could be retried or circuit breakers should activate.
    pub fn is_service_unavailable(&self) -> bool {
        matches!(
            self,
            RuvVectorError::ServiceUnavailable(_)
                | RuvVectorError::ConnectionFailed(_)
                | RuvVectorError::NotConfigured
        )
    }

    /// Check if the error is a timeout
    ///
    /// Returns true if the operation timed out waiting for a response.
    pub fn is_timeout(&self) -> bool {
        matches!(self, RuvVectorError::Timeout(_))
    }
}

/// Request to query vectors from ruvvector-service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Query text or embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Query vector (alternative to text query)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Maximum number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum similarity threshold (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Namespace or collection to query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Filter metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<serde_json::Value>,
    /// Include vector values in response
    #[serde(default)]
    pub include_vectors: bool,
    /// Include metadata in response
    #[serde(default = "default_true")]
    pub include_metadata: bool,
}

fn default_top_k() -> usize { 10 }
fn default_true() -> bool { true }

impl Default for QueryRequest {
    fn default() -> Self {
        Self {
            query: None,
            vector: None,
            top_k: 10,
            threshold: None,
            namespace: None,
            filter: None,
            include_vectors: false,
            include_metadata: true,
        }
    }
}

impl QueryRequest {
    /// Create a text-based query request
    pub fn text(query: impl Into<String>) -> Self {
        Self {
            query: Some(query.into()),
            ..Default::default()
        }
    }

    /// Create a vector-based query request
    pub fn vector(vector: Vec<f32>) -> Self {
        Self {
            vector: Some(vector),
            ..Default::default()
        }
    }

    /// Set the maximum number of results
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set minimum similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set namespace/collection
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Generate cache key for this request
    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        if let Some(q) = &self.query {
            q.hash(&mut hasher);
        }
        if let Some(v) = &self.vector {
            for f in v {
                f.to_bits().hash(&mut hasher);
            }
        }
        self.top_k.hash(&mut hasher);
        if let Some(ns) = &self.namespace {
            ns.hash(&mut hasher);
        }

        format!("ruvvector:query:{:016x}", hasher.finish())
    }
}

/// Response from vector query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Query results
    pub results: Vec<QueryResult>,
    /// Total number of matches (may be more than returned)
    #[serde(default)]
    pub total_matches: usize,
    /// Query execution time in milliseconds
    #[serde(default)]
    pub query_time_ms: u64,
    /// Namespace queried
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Individual query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Unique identifier
    pub id: String,
    /// Similarity score (0.0 - 1.0)
    pub score: f32,
    /// Vector values (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Associated metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Original text content (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Request for simulation via ruvvector-service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateRequest {
    /// Model identifier
    pub model: String,
    /// Input messages or prompt
    pub input: SimulateInput,
    /// Simulation parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<SimulateParameters>,
    /// Context from vector query (augmented generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<QueryResult>>,
    /// Request identifier for tracing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Input for simulation request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SimulateInput {
    /// Simple text prompt
    Text(String),
    /// Chat messages
    Messages(Vec<SimulateMessage>),
}

/// Message in simulation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateMessage {
    /// Message role (system, user, assistant)
    pub role: String,
    /// Message content
    pub content: String,
}

/// Parameters for simulation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulateParameters {
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Random seed for determinism
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Enable streaming response
    #[serde(default)]
    pub stream: bool,
}

/// Response from simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateResponse {
    /// Generated content
    pub content: String,
    /// Model used
    pub model: String,
    /// Token usage
    #[serde(default)]
    pub usage: SimulateUsage,
    /// Finish reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Request identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Latency in milliseconds
    #[serde(default)]
    pub latency_ms: u64,
}

/// Token usage in simulation response
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulateUsage {
    /// Input/prompt tokens
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Output/completion tokens
    #[serde(default)]
    pub completion_tokens: u32,
    /// Total tokens
    #[serde(default)]
    pub total_tokens: u32,
}

/// Health check response from ruvvector-service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Uptime in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime_secs: Option<u64>,
}

/// Consumer trait for RuvVector service integration
///
/// Implementors can consume data from the ruvvector-service
/// without modifying existing simulator APIs.
#[async_trait]
pub trait RuvVectorConsumer: Send + Sync {
    /// Query vectors from the service
    async fn query(&self, request: &QueryRequest) -> Result<QueryResponse, RuvVectorError>;

    /// Run simulation via the service
    async fn simulate(&self, request: &SimulateRequest) -> Result<SimulateResponse, RuvVectorError>;

    /// Check service health
    async fn health_check(&self) -> Result<HealthResponse, RuvVectorError>;

    /// Check if the adapter is configured and available
    fn is_available(&self) -> bool;
}

/// HTTP client adapter for ruvvector-service
pub struct RuvVectorAdapter {
    /// HTTP client
    client: Client,
    /// Configuration
    config: RuvVectorConfig,
    /// Request cache
    cache: Option<Arc<Cache>>,
    /// Retry policy
    retry_policy: RetryPolicy,
}

impl RuvVectorAdapter {
    /// Create a new adapter with the given configuration
    pub fn new(config: RuvVectorConfig) -> Result<Self, RuvVectorError> {
        config.validate()?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(|e| RuvVectorError::ConfigError(
                format!("Failed to create HTTP client: {}", e)
            ))?;

        let cache = if config.cache_enabled {
            Some(Arc::new(Cache::new(CacheConfig {
                default_ttl: Duration::from_secs(config.cache_ttl_secs),
                max_entries: 1000,
                ..Default::default()
            })))
        } else {
            None
        };

        let retry_policy = if config.retry_enabled {
            RetryPolicy::exponential_with_jitter()
                .max_retries(config.max_retries)
                .base_delay(Duration::from_millis(config.retry_base_delay_ms))
                .max_delay(Duration::from_secs(30))
        } else {
            RetryPolicy::new().max_retries(0)
        };

        info!(
            service_url = %config.service_url,
            cache_enabled = config.cache_enabled,
            retry_enabled = config.retry_enabled,
            "RuvVector adapter initialized"
        );

        Ok(Self {
            client,
            config,
            cache,
            retry_policy,
        })
    }

    /// Create adapter from environment variables
    pub fn from_env() -> Result<Self, RuvVectorError> {
        let config = RuvVectorConfig::from_env()?;
        Self::new(config)
    }

    /// Try to create adapter from environment, returning None if not configured
    pub fn try_from_env() -> Option<Self> {
        match Self::from_env() {
            Ok(adapter) => Some(adapter),
            Err(e) => {
                debug!("RuvVector adapter not configured: {}", e);
                None
            }
        }
    }

    /// Get the service base URL
    pub fn service_url(&self) -> &str {
        &self.config.service_url
    }

    /// Build full URL for an endpoint
    fn endpoint_url(&self, path: &str) -> String {
        let base = self.config.service_url.trim_end_matches('/');
        format!("{}{}", base, path)
    }

    /// Execute a POST request with retry logic
    #[instrument(skip(self, body), fields(endpoint = %endpoint))]
    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R, RuvVectorError> {
        let url = self.endpoint_url(endpoint);

        let result = self.retry_policy.retry(|| async {
            let response = self.client
                .post(&url)
                .json(body)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        RuvVectorError::Timeout(Duration::from_secs(self.config.timeout_secs))
                    } else if e.is_connect() {
                        RuvVectorError::ConnectionFailed(e.to_string())
                    } else {
                        RuvVectorError::RequestFailed {
                            status: 0,
                            message: e.to_string(),
                        }
                    }
                })?;

            let status = response.status();

            if status.is_success() {
                response.json::<R>().await.map_err(|e| {
                    RuvVectorError::InvalidResponse(format!("Failed to parse response: {}", e))
                })
            } else {
                let message = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                Err(self.map_status_to_error(status, message))
            }
        }).await;

        result.map_err(|e| match e {
            crate::infra::RetryError::MaxRetriesExceeded { last_error, .. } => last_error,
            crate::infra::RetryError::BudgetExhausted => {
                RuvVectorError::ServiceUnavailable("Retry budget exhausted".to_string())
            }
            crate::infra::RetryError::OperationFailed(e) => e,
            crate::infra::RetryError::Timeout(d) => RuvVectorError::Timeout(d),
        })
    }

    /// Execute a GET request with retry logic
    #[instrument(skip(self), fields(endpoint = %endpoint))]
    async fn get<R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
    ) -> Result<R, RuvVectorError> {
        let url = self.endpoint_url(endpoint);

        let result = self.retry_policy.retry(|| async {
            let response = self.client
                .get(&url)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        RuvVectorError::Timeout(Duration::from_secs(self.config.timeout_secs))
                    } else if e.is_connect() {
                        RuvVectorError::ConnectionFailed(e.to_string())
                    } else {
                        RuvVectorError::RequestFailed {
                            status: 0,
                            message: e.to_string(),
                        }
                    }
                })?;

            let status = response.status();

            if status.is_success() {
                response.json::<R>().await.map_err(|e| {
                    RuvVectorError::InvalidResponse(format!("Failed to parse response: {}", e))
                })
            } else {
                let message = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                Err(self.map_status_to_error(status, message))
            }
        }).await;

        result.map_err(|e| match e {
            crate::infra::RetryError::MaxRetriesExceeded { last_error, .. } => last_error,
            crate::infra::RetryError::BudgetExhausted => {
                RuvVectorError::ServiceUnavailable("Retry budget exhausted".to_string())
            }
            crate::infra::RetryError::OperationFailed(e) => e,
            crate::infra::RetryError::Timeout(d) => RuvVectorError::Timeout(d),
        })
    }

    /// Map HTTP status to error type
    fn map_status_to_error(&self, status: StatusCode, message: String) -> RuvVectorError {
        match status {
            StatusCode::SERVICE_UNAVAILABLE | StatusCode::BAD_GATEWAY | StatusCode::GATEWAY_TIMEOUT => {
                RuvVectorError::ServiceUnavailable(message)
            }
            StatusCode::REQUEST_TIMEOUT => {
                RuvVectorError::Timeout(Duration::from_secs(self.config.timeout_secs))
            }
            _ => RuvVectorError::RequestFailed {
                status: status.as_u16(),
                message,
            }
        }
    }

    /// Get cache statistics if caching is enabled
    pub fn cache_stats(&self) -> Option<crate::infra::CacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Clear the request cache
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }
}

#[async_trait]
impl RuvVectorConsumer for RuvVectorAdapter {
    #[instrument(skip(self, request))]
    async fn query(&self, request: &QueryRequest) -> Result<QueryResponse, RuvVectorError> {
        // Check cache first
        if let Some(cache) = &self.cache {
            let cache_key = request.cache_key();
            if let Some(cached) = cache.get::<QueryResponse>(&cache_key) {
                debug!(cache_key = %cache_key, "Cache hit for query");
                return Ok(cached);
            }
        }

        let response: QueryResponse = self.post("/query", request).await?;

        // Cache the response
        if let Some(cache) = &self.cache {
            let cache_key = request.cache_key();
            if let Err(e) = cache.set(&cache_key, &response, None) {
                warn!(error = %e, "Failed to cache query response");
            }
        }

        Ok(response)
    }

    #[instrument(skip(self, request), fields(model = %request.model))]
    async fn simulate(&self, request: &SimulateRequest) -> Result<SimulateResponse, RuvVectorError> {
        // Simulation requests are not cached as they should produce varied results
        self.post("/simulate", request).await
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<HealthResponse, RuvVectorError> {
        self.get("/health").await
    }

    fn is_available(&self) -> bool {
        self.config.is_configured()
    }
}

/// Nullable adapter wrapper that gracefully handles missing configuration
///
/// This wrapper allows the simulator to function normally when ruvvector-service
/// is not configured, falling back to built-in mock data generation.
pub struct OptionalRuvVectorAdapter {
    inner: Option<RuvVectorAdapter>,
}

impl OptionalRuvVectorAdapter {
    /// Create from environment, returning a no-op adapter if not configured
    pub fn from_env() -> Self {
        Self {
            inner: RuvVectorAdapter::try_from_env(),
        }
    }

    /// Create from explicit configuration
    pub fn new(config: Option<RuvVectorConfig>) -> Self {
        Self {
            inner: config.and_then(|c| RuvVectorAdapter::new(c).ok()),
        }
    }

    /// Check if the underlying adapter is configured and available
    pub fn is_configured(&self) -> bool {
        self.inner.is_some()
    }

    /// Get reference to the underlying adapter if configured
    pub fn adapter(&self) -> Option<&RuvVectorAdapter> {
        self.inner.as_ref()
    }

    /// Query with fallback to None if not configured
    pub async fn query(&self, request: &QueryRequest) -> Option<Result<QueryResponse, RuvVectorError>> {
        if let Some(adapter) = &self.inner {
            Some(adapter.query(request).await)
        } else {
            None
        }
    }

    /// Query that requires RuvVector to be configured
    ///
    /// Returns `NotConfigured` error instead of None when the adapter is not configured.
    /// Use this when `require_ruvvector: true` in the configuration.
    pub async fn query_required(&self, request: &QueryRequest) -> Result<QueryResponse, RuvVectorError> {
        if let Some(adapter) = &self.inner {
            adapter.query(request).await
        } else {
            Err(RuvVectorError::NotConfigured)
        }
    }

    /// Simulate with fallback to None if not configured
    pub async fn simulate(&self, request: &SimulateRequest) -> Option<Result<SimulateResponse, RuvVectorError>> {
        if let Some(adapter) = &self.inner {
            Some(adapter.simulate(request).await)
        } else {
            None
        }
    }

    /// Simulate that requires RuvVector to be configured
    ///
    /// Returns `NotConfigured` error instead of None when the adapter is not configured.
    /// Use this when `require_ruvvector: true` in the configuration.
    pub async fn simulate_required(&self, request: &SimulateRequest) -> Result<SimulateResponse, RuvVectorError> {
        if let Some(adapter) = &self.inner {
            adapter.simulate(request).await
        } else {
            Err(RuvVectorError::NotConfigured)
        }
    }

    /// Health check with fallback to error if not configured
    pub async fn health_check(&self) -> Result<HealthResponse, RuvVectorError> {
        if let Some(adapter) = &self.inner {
            adapter.health_check().await
        } else {
            Err(RuvVectorError::NotConfigured)
        }
    }
}

impl Default for OptionalRuvVectorAdapter {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RuvVectorConfig::default();
        assert!(config.service_url.is_empty());
        assert!(!config.is_configured());
        assert!(config.cache_enabled);
        assert!(config.retry_enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = RuvVectorConfig::default();

        // Empty URL should fail
        assert!(config.validate().is_err());

        // Invalid URL should fail
        config.service_url = "not-a-url".to_string();
        assert!(config.validate().is_err());

        // Valid URL should pass
        config.service_url = "http://localhost:8081".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_query_request_builder() {
        let request = QueryRequest::text("test query")
            .with_top_k(5)
            .with_threshold(0.8)
            .with_namespace("test-ns");

        assert_eq!(request.query, Some("test query".to_string()));
        assert_eq!(request.top_k, 5);
        assert_eq!(request.threshold, Some(0.8));
        assert_eq!(request.namespace, Some("test-ns".to_string()));
    }

    #[test]
    fn test_query_request_cache_key() {
        let request1 = QueryRequest::text("hello");
        let request2 = QueryRequest::text("hello");
        let request3 = QueryRequest::text("world");

        assert_eq!(request1.cache_key(), request2.cache_key());
        assert_ne!(request1.cache_key(), request3.cache_key());
    }

    #[test]
    fn test_optional_adapter_not_configured() {
        // Clear any existing env var
        std::env::remove_var(RUVVECTOR_SERVICE_URL_ENV);

        let adapter = OptionalRuvVectorAdapter::from_env();
        assert!(!adapter.is_configured());
    }

    #[tokio::test]
    async fn test_optional_adapter_query_when_not_configured() {
        std::env::remove_var(RUVVECTOR_SERVICE_URL_ENV);

        let adapter = OptionalRuvVectorAdapter::from_env();
        let request = QueryRequest::text("test");

        let result = adapter.query(&request).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_optional_adapter_health_when_not_configured() {
        std::env::remove_var(RUVVECTOR_SERVICE_URL_ENV);

        let adapter = OptionalRuvVectorAdapter::from_env();
        let result = adapter.health_check().await;

        assert!(matches!(result, Err(RuvVectorError::NotConfigured)));
    }
}
