//! LLM-Latency-Lens Adapter
//!
//! Thin runtime consumer for latency profiles, throughput data, and cold-start
//! distributions from the LLM-Latency-Lens service.
//!
//! This adapter provides:
//! - Consumption of external latency profiles
//! - Throughput data integration
//! - Cold-start distribution tracking
//!
//! ## Usage
//!
//! ```rust,ignore
//! use llm_simulator::adapters::latency_lens::{LatencyLensConsumer, LatencyLensAdapter};
//!
//! let adapter = LatencyLensAdapter::new();
//! let profile = adapter.consume_latency_profile("gpt-4").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Latency profile data consumed from LLM-Latency-Lens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedLatencyProfile {
    /// Profile identifier
    pub id: String,
    /// Provider name (e.g., "openai", "anthropic")
    pub provider: String,
    /// Model name
    pub model: String,
    /// Time-to-first-token statistics in milliseconds
    pub ttft: LatencyDistributionData,
    /// Inter-token latency statistics in milliseconds
    pub itl: LatencyDistributionData,
    /// Total request latency statistics
    pub total_latency: LatencyDistributionData,
    /// Timestamp when profile was captured
    pub captured_at: u64,
}

/// Statistical distribution data for latency measurements
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyDistributionData {
    /// Minimum observed value (ms)
    pub min_ms: f64,
    /// Maximum observed value (ms)
    pub max_ms: f64,
    /// Mean value (ms)
    pub mean_ms: f64,
    /// Standard deviation (ms)
    pub std_dev_ms: f64,
    /// 50th percentile (ms)
    pub p50_ms: f64,
    /// 95th percentile (ms)
    pub p95_ms: f64,
    /// 99th percentile (ms)
    pub p99_ms: f64,
    /// Sample count
    pub sample_count: u64,
}

/// Throughput data consumed from LLM-Latency-Lens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedThroughputData {
    /// Provider name
    pub provider: String,
    /// Model name
    pub model: String,
    /// Tokens per second (output)
    pub tokens_per_second: f64,
    /// Requests per second
    pub requests_per_second: f64,
    /// Concurrent request capacity
    pub concurrent_capacity: u32,
    /// Measurement window duration
    pub window_duration: Duration,
    /// Timestamp when measured
    pub measured_at: u64,
}

/// Cold-start distribution data consumed from LLM-Latency-Lens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedColdStartData {
    /// Provider name
    pub provider: String,
    /// Model name
    pub model: String,
    /// Cold-start latency distribution
    pub cold_start_latency: LatencyDistributionData,
    /// Warm-start latency distribution (for comparison)
    pub warm_start_latency: LatencyDistributionData,
    /// Estimated cold-start probability (0.0 - 1.0)
    pub cold_start_probability: f64,
    /// Time since last request that triggers cold start
    pub cold_start_threshold: Duration,
}

/// Token event data for streaming analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedTokenEvent {
    /// Request identifier
    pub request_id: String,
    /// Token sequence number
    pub sequence: u32,
    /// Token content
    pub content: String,
    /// Time since request start
    pub elapsed: Duration,
    /// Inter-token latency (time since previous token)
    pub itl: Duration,
}

/// Consumer trait for LLM-Latency-Lens integration
///
/// Implementors consume latency data from the Latency-Lens service
/// without modifying any existing simulator APIs.
#[async_trait]
pub trait LatencyLensConsumer: Send + Sync {
    /// Consume a latency profile for a specific model
    async fn consume_latency_profile(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Option<ConsumedLatencyProfile>, AdapterError>;

    /// Consume throughput data for a specific model
    async fn consume_throughput_data(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Option<ConsumedThroughputData>, AdapterError>;

    /// Consume cold-start distribution data
    async fn consume_cold_start_data(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Option<ConsumedColdStartData>, AdapterError>;

    /// Consume streaming token events for analysis
    async fn consume_token_events(
        &self,
        request_id: &str,
    ) -> Result<Vec<ConsumedTokenEvent>, AdapterError>;

    /// Check if the consumer is connected and healthy
    async fn health_check(&self) -> Result<bool, AdapterError>;
}

/// Error type for adapter operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum AdapterError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Data not found: {0}")]
    NotFound(String),

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Upstream service error: {0}")]
    UpstreamError(String),
}

/// Default implementation that bridges to llm-latency-lens-core types
pub struct LatencyLensAdapter {
    /// Cached latency profiles
    profile_cache: Arc<parking_lot::RwLock<HashMap<String, ConsumedLatencyProfile>>>,
    /// Cached throughput data
    throughput_cache: Arc<parking_lot::RwLock<HashMap<String, ConsumedThroughputData>>>,
    /// Cache TTL
    cache_ttl: Duration,
}

impl LatencyLensAdapter {
    /// Create a new adapter with default settings
    pub fn new() -> Self {
        Self {
            profile_cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            throughput_cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            cache_ttl: Duration::from_secs(300), // 5 minute cache
        }
    }

    /// Create adapter with custom cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Generate cache key for provider/model pair
    fn cache_key(provider: &str, model: &str) -> String {
        format!("{}:{}", provider, model)
    }

    /// Convert from llm-latency-lens-core types to our consumed types
    ///
    /// This bridge function is prepared for when upstream compilation is fixed.
    /// Currently uses placeholder implementation.
    #[allow(dead_code)]
    fn convert_from_upstream_profile(
        provider: &str,
        model: &str,
        ttft_mean: f64,
        itl_mean: f64,
    ) -> ConsumedLatencyProfile {
        // Bridge conversion - ready for upstream types when available
        ConsumedLatencyProfile {
            id: uuid::Uuid::new_v4().to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            ttft: LatencyDistributionData {
                mean_ms: ttft_mean,
                ..Default::default()
            },
            itl: LatencyDistributionData {
                mean_ms: itl_mean,
                ..Default::default()
            },
            total_latency: LatencyDistributionData::default(),
            captured_at: chrono::Utc::now().timestamp_millis() as u64,
        }
    }
}

impl Default for LatencyLensAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LatencyLensConsumer for LatencyLensAdapter {
    async fn consume_latency_profile(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Option<ConsumedLatencyProfile>, AdapterError> {
        let key = Self::cache_key(provider, model);

        // Check cache first
        if let Some(cached) = self.profile_cache.read().get(&key) {
            return Ok(Some(cached.clone()));
        }

        // In a full implementation, this would query llm-latency-lens-core
        // For now, return None to indicate no external profile available
        // The simulator will fall back to its built-in profiles
        Ok(None)
    }

    async fn consume_throughput_data(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Option<ConsumedThroughputData>, AdapterError> {
        let key = Self::cache_key(provider, model);

        // Check cache first
        if let Some(cached) = self.throughput_cache.read().get(&key) {
            return Ok(Some(cached.clone()));
        }

        // Return None - simulator uses its own throughput simulation
        Ok(None)
    }

    async fn consume_cold_start_data(
        &self,
        _provider: &str,
        _model: &str,
    ) -> Result<Option<ConsumedColdStartData>, AdapterError> {
        // Cold-start data consumption
        // Returns None when no external data available
        Ok(None)
    }

    async fn consume_token_events(
        &self,
        _request_id: &str,
    ) -> Result<Vec<ConsumedTokenEvent>, AdapterError> {
        // Token event consumption for streaming analysis
        Ok(Vec::new())
    }

    async fn health_check(&self) -> Result<bool, AdapterError> {
        // Adapter is always "healthy" as it's a thin consumption layer
        // Actual health depends on whether upstream is available
        Ok(true)
    }
}

/// Utility to convert consumed profile to simulator-compatible distribution parameters
impl ConsumedLatencyProfile {
    /// Convert TTFT data to duration with jitter
    pub fn sample_ttft(&self, jitter_factor: f64) -> Duration {
        let base_ms = self.ttft.mean_ms;
        let jitter = self.ttft.std_dev_ms * jitter_factor;
        let adjusted_ms = (base_ms + jitter).max(0.0);
        Duration::from_micros((adjusted_ms * 1000.0) as u64)
    }

    /// Convert ITL data to duration with jitter
    pub fn sample_itl(&self, jitter_factor: f64) -> Duration {
        let base_ms = self.itl.mean_ms;
        let jitter = self.itl.std_dev_ms * jitter_factor;
        let adjusted_ms = (base_ms + jitter).max(0.0);
        Duration::from_micros((adjusted_ms * 1000.0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = LatencyLensAdapter::new();
        assert!(adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_consume_profile_returns_none() {
        let adapter = LatencyLensAdapter::new();
        let result = adapter.consume_latency_profile("openai", "gpt-4").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_consume_throughput_returns_none() {
        let adapter = LatencyLensAdapter::new();
        let result = adapter.consume_throughput_data("openai", "gpt-4").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_cache_key_generation() {
        let key = LatencyLensAdapter::cache_key("openai", "gpt-4");
        assert_eq!(key, "openai:gpt-4");
    }

    #[test]
    fn test_profile_sampling() {
        let profile = ConsumedLatencyProfile {
            id: "test".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4".to_string(),
            ttft: LatencyDistributionData {
                mean_ms: 200.0,
                std_dev_ms: 50.0,
                ..Default::default()
            },
            itl: LatencyDistributionData {
                mean_ms: 30.0,
                std_dev_ms: 10.0,
                ..Default::default()
            },
            total_latency: LatencyDistributionData::default(),
            captured_at: 0,
        };

        let ttft = profile.sample_ttft(0.0);
        assert_eq!(ttft, Duration::from_micros(200_000));

        let itl = profile.sample_itl(0.0);
        assert_eq!(itl, Duration::from_micros(30_000));
    }
}
