//! Phase 2B Infrastructure Integration Module
//!
//! This module provides infrastructure utilities that complement the LLM-Simulator's
//! existing capabilities with caching, retry logic, and enhanced configuration loading.
//! These utilities are designed for consumption by the Phase 2B runtime adapters.
//!
//! ## Modules
//!
//! - `cache`: Generic caching layer with TTL support and multiple backends
//! - `retry`: Retry and backoff utilities for resilient adapter consumption
//! - `config`: Enhanced configuration loading with Infra-style patterns
//!
//! ## Usage
//!
//! ```rust,ignore
//! use llm_simulator::infra::{Cache, RetryPolicy, InfraConfig};
//!
//! // Create a cache for adapter data
//! let cache = Cache::new(CacheConfig::default());
//!
//! // Configure retry policy for upstream calls
//! let retry = RetryPolicy::exponential()
//!     .max_retries(3)
//!     .base_delay(Duration::from_millis(100));
//! ```

pub mod cache;
pub mod retry;

pub use cache::{Cache, CacheConfig, CacheEntry, CacheError, CacheStats};
pub use retry::{RetryPolicy, RetryConfig, RetryError, Backoff};

use std::sync::Arc;
use parking_lot::RwLock;

/// Unified infrastructure context for Phase 2B integrations
///
/// Provides shared access to caching and retry utilities for all adapters
#[derive(Clone)]
pub struct InfraContext {
    /// Shared cache instance
    cache: Arc<Cache>,
    /// Default retry policy
    retry_policy: RetryPolicy,
    /// Configuration
    config: Arc<InfraContextConfig>,
}

/// Configuration for InfraContext
#[derive(Debug, Clone)]
pub struct InfraContextConfig {
    /// Enable caching
    pub cache_enabled: bool,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Enable distributed tracing correlation
    pub trace_correlation: bool,
}

impl Default for InfraContextConfig {
    fn default() -> Self {
        Self {
            cache_enabled: true,
            cache: CacheConfig::default(),
            retry: RetryConfig::default(),
            trace_correlation: true,
        }
    }
}

impl InfraContext {
    /// Create a new infrastructure context with default configuration
    pub fn new() -> Self {
        Self::with_config(InfraContextConfig::default())
    }

    /// Create a new infrastructure context with custom configuration
    pub fn with_config(config: InfraContextConfig) -> Self {
        Self {
            cache: Arc::new(Cache::new(config.cache.clone())),
            retry_policy: RetryPolicy::from_config(&config.retry),
            config: Arc::new(config),
        }
    }

    /// Get a reference to the cache
    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    /// Get the default retry policy
    pub fn retry_policy(&self) -> &RetryPolicy {
        &self.retry_policy
    }

    /// Check if caching is enabled
    pub fn cache_enabled(&self) -> bool {
        self.config.cache_enabled
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear all cached data
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for InfraContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe shared infrastructure context
pub type SharedInfraContext = Arc<RwLock<InfraContext>>;

/// Create a new shared infrastructure context
pub fn shared_infra_context() -> SharedInfraContext {
    Arc::new(RwLock::new(InfraContext::new()))
}

/// Create a shared infrastructure context with custom configuration
pub fn shared_infra_context_with_config(config: InfraContextConfig) -> SharedInfraContext {
    Arc::new(RwLock::new(InfraContext::with_config(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infra_context_creation() {
        let ctx = InfraContext::new();
        assert!(ctx.cache_enabled());
    }

    #[test]
    fn test_shared_infra_context() {
        let shared = shared_infra_context();
        let guard = shared.read();
        assert!(guard.cache_enabled());
    }

    #[test]
    fn test_custom_config() {
        let config = InfraContextConfig {
            cache_enabled: false,
            ..Default::default()
        };
        let ctx = InfraContext::with_config(config);
        assert!(!ctx.cache_enabled());
    }
}
