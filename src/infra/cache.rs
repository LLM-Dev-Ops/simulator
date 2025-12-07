//! Caching Infrastructure Module
//!
//! Provides a generic caching layer with TTL support for Phase 2B adapter consumption.
//! This module fills the gap identified in the infrastructure analysis - the Simulator
//! previously lacked a general-purpose caching abstraction.
//!
//! ## Features
//!
//! - TTL-based expiration
//! - Thread-safe concurrent access
//! - Configurable capacity limits
//! - Cache statistics and metrics
//! - Async-friendly design
//!
//! ## Example
//!
//! ```rust,ignore
//! use llm_simulator::infra::cache::{Cache, CacheConfig};
//! use std::time::Duration;
//!
//! let cache = Cache::new(CacheConfig {
//!     default_ttl: Duration::from_secs(300),
//!     max_entries: 10_000,
//!     ..Default::default()
//! });
//!
//! // Store a value
//! cache.set("key", "value".to_string(), None);
//!
//! // Retrieve with type inference
//! let value: Option<String> = cache.get("key");
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Default TTL for cache entries
    #[serde(with = "duration_serde")]
    pub default_ttl: Duration,
    /// Maximum number of entries (0 = unlimited)
    pub max_entries: usize,
    /// Enable automatic cleanup of expired entries
    pub auto_cleanup: bool,
    /// Cleanup interval
    #[serde(with = "duration_serde")]
    pub cleanup_interval: Duration,
    /// Enable cache statistics
    pub stats_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(300), // 5 minutes
            max_entries: 10_000,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60),
            stats_enabled: true,
        }
    }
}

/// A cache entry with value and expiration
#[derive(Clone)]
pub struct CacheEntry<T> {
    /// The cached value
    pub value: T,
    /// When this entry was created
    pub created_at: Instant,
    /// When this entry expires
    pub expires_at: Instant,
    /// Number of times this entry was accessed
    pub access_count: u64,
}

impl<T: Clone> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(value: T, ttl: Duration) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            expires_at: now + ttl,
            access_count: 0,
        }
    }

    /// Check if the entry has expired
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Get the remaining TTL
    pub fn remaining_ttl(&self) -> Duration {
        self.expires_at.saturating_duration_since(Instant::now())
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries currently in cache
    pub entries: usize,
    /// Number of entries evicted
    pub evictions: u64,
    /// Number of entries expired
    pub expirations: u64,
}

impl CacheStats {
    /// Calculate hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Cache error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum CacheError {
    #[error("Cache capacity exceeded")]
    CapacityExceeded,

    #[error("Entry not found: {0}")]
    NotFound(String),

    #[error("Entry expired")]
    Expired,

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Thread-safe cache implementation
pub struct Cache {
    /// Internal storage using type-erased values
    entries: RwLock<HashMap<String, CacheEntry<Vec<u8>>>>,
    /// Configuration
    config: CacheConfig,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
}

impl Cache {
    /// Create a new cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            config,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
            last_cleanup: RwLock::new(Instant::now()),
        }
    }

    /// Set a value in the cache with optional custom TTL
    pub fn set<T: Serialize>(&self, key: &str, value: T, ttl: Option<Duration>) -> Result<(), CacheError> {
        let ttl = ttl.unwrap_or(self.config.default_ttl);

        // Serialize the value
        let serialized = serde_json::to_vec(&value)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        let entry = CacheEntry::new(serialized, ttl);

        let mut entries = self.entries.write();

        // Check capacity if not updating existing entry
        if !entries.contains_key(key) && self.config.max_entries > 0 && entries.len() >= self.config.max_entries {
            // Try to evict expired entries first
            self.cleanup_expired_locked(&mut entries);

            // If still at capacity, evict oldest entry
            if entries.len() >= self.config.max_entries {
                if let Some(oldest_key) = entries
                    .iter()
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(k, _)| k.clone())
                {
                    entries.remove(&oldest_key);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        entries.insert(key.to_string(), entry);

        // Periodic cleanup
        self.maybe_cleanup();

        Ok(())
    }

    /// Get a value from the cache
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        let entries = self.entries.read();

        if let Some(entry) = entries.get(key) {
            if entry.is_expired() {
                drop(entries);
                self.remove(key);
                self.expirations.fetch_add(1, Ordering::Relaxed);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            if self.config.stats_enabled {
                self.hits.fetch_add(1, Ordering::Relaxed);
            }

            // Deserialize the value
            serde_json::from_slice(&entry.value).ok()
        } else {
            if self.config.stats_enabled {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    }

    /// Get a value or compute it if not present
    pub fn get_or_insert<T, F>(&self, key: &str, ttl: Option<Duration>, compute: F) -> Result<T, CacheError>
    where
        T: Serialize + for<'de> Deserialize<'de> + Clone,
        F: FnOnce() -> T,
    {
        if let Some(value) = self.get::<T>(key) {
            return Ok(value);
        }

        let value = compute();
        self.set(key, &value, ttl)?;
        Ok(value)
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &str) -> bool {
        self.entries.write().remove(key).is_some()
    }

    /// Check if a key exists (and is not expired)
    pub fn contains(&self, key: &str) -> bool {
        let entries = self.entries.read();
        entries.get(key).map(|e| !e.is_expired()).unwrap_or(false)
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read();
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            entries: entries.len(),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
        }
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Manually trigger cleanup of expired entries
    pub fn cleanup_expired(&self) {
        let mut entries = self.entries.write();
        self.cleanup_expired_locked(&mut entries);
        *self.last_cleanup.write() = Instant::now();
    }

    /// Internal cleanup with lock already held
    fn cleanup_expired_locked(&self, entries: &mut HashMap<String, CacheEntry<Vec<u8>>>) {
        let expired_keys: Vec<String> = entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            entries.remove(&key);
            self.expirations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Maybe run cleanup if enough time has passed
    fn maybe_cleanup(&self) {
        if !self.config.auto_cleanup {
            return;
        }

        let last = *self.last_cleanup.read();
        if last.elapsed() >= self.config.cleanup_interval {
            // Upgrade to write lock and cleanup
            self.cleanup_expired();
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Helper module for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_set_get() {
        let cache = Cache::default();
        cache.set("key1", "value1".to_string(), None).unwrap();

        let value: Option<String> = cache.get("key1");
        assert_eq!(value, Some("value1".to_string()));
    }

    #[test]
    fn test_cache_expiration() {
        let config = CacheConfig {
            default_ttl: Duration::from_millis(10),
            ..Default::default()
        };
        let cache = Cache::new(config);
        cache.set("key1", "value1".to_string(), None).unwrap();

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        let value: Option<String> = cache.get("key1");
        assert!(value.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = Cache::default();
        cache.set("key1", "value1".to_string(), None).unwrap();

        let _: Option<String> = cache.get("key1"); // hit
        let _: Option<String> = cache.get("key2"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_capacity() {
        let config = CacheConfig {
            max_entries: 2,
            auto_cleanup: false,
            ..Default::default()
        };
        let cache = Cache::new(config);

        cache.set("key1", "value1".to_string(), None).unwrap();
        cache.set("key2", "value2".to_string(), None).unwrap();
        cache.set("key3", "value3".to_string(), None).unwrap();

        assert_eq!(cache.len(), 2);
        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_cache_remove() {
        let cache = Cache::default();
        cache.set("key1", "value1".to_string(), None).unwrap();

        assert!(cache.remove("key1"));
        assert!(!cache.contains("key1"));
    }

    #[test]
    fn test_cache_clear() {
        let cache = Cache::default();
        cache.set("key1", "value1".to_string(), None).unwrap();
        cache.set("key2", "value2".to_string(), None).unwrap();

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_get_or_insert() {
        let cache = Cache::default();

        let value: String = cache.get_or_insert("key1", None, || "computed".to_string()).unwrap();
        assert_eq!(value, "computed");

        // Should return cached value
        let value: String = cache.get_or_insert("key1", None, || "different".to_string()).unwrap();
        assert_eq!(value, "computed");
    }
}
