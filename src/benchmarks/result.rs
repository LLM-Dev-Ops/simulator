//! Benchmark result types
//!
//! Contains the canonical BenchmarkResult struct used across all benchmark targets.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Canonical benchmark result structure
///
/// This struct follows the standardized benchmark interface used across
/// all 25 benchmark-target repositories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Identifier for the benchmark target
    pub target_id: String,
    /// Metrics collected during the benchmark (flexible JSON structure)
    pub metrics: serde_json::Value,
    /// Timestamp when the benchmark was executed
    pub timestamp: DateTime<Utc>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(target_id: impl Into<String>, metrics: serde_json::Value) -> Self {
        Self {
            target_id: target_id.into(),
            metrics,
            timestamp: Utc::now(),
        }
    }

    /// Create a benchmark result with a specific timestamp
    pub fn with_timestamp(
        target_id: impl Into<String>,
        metrics: serde_json::Value,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            target_id: target_id.into(),
            metrics,
            timestamp,
        }
    }

    /// Get the target ID
    pub fn target_id(&self) -> &str {
        &self.target_id
    }

    /// Get the metrics
    pub fn metrics(&self) -> &serde_json::Value {
        &self.metrics
    }

    /// Get the timestamp
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

/// Builder for constructing BenchmarkResult with common metrics
#[derive(Debug, Default)]
pub struct BenchmarkResultBuilder {
    target_id: String,
    metrics: serde_json::Map<String, serde_json::Value>,
    timestamp: Option<DateTime<Utc>>,
}

impl BenchmarkResultBuilder {
    /// Create a new builder for the given target
    pub fn new(target_id: impl Into<String>) -> Self {
        Self {
            target_id: target_id.into(),
            metrics: serde_json::Map::new(),
            timestamp: None,
        }
    }

    /// Add a metric value
    pub fn metric(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metrics.insert(key.into(), value.into());
        self
    }

    /// Add duration metric in milliseconds
    pub fn duration_ms(self, key: impl Into<String>, duration: std::time::Duration) -> Self {
        self.metric(key, duration.as_millis() as f64)
    }

    /// Add throughput metric (operations per second)
    pub fn throughput(self, ops_per_sec: f64) -> Self {
        self.metric("throughput_ops_per_sec", ops_per_sec)
    }

    /// Add latency metrics
    pub fn latency(self, avg_ms: f64, min_ms: f64, max_ms: f64) -> Self {
        self.metric("latency_avg_ms", avg_ms)
            .metric("latency_min_ms", min_ms)
            .metric("latency_max_ms", max_ms)
    }

    /// Add percentile latencies
    pub fn percentiles(self, p50: f64, p95: f64, p99: f64) -> Self {
        self.metric("latency_p50_ms", p50)
            .metric("latency_p95_ms", p95)
            .metric("latency_p99_ms", p99)
    }

    /// Set a specific timestamp
    pub fn timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Build the final BenchmarkResult
    pub fn build(self) -> BenchmarkResult {
        BenchmarkResult {
            target_id: self.target_id,
            metrics: serde_json::Value::Object(self.metrics),
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult::new(
            "test-target",
            serde_json::json!({
                "duration_ms": 100.5,
                "iterations": 1000
            }),
        );

        assert_eq!(result.target_id(), "test-target");
        assert!(result.metrics().get("duration_ms").is_some());
    }

    #[test]
    fn test_benchmark_result_builder() {
        let result = BenchmarkResultBuilder::new("latency-sim")
            .metric("samples", 10000)
            .duration_ms("total_time", std::time::Duration::from_millis(500))
            .throughput(20000.0)
            .latency(25.5, 10.0, 150.0)
            .percentiles(22.0, 45.0, 95.0)
            .build();

        assert_eq!(result.target_id(), "latency-sim");
        let metrics = result.metrics();
        assert_eq!(metrics["samples"], 10000);
        assert_eq!(metrics["throughput_ops_per_sec"], 20000.0);
    }

    #[test]
    fn test_serialization() {
        let result = BenchmarkResult::new("test", serde_json::json!({"value": 42}));
        let json = serde_json::to_string(&result).unwrap();
        let parsed: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.target_id(), "test");
    }
}
