//! Benchmark adapters module
//!
//! Implements the canonical BenchTarget trait and provides adapters
//! for all simulator benchmark targets.

use std::time::{Duration, Instant};

use crate::config::{LatencyConfig, LatencyDistribution, GenerationConfig, GenerationStrategy};
use crate::engine::{ResponseGenerator, SimulationEngine};
use crate::latency::{DistributionSampler, LatencySimulator, LatencyStats};
use crate::types::Message;
use crate::SimulatorConfig;

use super::{BenchmarkResult, BenchmarkResultBuilder};

/// Canonical benchmark target trait
///
/// All benchmark targets must implement this trait to be included
/// in the benchmark suite. This follows the standardized interface
/// used across all 25 benchmark-target repositories.
pub trait BenchTarget: Send + Sync {
    /// Returns the unique identifier for this benchmark target
    fn id(&self) -> &str;

    /// Runs the benchmark and returns the result
    fn run(&self) -> BenchmarkResult;

    /// Optional description of what this benchmark measures
    fn description(&self) -> &str {
        ""
    }
}

/// Returns all registered benchmark targets
///
/// This is the canonical registry function that provides access to
/// all available benchmark targets in the system.
pub fn all_targets() -> Vec<Box<dyn BenchTarget>> {
    vec![
        // Latency simulation benchmarks
        Box::new(LatencySamplingBenchmark::new()),
        Box::new(TtftSamplingBenchmark::new()),
        Box::new(ItlSamplingBenchmark::new()),
        Box::new(ScheduleGenerationBenchmark::new()),
        Box::new(DistributionSamplingBenchmark::new()),
        // Token throughput benchmarks
        Box::new(TokenThroughputBenchmark::new()),
        Box::new(ResponseGenerationBenchmark::new()),
        Box::new(TokenizationBenchmark::new()),
        // Model stress benchmarks
        Box::new(ConcurrentRequestBenchmark::new()),
        Box::new(LargeContextBenchmark::new()),
        // Memory profile benchmarks
        Box::new(EmbeddingGenerationBenchmark::new()),
        Box::new(BatchEmbeddingBenchmark::new()),
        // Profile lookup benchmarks
        Box::new(ProfileLookupBenchmark::new()),
    ]
}

// =============================================================================
// Latency Simulation Benchmarks
// =============================================================================

/// Benchmark for overall latency sampling performance
pub struct LatencySamplingBenchmark {
    iterations: usize,
    seed: u64,
}

impl LatencySamplingBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 100_000,
            seed: 42,
        }
    }
}

impl BenchTarget for LatencySamplingBenchmark {
    fn id(&self) -> &str {
        "latency-sampling"
    }

    fn description(&self) -> &str {
        "Measures latency distribution sampling throughput"
    }

    fn run(&self) -> BenchmarkResult {
        let config = LatencyConfig::default();
        let simulator = LatencySimulator::with_seed(config, self.seed);

        let start = Instant::now();
        let mut samples = Vec::with_capacity(self.iterations);

        for _ in 0..self.iterations {
            let ttft = simulator.sample_ttft(Some("standard"));
            samples.push(ttft.as_micros() as f64 / 1000.0);
        }

        let elapsed = start.elapsed();
        let ops_per_sec = self.iterations as f64 / elapsed.as_secs_f64();
        let stats = LatencyStats::from_samples(&samples);

        BenchmarkResultBuilder::new(self.id())
            .metric("iterations", self.iterations)
            .duration_ms("total_duration", elapsed)
            .throughput(ops_per_sec)
            .latency(stats.mean_ms, stats.min_ms, stats.max_ms)
            .percentiles(stats.p50_ms, stats.p95_ms, stats.p99_ms)
            .build()
    }
}

/// Benchmark for TTFT (Time-To-First-Token) sampling
pub struct TtftSamplingBenchmark {
    iterations: usize,
    seed: u64,
}

impl TtftSamplingBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 50_000,
            seed: 42,
        }
    }
}

impl BenchTarget for TtftSamplingBenchmark {
    fn id(&self) -> &str {
        "ttft-sampling"
    }

    fn description(&self) -> &str {
        "Measures Time-To-First-Token sampling performance across profiles"
    }

    fn run(&self) -> BenchmarkResult {
        let config = LatencyConfig::default();
        let simulator = LatencySimulator::with_seed(config, self.seed);
        let profiles = ["fast", "standard", "slow", "gpt4", "claude"];

        let start = Instant::now();
        let mut profile_stats = serde_json::Map::new();

        for profile in &profiles {
            let mut samples = Vec::with_capacity(self.iterations / profiles.len());
            for _ in 0..(self.iterations / profiles.len()) {
                let ttft = simulator.sample_ttft(Some(profile));
                samples.push(ttft.as_micros() as f64 / 1000.0);
            }
            let stats = LatencyStats::from_samples(&samples);
            profile_stats.insert(
                profile.to_string(),
                serde_json::json!({
                    "mean_ms": stats.mean_ms,
                    "p95_ms": stats.p95_ms
                }),
            );
        }

        let elapsed = start.elapsed();
        let ops_per_sec = self.iterations as f64 / elapsed.as_secs_f64();

        BenchmarkResultBuilder::new(self.id())
            .metric("iterations", self.iterations)
            .duration_ms("total_duration", elapsed)
            .throughput(ops_per_sec)
            .metric("profile_stats", serde_json::Value::Object(profile_stats))
            .build()
    }
}

/// Benchmark for ITL (Inter-Token-Latency) sampling
pub struct ItlSamplingBenchmark {
    iterations: usize,
    seed: u64,
}

impl ItlSamplingBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 50_000,
            seed: 42,
        }
    }
}

impl BenchTarget for ItlSamplingBenchmark {
    fn id(&self) -> &str {
        "itl-sampling"
    }

    fn description(&self) -> &str {
        "Measures Inter-Token-Latency sampling performance"
    }

    fn run(&self) -> BenchmarkResult {
        let config = LatencyConfig::default();
        let simulator = LatencySimulator::with_seed(config, self.seed);

        let start = Instant::now();
        let mut samples = Vec::with_capacity(self.iterations);

        for _ in 0..self.iterations {
            let itl = simulator.sample_itl(Some("standard"));
            samples.push(itl.as_micros() as f64 / 1000.0);
        }

        let elapsed = start.elapsed();
        let ops_per_sec = self.iterations as f64 / elapsed.as_secs_f64();
        let stats = LatencyStats::from_samples(&samples);

        BenchmarkResultBuilder::new(self.id())
            .metric("iterations", self.iterations)
            .duration_ms("total_duration", elapsed)
            .throughput(ops_per_sec)
            .latency(stats.mean_ms, stats.min_ms, stats.max_ms)
            .percentiles(stats.p50_ms, stats.p95_ms, stats.p99_ms)
            .build()
    }
}

/// Benchmark for latency schedule generation
pub struct ScheduleGenerationBenchmark {
    iterations: usize,
    token_counts: Vec<usize>,
    seed: u64,
}

impl ScheduleGenerationBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 1_000,
            token_counts: vec![100, 500, 1000, 2000],
            seed: 42,
        }
    }
}

impl BenchTarget for ScheduleGenerationBenchmark {
    fn id(&self) -> &str {
        "schedule-generation"
    }

    fn description(&self) -> &str {
        "Measures latency schedule generation for streaming responses"
    }

    fn run(&self) -> BenchmarkResult {
        let config = LatencyConfig::default();
        let simulator = LatencySimulator::with_seed(config, self.seed);

        let mut token_results = serde_json::Map::new();
        let start = Instant::now();
        let mut total_ops = 0;

        for &token_count in &self.token_counts {
            let iter_start = Instant::now();
            for _ in 0..self.iterations {
                let _schedule = simulator.generate_schedule(token_count, Some("standard"));
            }
            let iter_elapsed = iter_start.elapsed();
            total_ops += self.iterations;

            token_results.insert(
                format!("tokens_{}", token_count),
                serde_json::json!({
                    "iterations": self.iterations,
                    "duration_ms": iter_elapsed.as_millis(),
                    "ops_per_sec": self.iterations as f64 / iter_elapsed.as_secs_f64()
                }),
            );
        }

        let elapsed = start.elapsed();
        let total_ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

        BenchmarkResultBuilder::new(self.id())
            .metric("total_iterations", total_ops)
            .duration_ms("total_duration", elapsed)
            .throughput(total_ops_per_sec)
            .metric("token_results", serde_json::Value::Object(token_results))
            .build()
    }
}

/// Benchmark for distribution sampling across all distribution types
pub struct DistributionSamplingBenchmark {
    samples_per_dist: usize,
    seed: u64,
}

impl DistributionSamplingBenchmark {
    pub fn new() -> Self {
        Self {
            samples_per_dist: 10_000,
            seed: 42,
        }
    }
}

impl BenchTarget for DistributionSamplingBenchmark {
    fn id(&self) -> &str {
        "distribution-sampling"
    }

    fn description(&self) -> &str {
        "Measures sampling performance across all distribution types"
    }

    fn run(&self) -> BenchmarkResult {
        let sampler = DistributionSampler::with_seed(self.seed);

        let distributions = [
            ("fixed", LatencyDistribution::Fixed { value_ms: 100.0 }),
            (
                "normal",
                LatencyDistribution::Normal {
                    mean_ms: 100.0,
                    std_dev_ms: 20.0,
                },
            ),
            (
                "log_normal",
                LatencyDistribution::LogNormal {
                    mean_ms: 100.0,
                    std_dev_ms: 50.0,
                },
            ),
            (
                "uniform",
                LatencyDistribution::Uniform {
                    min_ms: 50.0,
                    max_ms: 150.0,
                },
            ),
            (
                "exponential",
                LatencyDistribution::Exponential { mean_ms: 100.0 },
            ),
            (
                "pareto",
                LatencyDistribution::Pareto {
                    scale_ms: 10.0,
                    shape: 2.0,
                },
            ),
        ];

        let mut dist_results = serde_json::Map::new();
        let start = Instant::now();
        let mut total_samples = 0;

        for (name, dist) in &distributions {
            let iter_start = Instant::now();
            let samples = sampler.sample_n(dist, self.samples_per_dist);
            let iter_elapsed = iter_start.elapsed();
            total_samples += samples.len();

            let stats = LatencyStats::from_samples(&samples);
            dist_results.insert(
                name.to_string(),
                serde_json::json!({
                    "samples": samples.len(),
                    "duration_ms": iter_elapsed.as_millis(),
                    "mean_ms": stats.mean_ms,
                    "std_dev_ms": stats.std_dev_ms
                }),
            );
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("total_samples", total_samples)
            .duration_ms("total_duration", elapsed)
            .throughput(total_samples as f64 / elapsed.as_secs_f64())
            .metric("distributions", serde_json::Value::Object(dist_results))
            .build()
    }
}

// =============================================================================
// Token Throughput Benchmarks
// =============================================================================

/// Benchmark for token throughput emulation
pub struct TokenThroughputBenchmark {
    iterations: usize,
    token_counts: Vec<u32>,
    seed: u64,
}

impl TokenThroughputBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 1_000,
            token_counts: vec![100, 500, 1000, 2000],
            seed: 42,
        }
    }
}

impl BenchTarget for TokenThroughputBenchmark {
    fn id(&self) -> &str {
        "token-throughput"
    }

    fn description(&self) -> &str {
        "Measures token generation throughput"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);
        let messages = vec![Message::user("Generate a response for testing purposes.")];
        let config = GenerationConfig::default();

        let mut token_results = serde_json::Map::new();
        let start = Instant::now();
        let mut total_tokens = 0u64;

        for &max_tokens in &self.token_counts {
            let iter_start = Instant::now();
            let mut tokens_generated = 0u64;

            for _ in 0..self.iterations {
                let (_, tokens) = generator.generate_response(&messages, max_tokens, &config);
                tokens_generated += tokens as u64;
            }

            let iter_elapsed = iter_start.elapsed();
            total_tokens += tokens_generated;

            token_results.insert(
                format!("max_tokens_{}", max_tokens),
                serde_json::json!({
                    "iterations": self.iterations,
                    "tokens_generated": tokens_generated,
                    "duration_ms": iter_elapsed.as_millis(),
                    "tokens_per_sec": tokens_generated as f64 / iter_elapsed.as_secs_f64()
                }),
            );
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("total_iterations", self.iterations * self.token_counts.len())
            .metric("total_tokens", total_tokens)
            .duration_ms("total_duration", elapsed)
            .throughput(total_tokens as f64 / elapsed.as_secs_f64())
            .metric("token_results", serde_json::Value::Object(token_results))
            .build()
    }
}

/// Benchmark for response generation
pub struct ResponseGenerationBenchmark {
    iterations: usize,
    seed: u64,
}

impl ResponseGenerationBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 5_000,
            seed: 42,
        }
    }
}

impl BenchTarget for ResponseGenerationBenchmark {
    fn id(&self) -> &str {
        "response-generation"
    }

    fn description(&self) -> &str {
        "Measures response generation across different strategies"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);
        let messages = vec![Message::user("Explain quantum computing.")];

        let strategies = [
            ("template", GenerationStrategy::Template),
            ("lorem", GenerationStrategy::Lorem),
            ("echo", GenerationStrategy::Echo),
            ("random", GenerationStrategy::Random),
        ];

        let mut strategy_results = serde_json::Map::new();
        let start = Instant::now();

        for (name, strategy) in &strategies {
            let config = GenerationConfig {
                strategy: strategy.clone(),
                min_tokens: 50,
                max_tokens: 200,
                ..Default::default()
            };

            let iter_start = Instant::now();
            let mut total_chars = 0;

            for _ in 0..self.iterations {
                let (response, _) = generator.generate_response(&messages, 200, &config);
                total_chars += response.len();
            }

            let iter_elapsed = iter_start.elapsed();

            strategy_results.insert(
                name.to_string(),
                serde_json::json!({
                    "iterations": self.iterations,
                    "total_chars": total_chars,
                    "duration_ms": iter_elapsed.as_millis(),
                    "responses_per_sec": self.iterations as f64 / iter_elapsed.as_secs_f64()
                }),
            );
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("total_iterations", self.iterations * strategies.len())
            .duration_ms("total_duration", elapsed)
            .metric("strategies", serde_json::Value::Object(strategy_results))
            .build()
    }
}

/// Benchmark for tokenization performance
pub struct TokenizationBenchmark {
    iterations: usize,
    seed: u64,
}

impl TokenizationBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 10_000,
            seed: 42,
        }
    }
}

impl BenchTarget for TokenizationBenchmark {
    fn id(&self) -> &str {
        "tokenization"
    }

    fn description(&self) -> &str {
        "Measures text tokenization performance for streaming"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);
        let test_texts = [
            "Hello, world!",
            "This is a longer test sentence with multiple words and punctuation.",
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
            "In the realm of artificial intelligence, large language models have revolutionized how we interact with computers and process natural language.",
        ];

        let start = Instant::now();
        let mut total_tokens = 0;
        let mut total_chars = 0;

        for text in &test_texts {
            for _ in 0..(self.iterations / test_texts.len()) {
                let tokens = generator.tokenize(text);
                total_tokens += tokens.len();
                total_chars += text.len();
            }
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("iterations", self.iterations)
            .metric("total_tokens", total_tokens)
            .metric("total_chars", total_chars)
            .duration_ms("total_duration", elapsed)
            .throughput(total_chars as f64 / elapsed.as_secs_f64())
            .metric("avg_chars_per_token", total_chars as f64 / total_tokens as f64)
            .build()
    }
}

// =============================================================================
// Model Stress Benchmarks
// =============================================================================

/// Benchmark for concurrent request handling (simulated)
pub struct ConcurrentRequestBenchmark {
    total_requests: usize,
    seed: u64,
}

impl ConcurrentRequestBenchmark {
    pub fn new() -> Self {
        Self {
            total_requests: 10_000,
            seed: 42,
        }
    }
}

impl BenchTarget for ConcurrentRequestBenchmark {
    fn id(&self) -> &str {
        "concurrent-requests"
    }

    fn description(&self) -> &str {
        "Simulates concurrent request processing stress"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);
        let config = GenerationConfig::default();
        let latency_config = LatencyConfig::default();
        let latency_sim = LatencySimulator::with_seed(latency_config, self.seed);

        let messages = vec![Message::user("Test request")];

        let start = Instant::now();
        let mut latencies = Vec::with_capacity(self.total_requests);

        for _ in 0..self.total_requests {
            let req_start = Instant::now();

            // Simulate request processing
            let _response = generator.generate_response(&messages, 100, &config);
            let _ttft = latency_sim.sample_ttft(Some("standard"));

            latencies.push(req_start.elapsed().as_micros() as f64 / 1000.0);
        }

        let elapsed = start.elapsed();
        let stats = LatencyStats::from_samples(&latencies);

        BenchmarkResultBuilder::new(self.id())
            .metric("total_requests", self.total_requests)
            .duration_ms("total_duration", elapsed)
            .throughput(self.total_requests as f64 / elapsed.as_secs_f64())
            .latency(stats.mean_ms, stats.min_ms, stats.max_ms)
            .percentiles(stats.p50_ms, stats.p95_ms, stats.p99_ms)
            .build()
    }
}

/// Benchmark for large context handling
pub struct LargeContextBenchmark {
    iterations: usize,
    context_sizes: Vec<usize>,
    seed: u64,
}

impl LargeContextBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 100,
            context_sizes: vec![1000, 5000, 10000, 50000],
            seed: 42,
        }
    }
}

impl BenchTarget for LargeContextBenchmark {
    fn id(&self) -> &str {
        "large-context"
    }

    fn description(&self) -> &str {
        "Measures handling of large context windows"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);
        let config = GenerationConfig::default();

        let mut context_results = serde_json::Map::new();
        let start = Instant::now();

        for &size in &self.context_sizes {
            // Create large context
            let large_content = "x".repeat(size);
            let messages = vec![
                Message::system("You are a helpful assistant."),
                Message::user(&large_content),
            ];

            let iter_start = Instant::now();

            for _ in 0..self.iterations {
                let _response = generator.generate_response(&messages, 100, &config);
            }

            let iter_elapsed = iter_start.elapsed();

            context_results.insert(
                format!("context_{}", size),
                serde_json::json!({
                    "iterations": self.iterations,
                    "context_chars": size,
                    "duration_ms": iter_elapsed.as_millis(),
                    "ops_per_sec": self.iterations as f64 / iter_elapsed.as_secs_f64()
                }),
            );
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("total_iterations", self.iterations * self.context_sizes.len())
            .duration_ms("total_duration", elapsed)
            .metric("context_results", serde_json::Value::Object(context_results))
            .build()
    }
}

// =============================================================================
// Memory Profile Benchmarks
// =============================================================================

/// Benchmark for embedding generation
pub struct EmbeddingGenerationBenchmark {
    iterations: usize,
    dimensions: usize,
    seed: u64,
}

impl EmbeddingGenerationBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 10_000,
            dimensions: 1536,
            seed: 42,
        }
    }
}

impl BenchTarget for EmbeddingGenerationBenchmark {
    fn id(&self) -> &str {
        "embedding-generation"
    }

    fn description(&self) -> &str {
        "Measures embedding vector generation performance"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);
        let inputs = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence and machine learning",
            "Enterprise software development practices",
        ];

        let start = Instant::now();
        let mut total_elements = 0;

        for input in &inputs {
            for _ in 0..(self.iterations / inputs.len()) {
                let embedding = generator.generate_embedding(self.dimensions, input);
                total_elements += embedding.len();
            }
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("iterations", self.iterations)
            .metric("dimensions", self.dimensions)
            .metric("total_elements", total_elements)
            .duration_ms("total_duration", elapsed)
            .throughput(self.iterations as f64 / elapsed.as_secs_f64())
            .metric("elements_per_sec", total_elements as f64 / elapsed.as_secs_f64())
            .build()
    }
}

/// Benchmark for batch embedding generation
pub struct BatchEmbeddingBenchmark {
    batch_sizes: Vec<usize>,
    iterations_per_batch: usize,
    dimensions: usize,
    seed: u64,
}

impl BatchEmbeddingBenchmark {
    pub fn new() -> Self {
        Self {
            batch_sizes: vec![1, 5, 10, 50, 100],
            iterations_per_batch: 100,
            dimensions: 1536,
            seed: 42,
        }
    }
}

impl BenchTarget for BatchEmbeddingBenchmark {
    fn id(&self) -> &str {
        "batch-embedding"
    }

    fn description(&self) -> &str {
        "Measures batch embedding generation performance"
    }

    fn run(&self) -> BenchmarkResult {
        let generator = ResponseGenerator::with_seed(self.seed);

        let mut batch_results = serde_json::Map::new();
        let start = Instant::now();
        let mut total_embeddings = 0;

        for &batch_size in &self.batch_sizes {
            let inputs: Vec<String> = (0..batch_size)
                .map(|i| format!("Input text number {} for embedding", i))
                .collect();

            let iter_start = Instant::now();

            for _ in 0..self.iterations_per_batch {
                for input in &inputs {
                    let _embedding = generator.generate_embedding(self.dimensions, input);
                    total_embeddings += 1;
                }
            }

            let iter_elapsed = iter_start.elapsed();
            let embeddings_in_batch = batch_size * self.iterations_per_batch;

            batch_results.insert(
                format!("batch_{}", batch_size),
                serde_json::json!({
                    "batch_size": batch_size,
                    "iterations": self.iterations_per_batch,
                    "total_embeddings": embeddings_in_batch,
                    "duration_ms": iter_elapsed.as_millis(),
                    "embeddings_per_sec": embeddings_in_batch as f64 / iter_elapsed.as_secs_f64()
                }),
            );
        }

        let elapsed = start.elapsed();

        BenchmarkResultBuilder::new(self.id())
            .metric("total_embeddings", total_embeddings)
            .metric("dimensions", self.dimensions)
            .duration_ms("total_duration", elapsed)
            .throughput(total_embeddings as f64 / elapsed.as_secs_f64())
            .metric("batch_results", serde_json::Value::Object(batch_results))
            .build()
    }
}

/// Benchmark for latency profile lookup
pub struct ProfileLookupBenchmark {
    iterations: usize,
}

impl ProfileLookupBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 100_000,
        }
    }
}

impl BenchTarget for ProfileLookupBenchmark {
    fn id(&self) -> &str {
        "profile-lookup"
    }

    fn description(&self) -> &str {
        "Measures latency profile lookup performance"
    }

    fn run(&self) -> BenchmarkResult {
        let config = LatencyConfig::default();
        let simulator = LatencySimulator::new(config);
        let profiles = ["fast", "standard", "slow", "gpt4", "claude", "gemini", "instant"];
        let nonexistent = "nonexistent_profile";

        let start = Instant::now();

        // Benchmark hits
        let hit_start = Instant::now();
        for _ in 0..self.iterations {
            for profile in &profiles {
                let _ = simulator.get_profile(profile);
            }
        }
        let hit_elapsed = hit_start.elapsed();

        // Benchmark misses
        let miss_start = Instant::now();
        for _ in 0..self.iterations {
            let _ = simulator.get_profile(nonexistent);
        }
        let miss_elapsed = miss_start.elapsed();

        let elapsed = start.elapsed();
        let hit_lookups = self.iterations * profiles.len();

        BenchmarkResultBuilder::new(self.id())
            .metric("hit_lookups", hit_lookups)
            .metric("miss_lookups", self.iterations)
            .duration_ms("total_duration", elapsed)
            .metric("hit_ops_per_sec", hit_lookups as f64 / hit_elapsed.as_secs_f64())
            .metric("miss_ops_per_sec", self.iterations as f64 / miss_elapsed.as_secs_f64())
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_targets() {
        let targets = all_targets();
        assert!(!targets.is_empty());

        // Verify all targets have unique IDs
        let mut ids: Vec<&str> = targets.iter().map(|t| t.id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), targets.len(), "All targets must have unique IDs");
    }

    #[test]
    fn test_latency_sampling_benchmark() {
        let mut bench = LatencySamplingBenchmark::new();
        bench.iterations = 100; // Reduce for test speed
        let result = bench.run();

        assert_eq!(result.target_id(), "latency-sampling");
        assert!(result.metrics().get("throughput_ops_per_sec").is_some());
    }

    #[test]
    fn test_token_throughput_benchmark() {
        let mut bench = TokenThroughputBenchmark::new();
        bench.iterations = 10;
        let result = bench.run();

        assert_eq!(result.target_id(), "token-throughput");
        assert!(result.metrics().get("total_tokens").is_some());
    }

    #[test]
    fn test_embedding_benchmark() {
        let mut bench = EmbeddingGenerationBenchmark::new();
        bench.iterations = 100;
        let result = bench.run();

        assert_eq!(result.target_id(), "embedding-generation");
        assert!(result.metrics().get("dimensions").is_some());
    }
}
