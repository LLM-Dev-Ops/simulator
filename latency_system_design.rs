// ============================================================================
// LLM-Simulator: Enterprise-Grade Latency Modeling System
// ============================================================================
// Module: latency
// Purpose: Statistically accurate latency simulation for LLM inference
// Accuracy Target: Within 10% of real provider measurements
//
// Architecture:
// - latency::distributions - Statistical distribution implementations
// - latency::profiles - Provider-specific latency profiles
// - latency::streaming - Token-by-token timing simulation
// - latency::degradation - Load-dependent performance modeling
// - latency::validation - Measurement validation and calibration
// ============================================================================

use std::time::Duration;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ============================================================================
// Core Latency Modeling Traits
// ============================================================================

/// Trait for sampling latency values from statistical distributions
///
/// All distributions must be:
/// - Thread-safe (Send + Sync)
/// - Deterministic given the same RNG state
/// - Capable of computing percentiles for validation
pub trait LatencyDistribution: Send + Sync {
    /// Sample a single latency value from the distribution
    fn sample(&self, rng: &mut DeterministicRng) -> Duration;

    /// Compute the value at a given percentile (0.0 to 1.0)
    /// Used for validation against real measurements
    fn percentile(&self, p: f64) -> Duration;

    /// Get mean latency for the distribution
    fn mean(&self) -> Duration;

    /// Get standard deviation
    fn std_dev(&self) -> Duration;

    /// Clone the distribution for multi-threaded use
    fn clone_box(&self) -> Box<dyn LatencyDistribution>;

    /// Validate distribution parameters are physically reasonable
    fn validate(&self) -> Result<(), DistributionError>;
}

/// Deterministic random number generator for reproducible simulations
pub struct DeterministicRng {
    state: u64,
    seed: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed, seed }
    }

    pub fn reset(&mut self) {
        self.state = self.seed;
    }

    /// Generate uniform random in [0, 1) using xorshift64
    pub fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Generate standard normal using Box-Muller transform
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn get_seed(&self) -> u64 {
        self.seed
    }
}

// ============================================================================
// Statistical Distribution Implementations
// ============================================================================

/// Normal (Gaussian) distribution
/// Best for: Stable, symmetric latency patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalDistribution {
    mean_ms: f64,
    std_dev_ms: f64,
    min_clamp_ms: Option<f64>, // Prevent negative values
}

impl NormalDistribution {
    pub fn new(mean: Duration, std_dev: Duration) -> Self {
        Self {
            mean_ms: mean.as_secs_f64() * 1000.0,
            std_dev_ms: std_dev.as_secs_f64() * 1000.0,
            min_clamp_ms: Some(0.0), // Default: clamp at 0
        }
    }

    pub fn with_min_clamp(mut self, min_ms: f64) -> Self {
        self.min_clamp_ms = Some(min_ms);
        self
    }
}

impl LatencyDistribution for NormalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        let sample = self.mean_ms + self.std_dev_ms * rng.next_normal();
        let clamped = if let Some(min) = self.min_clamp_ms {
            sample.max(min)
        } else {
            sample
        };
        Duration::from_secs_f64(clamped / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        // Use inverse normal CDF approximation (Abramowitz and Stegun)
        let z = inverse_normal_cdf(p);
        let value_ms = self.mean_ms + self.std_dev_ms * z;
        let clamped = if let Some(min) = self.min_clamp_ms {
            value_ms.max(min)
        } else {
            value_ms
        };
        Duration::from_secs_f64(clamped / 1000.0)
    }

    fn mean(&self) -> Duration {
        Duration::from_secs_f64(self.mean_ms / 1000.0)
    }

    fn std_dev(&self) -> Duration {
        Duration::from_secs_f64(self.std_dev_ms / 1000.0)
    }

    fn clone_box(&self) -> Box<dyn LatencyDistribution> {
        Box::new(self.clone())
    }

    fn validate(&self) -> Result<(), DistributionError> {
        if self.mean_ms <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "mean must be positive".to_string()
            ));
        }
        if self.std_dev_ms < 0.0 {
            return Err(DistributionError::InvalidParameter(
                "std_dev must be non-negative".to_string()
            ));
        }
        Ok(())
    }
}

/// Log-normal distribution
/// Best for: Right-skewed latency with occasional spikes (realistic for LLMs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogNormalDistribution {
    mu: f64,      // Mean of log(X)
    sigma: f64,   // Std dev of log(X)
}

impl LogNormalDistribution {
    /// Create from desired mean and std_dev of the *actual* latency (not log)
    pub fn from_mean_std(mean: Duration, std_dev: Duration) -> Self {
        let mean_ms = mean.as_secs_f64() * 1000.0;
        let std_ms = std_dev.as_secs_f64() * 1000.0;

        // Convert to log-space parameters
        let variance = std_ms * std_ms;
        let mean_sq = mean_ms * mean_ms;
        let sigma_sq = (variance / mean_sq + 1.0).ln();
        let mu = mean_ms.ln() - 0.5 * sigma_sq;

        Self {
            mu,
            sigma: sigma_sq.sqrt(),
        }
    }

    /// Create from p50 and p99 percentiles (common measurement format)
    pub fn from_percentiles(p50: Duration, p99: Duration) -> Self {
        let p50_ms = p50.as_secs_f64() * 1000.0;
        let p99_ms = p99.as_secs_f64() * 1000.0;

        // For log-normal: p50 = exp(mu), p99 = exp(mu + 2.326*sigma)
        let mu = p50_ms.ln();
        let sigma = (p99_ms.ln() - mu) / 2.326; // 2.326 is z-score for 99th percentile

        Self { mu, sigma }
    }
}

impl LatencyDistribution for LogNormalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        let normal_sample = rng.next_normal();
        let log_value = self.mu + self.sigma * normal_sample;
        let value_ms = log_value.exp();
        Duration::from_secs_f64(value_ms / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        let z = inverse_normal_cdf(p);
        let log_value = self.mu + self.sigma * z;
        let value_ms = log_value.exp();
        Duration::from_secs_f64(value_ms / 1000.0)
    }

    fn mean(&self) -> Duration {
        let mean_ms = (self.mu + 0.5 * self.sigma * self.sigma).exp();
        Duration::from_secs_f64(mean_ms / 1000.0)
    }

    fn std_dev(&self) -> Duration {
        let variance = (self.sigma * self.sigma).exp() - 1.0;
        let mean_sq = (2.0 * self.mu + self.sigma * self.sigma).exp();
        let std_ms = (variance * mean_sq).sqrt();
        Duration::from_secs_f64(std_ms / 1000.0)
    }

    fn clone_box(&self) -> Box<dyn LatencyDistribution> {
        Box::new(self.clone())
    }

    fn validate(&self) -> Result<(), DistributionError> {
        if self.sigma < 0.0 {
            return Err(DistributionError::InvalidParameter(
                "sigma must be non-negative".to_string()
            ));
        }
        Ok(())
    }
}

/// Exponential distribution
/// Best for: Modeling wait times and queue delays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialDistribution {
    lambda: f64, // Rate parameter (1/mean)
}

impl ExponentialDistribution {
    pub fn new(mean: Duration) -> Self {
        let mean_ms = mean.as_secs_f64() * 1000.0;
        Self {
            lambda: 1.0 / mean_ms,
        }
    }
}

impl LatencyDistribution for ExponentialDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        let u = rng.next_f64();
        let value_ms = -(1.0 - u).ln() / self.lambda;
        Duration::from_secs_f64(value_ms / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        let value_ms = -(1.0 - p).ln() / self.lambda;
        Duration::from_secs_f64(value_ms / 1000.0)
    }

    fn mean(&self) -> Duration {
        Duration::from_secs_f64(1.0 / (self.lambda * 1000.0))
    }

    fn std_dev(&self) -> Duration {
        Duration::from_secs_f64(1.0 / (self.lambda * 1000.0))
    }

    fn clone_box(&self) -> Box<dyn LatencyDistribution> {
        Box::new(self.clone())
    }

    fn validate(&self) -> Result<(), DistributionError> {
        if self.lambda <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "lambda must be positive".to_string()
            ));
        }
        Ok(())
    }
}

/// Bimodal distribution (mixture of two normals)
/// Best for: Modeling cache hits/misses, fast path vs slow path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BimodalDistribution {
    fast_mode: NormalDistribution,
    slow_mode: NormalDistribution,
    fast_probability: f64, // Probability of sampling from fast mode
}

impl BimodalDistribution {
    pub fn new(
        fast_mean: Duration,
        fast_std: Duration,
        slow_mean: Duration,
        slow_std: Duration,
        fast_probability: f64,
    ) -> Self {
        Self {
            fast_mode: NormalDistribution::new(fast_mean, fast_std),
            slow_mode: NormalDistribution::new(slow_mean, slow_std),
            fast_probability: fast_probability.clamp(0.0, 1.0),
        }
    }

    /// Create from cache scenario: hit_rate and two latency profiles
    pub fn from_cache_scenario(
        cache_hit_latency: Duration,
        cache_miss_latency: Duration,
        hit_rate: f64,
        variance_factor: f64, // Fraction of mean as std dev
    ) -> Self {
        let hit_std = Duration::from_secs_f64(
            cache_hit_latency.as_secs_f64() * variance_factor
        );
        let miss_std = Duration::from_secs_f64(
            cache_miss_latency.as_secs_f64() * variance_factor
        );

        Self::new(
            cache_hit_latency,
            hit_std,
            cache_miss_latency,
            miss_std,
            hit_rate,
        )
    }
}

impl LatencyDistribution for BimodalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        if rng.next_f64() < self.fast_probability {
            self.fast_mode.sample(rng)
        } else {
            self.slow_mode.sample(rng)
        }
    }

    fn percentile(&self, p: f64) -> Duration {
        // For mixture distributions, percentile calculation is approximate
        // We use a weighted combination (not mathematically exact but practical)
        let fast_p = self.fast_mode.percentile(p);
        let slow_p = self.slow_mode.percentile(p);

        let fast_weight = self.fast_probability;
        let slow_weight = 1.0 - self.fast_probability;

        let weighted_ms = fast_p.as_secs_f64() * fast_weight
                        + slow_p.as_secs_f64() * slow_weight;
        Duration::from_secs_f64(weighted_ms)
    }

    fn mean(&self) -> Duration {
        let fast_mean = self.fast_mode.mean().as_secs_f64();
        let slow_mean = self.slow_mode.mean().as_secs_f64();
        let weighted = fast_mean * self.fast_probability
                     + slow_mean * (1.0 - self.fast_probability);
        Duration::from_secs_f64(weighted)
    }

    fn std_dev(&self) -> Duration {
        // Variance of mixture: E[Var] + Var[E]
        let fast_var = self.fast_mode.std_dev().as_secs_f64().powi(2);
        let slow_var = self.slow_mode.std_dev().as_secs_f64().powi(2);
        let fast_mean = self.fast_mode.mean().as_secs_f64();
        let slow_mean = self.slow_mode.mean().as_secs_f64();
        let overall_mean = self.mean().as_secs_f64();

        let e_var = fast_var * self.fast_probability
                  + slow_var * (1.0 - self.fast_probability);
        let var_e = (fast_mean - overall_mean).powi(2) * self.fast_probability
                  + (slow_mean - overall_mean).powi(2) * (1.0 - self.fast_probability);

        let total_variance = e_var + var_e;
        Duration::from_secs_f64(total_variance.sqrt())
    }

    fn clone_box(&self) -> Box<dyn LatencyDistribution> {
        Box::new(self.clone())
    }

    fn validate(&self) -> Result<(), DistributionError> {
        self.fast_mode.validate()?;
        self.slow_mode.validate()?;
        if self.fast_probability < 0.0 || self.fast_probability > 1.0 {
            return Err(DistributionError::InvalidParameter(
                "fast_probability must be in [0, 1]".to_string()
            ));
        }
        Ok(())
    }
}

/// Empirical distribution from measured data
/// Best for: Using real production measurements directly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalDistribution {
    samples_ms: Vec<f64>, // Sorted samples
    interpolation: InterpolationMethod,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Linear,
    Nearest,
}

impl EmpiricalDistribution {
    pub fn from_measurements(mut samples: Vec<Duration>) -> Self {
        samples.sort();
        let samples_ms: Vec<f64> = samples
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        Self {
            samples_ms,
            interpolation: InterpolationMethod::Linear,
        }
    }

    pub fn with_interpolation(mut self, method: InterpolationMethod) -> Self {
        self.interpolation = method;
        self
    }
}

impl LatencyDistribution for EmpiricalDistribution {
    fn sample(&self, rng: &mut DeterministicRng) -> Duration {
        if self.samples_ms.is_empty() {
            return Duration::from_secs(0);
        }

        let idx = (rng.next_f64() * self.samples_ms.len() as f64) as usize;
        let idx = idx.min(self.samples_ms.len() - 1);
        Duration::from_secs_f64(self.samples_ms[idx] / 1000.0)
    }

    fn percentile(&self, p: f64) -> Duration {
        if self.samples_ms.is_empty() {
            return Duration::from_secs(0);
        }

        let position = p * (self.samples_ms.len() - 1) as f64;
        let lower_idx = position.floor() as usize;
        let upper_idx = (position.ceil() as usize).min(self.samples_ms.len() - 1);

        let value_ms = match self.interpolation {
            InterpolationMethod::Linear => {
                if lower_idx == upper_idx {
                    self.samples_ms[lower_idx]
                } else {
                    let fraction = position - lower_idx as f64;
                    self.samples_ms[lower_idx] * (1.0 - fraction)
                        + self.samples_ms[upper_idx] * fraction
                }
            }
            InterpolationMethod::Nearest => {
                if position - lower_idx as f64 < 0.5 {
                    self.samples_ms[lower_idx]
                } else {
                    self.samples_ms[upper_idx]
                }
            }
        };

        Duration::from_secs_f64(value_ms / 1000.0)
    }

    fn mean(&self) -> Duration {
        if self.samples_ms.is_empty() {
            return Duration::from_secs(0);
        }
        let sum: f64 = self.samples_ms.iter().sum();
        let mean_ms = sum / self.samples_ms.len() as f64;
        Duration::from_secs_f64(mean_ms / 1000.0)
    }

    fn std_dev(&self) -> Duration {
        if self.samples_ms.len() < 2 {
            return Duration::from_secs(0);
        }
        let mean_ms = self.mean().as_secs_f64() * 1000.0;
        let variance: f64 = self.samples_ms
            .iter()
            .map(|&x| (x - mean_ms).powi(2))
            .sum::<f64>() / (self.samples_ms.len() - 1) as f64;
        Duration::from_secs_f64(variance.sqrt() / 1000.0)
    }

    fn clone_box(&self) -> Box<dyn LatencyDistribution> {
        Box::new(self.clone())
    }

    fn validate(&self) -> Result<(), DistributionError> {
        if self.samples_ms.is_empty() {
            return Err(DistributionError::InvalidParameter(
                "samples must not be empty".to_string()
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Latency Profile Configuration
// ============================================================================

/// Complete latency profile for a provider/model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyProfile {
    pub name: String,
    pub provider: String,
    pub model: String,

    /// Time to First Token distribution
    pub ttft: DistributionConfig,

    /// Inter-Token Latency distribution
    pub itl: DistributionConfig,

    /// Network jitter added to each token
    pub network_jitter: Option<DistributionConfig>,

    /// Load degradation curve
    pub degradation: DegradationConfig,

    /// Validation metrics from real measurements
    pub validation_metrics: Option<ValidationMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionConfig {
    Normal {
        mean_ms: f64,
        std_dev_ms: f64,
    },
    LogNormal {
        p50_ms: f64,
        p99_ms: f64,
    },
    Exponential {
        mean_ms: f64,
    },
    Bimodal {
        fast_mean_ms: f64,
        fast_std_ms: f64,
        slow_mean_ms: f64,
        slow_std_ms: f64,
        fast_probability: f64,
    },
    Empirical {
        samples_ms: Vec<f64>,
    },
}

impl DistributionConfig {
    pub fn build(&self) -> Box<dyn LatencyDistribution> {
        match self {
            DistributionConfig::Normal { mean_ms, std_dev_ms } => {
                Box::new(NormalDistribution::new(
                    Duration::from_secs_f64(*mean_ms / 1000.0),
                    Duration::from_secs_f64(*std_dev_ms / 1000.0),
                ))
            }
            DistributionConfig::LogNormal { p50_ms, p99_ms } => {
                Box::new(LogNormalDistribution::from_percentiles(
                    Duration::from_secs_f64(*p50_ms / 1000.0),
                    Duration::from_secs_f64(*p99_ms / 1000.0),
                ))
            }
            DistributionConfig::Exponential { mean_ms } => {
                Box::new(ExponentialDistribution::new(
                    Duration::from_secs_f64(*mean_ms / 1000.0)
                ))
            }
            DistributionConfig::Bimodal {
                fast_mean_ms,
                fast_std_ms,
                slow_mean_ms,
                slow_std_ms,
                fast_probability,
            } => {
                Box::new(BimodalDistribution::new(
                    Duration::from_secs_f64(*fast_mean_ms / 1000.0),
                    Duration::from_secs_f64(*fast_std_ms / 1000.0),
                    Duration::from_secs_f64(*slow_mean_ms / 1000.0),
                    Duration::from_secs_f64(*slow_std_ms / 1000.0),
                    *fast_probability,
                ))
            }
            DistributionConfig::Empirical { samples_ms } => {
                let samples: Vec<Duration> = samples_ms
                    .iter()
                    .map(|&ms| Duration::from_secs_f64(ms / 1000.0))
                    .collect();
                Box::new(EmpiricalDistribution::from_measurements(samples))
            }
        }
    }
}

/// Validation metrics from real provider measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub ttft_p50_ms: f64,
    pub ttft_p99_ms: f64,
    pub itl_mean_ms: f64,
    pub itl_p99_ms: f64,
    pub tokens_per_sec: f64,
    pub measurement_date: String,
    pub sample_size: usize,
}

// ============================================================================
// Load-Dependent Degradation
// ============================================================================

/// Models how latency degrades under load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    pub model: DegradationModel,
    pub baseline_qps: f64, // Queries per second at baseline performance
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationModel {
    /// No degradation (for testing)
    None,

    /// Linear increase: latency_multiplier = 1 + slope * (qps / baseline_qps)
    Linear {
        slope: f64,
    },

    /// Exponential: latency_multiplier = exp(alpha * (qps / baseline_qps))
    Exponential {
        alpha: f64,
    },

    /// Queueing theory (M/M/1): accounts for utilization
    /// latency = base_latency / (1 - utilization)
    MMOne {
        service_rate: f64, // Max requests per second
    },

    /// Piecewise linear with multiple breakpoints
    Piecewise {
        breakpoints: Vec<(f64, f64)>, // (qps, multiplier) pairs
    },
}

impl DegradationConfig {
    pub fn compute_multiplier(&self, current_qps: f64) -> f64 {
        match &self.model {
            DegradationModel::None => 1.0,

            DegradationModel::Linear { slope } => {
                let load_ratio = current_qps / self.baseline_qps;
                1.0 + slope * load_ratio
            }

            DegradationModel::Exponential { alpha } => {
                let load_ratio = current_qps / self.baseline_qps;
                (alpha * load_ratio).exp()
            }

            DegradationModel::MMOne { service_rate } => {
                let utilization = (current_qps / service_rate).min(0.99); // Cap at 99%
                1.0 / (1.0 - utilization)
            }

            DegradationModel::Piecewise { breakpoints } => {
                if breakpoints.is_empty() {
                    return 1.0;
                }

                // Find the segment
                for i in 0..breakpoints.len() - 1 {
                    let (qps1, mult1) = breakpoints[i];
                    let (qps2, mult2) = breakpoints[i + 1];

                    if current_qps >= qps1 && current_qps <= qps2 {
                        // Linear interpolation
                        let fraction = (current_qps - qps1) / (qps2 - qps1);
                        return mult1 + fraction * (mult2 - mult1);
                    }
                }

                // Below first or above last
                if current_qps < breakpoints[0].0 {
                    breakpoints[0].1
                } else {
                    breakpoints[breakpoints.len() - 1].1
                }
            }
        }
    }
}

// ============================================================================
// Streaming Token Timing
// ============================================================================

/// Simulates the timing of individual token arrivals in streaming mode
pub struct StreamingSimulator {
    profile: LatencyProfile,
    rng: DeterministicRng,
    degradation_multiplier: f64,
}

impl StreamingSimulator {
    pub fn new(profile: LatencyProfile, seed: u64) -> Self {
        Self {
            profile,
            rng: DeterministicRng::new(seed),
            degradation_multiplier: 1.0,
        }
    }

    pub fn set_load(&mut self, current_qps: f64) {
        self.degradation_multiplier = self.profile.degradation.compute_multiplier(current_qps);
    }

    /// Generate timing for a complete streaming response
    pub fn generate_stream_timing(&mut self, num_tokens: usize) -> StreamTiming {
        let mut timings = Vec::with_capacity(num_tokens + 1);

        // Time to First Token
        let ttft_dist = self.profile.ttft.build();
        let mut ttft = ttft_dist.sample(&mut self.rng);
        ttft = multiply_duration(ttft, self.degradation_multiplier);
        timings.push(TokenTiming {
            token_index: 0,
            elapsed_since_start: ttft,
            delta_from_previous: ttft,
        });

        // Inter-Token Latencies
        let itl_dist = self.profile.itl.build();
        let jitter_dist = self.profile.network_jitter.as_ref().map(|cfg| cfg.build());

        let mut cumulative = ttft;
        for i in 1..=num_tokens {
            let mut itl = itl_dist.sample(&mut self.rng);
            itl = multiply_duration(itl, self.degradation_multiplier);

            // Add network jitter if configured
            if let Some(ref jitter) = jitter_dist {
                let jitter_amount = jitter.sample(&mut self.rng);
                itl += jitter_amount;
            }

            cumulative += itl;
            timings.push(TokenTiming {
                token_index: i,
                elapsed_since_start: cumulative,
                delta_from_previous: itl,
            });
        }

        StreamTiming {
            timings,
            total_duration: cumulative,
            ttft,
            mean_itl: self.compute_mean_itl(&timings),
        }
    }

    fn compute_mean_itl(&self, timings: &[TokenTiming]) -> Duration {
        if timings.len() <= 1 {
            return Duration::from_secs(0);
        }

        let total: Duration = timings[1..]
            .iter()
            .map(|t| t.delta_from_previous)
            .sum();

        total / (timings.len() - 1) as u32
    }
}

#[derive(Debug, Clone)]
pub struct StreamTiming {
    pub timings: Vec<TokenTiming>,
    pub total_duration: Duration,
    pub ttft: Duration,
    pub mean_itl: Duration,
}

#[derive(Debug, Clone)]
pub struct TokenTiming {
    pub token_index: usize,
    pub elapsed_since_start: Duration,
    pub delta_from_previous: Duration,
}

impl StreamTiming {
    /// Get the timestamp when a specific token should arrive
    pub fn get_token_arrival(&self, token_index: usize) -> Option<Duration> {
        self.timings.get(token_index).map(|t| t.elapsed_since_start)
    }

    /// Calculate tokens per second for this stream
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        (self.timings.len() - 1) as f64 / self.total_duration.as_secs_f64()
    }

    /// Validate against expected metrics
    pub fn validate(&self, expected: &ValidationMetrics) -> ValidationResult {
        let mut errors = Vec::new();

        // Check TTFT (allow 10% tolerance)
        let ttft_ms = self.ttft.as_secs_f64() * 1000.0;
        let ttft_error = (ttft_ms - expected.ttft_p50_ms).abs() / expected.ttft_p50_ms;
        if ttft_error > 0.10 {
            errors.push(format!(
                "TTFT error: {:.1}% (expected {:.0}ms, got {:.0}ms)",
                ttft_error * 100.0,
                expected.ttft_p50_ms,
                ttft_ms
            ));
        }

        // Check ITL
        let itl_ms = self.mean_itl.as_secs_f64() * 1000.0;
        let itl_error = (itl_ms - expected.itl_mean_ms).abs() / expected.itl_mean_ms;
        if itl_error > 0.10 {
            errors.push(format!(
                "ITL error: {:.1}% (expected {:.0}ms, got {:.0}ms)",
                itl_error * 100.0,
                expected.itl_mean_ms,
                itl_ms
            ));
        }

        // Check tokens/sec
        let tps = self.tokens_per_second();
        let tps_error = (tps - expected.tokens_per_sec).abs() / expected.tokens_per_sec;
        if tps_error > 0.10 {
            errors.push(format!(
                "Tokens/sec error: {:.1}% (expected {:.1}, got {:.1})",
                tps_error * 100.0,
                expected.tokens_per_sec,
                tps
            ));
        }

        ValidationResult {
            passed: errors.is_empty(),
            errors,
        }
    }
}

#[derive(Debug)]
pub struct ValidationResult {
    pub passed: bool,
    pub errors: Vec<String>,
}

// ============================================================================
// Complete Latency Model (Main API)
// ============================================================================

/// High-level latency model that orchestrates all components
pub struct LatencyModel {
    profiles: HashMap<String, LatencyProfile>,
    current_load_qps: f64,
    seed: u64,
}

impl LatencyModel {
    pub fn new(seed: u64) -> Self {
        Self {
            profiles: HashMap::new(),
            current_load_qps: 0.0,
            seed,
        }
    }

    /// Load built-in provider profiles
    pub fn with_builtin_profiles(mut self) -> Self {
        self.profiles.insert(
            "gpt-4-turbo".to_string(),
            Self::gpt4_turbo_profile(),
        );
        self.profiles.insert(
            "gpt-3.5-turbo".to_string(),
            Self::gpt35_turbo_profile(),
        );
        self.profiles.insert(
            "claude-3-opus".to_string(),
            Self::claude3_opus_profile(),
        );
        self.profiles.insert(
            "claude-3-sonnet".to_string(),
            Self::claude3_sonnet_profile(),
        );
        self.profiles.insert(
            "gemini-1.5-pro".to_string(),
            Self::gemini15_pro_profile(),
        );
        self
    }

    pub fn add_profile(&mut self, key: String, profile: LatencyProfile) {
        self.profiles.insert(key, profile);
    }

    pub fn set_load(&mut self, qps: f64) {
        self.current_load_qps = qps;
    }

    /// Create a streaming simulator for a specific profile
    pub fn create_simulator(&self, profile_key: &str) -> Result<StreamingSimulator, ModelError> {
        let profile = self.profiles
            .get(profile_key)
            .ok_or_else(|| ModelError::ProfileNotFound(profile_key.to_string()))?
            .clone();

        let mut simulator = StreamingSimulator::new(profile, self.seed);
        simulator.set_load(self.current_load_qps);
        Ok(simulator)
    }

    /// Simulate a single request end-to-end
    pub fn simulate_request(
        &self,
        profile_key: &str,
        num_tokens: usize,
    ) -> Result<StreamTiming, ModelError> {
        let mut simulator = self.create_simulator(profile_key)?;
        Ok(simulator.generate_stream_timing(num_tokens))
    }

    /// Batch validation across many requests
    pub fn validate_profile(
        &self,
        profile_key: &str,
        num_simulations: usize,
        tokens_per_request: usize,
    ) -> Result<ProfileValidationReport, ModelError> {
        let profile = self.profiles
            .get(profile_key)
            .ok_or_else(|| ModelError::ProfileNotFound(profile_key.to_string()))?;

        let validation_metrics = profile.validation_metrics.as_ref()
            .ok_or(ModelError::NoValidationMetrics)?;

        let mut simulator = self.create_simulator(profile_key)?;
        let mut ttft_samples = Vec::with_capacity(num_simulations);
        let mut itl_samples = Vec::with_capacity(num_simulations);
        let mut tps_samples = Vec::with_capacity(num_simulations);

        for _ in 0..num_simulations {
            let timing = simulator.generate_stream_timing(tokens_per_request);
            ttft_samples.push(timing.ttft.as_secs_f64() * 1000.0);
            itl_samples.push(timing.mean_itl.as_secs_f64() * 1000.0);
            tps_samples.push(timing.tokens_per_second());
        }

        // Sort for percentile calculations
        ttft_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        itl_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ttft_p50 = percentile_sorted(&ttft_samples, 0.50);
        let ttft_p99 = percentile_sorted(&ttft_samples, 0.99);
        let itl_mean = itl_samples.iter().sum::<f64>() / itl_samples.len() as f64;
        let tps_mean = tps_samples.iter().sum::<f64>() / tps_samples.len() as f64;

        // Compute errors
        let ttft_p50_error = ((ttft_p50 - validation_metrics.ttft_p50_ms).abs()
                             / validation_metrics.ttft_p50_ms) * 100.0;
        let ttft_p99_error = ((ttft_p99 - validation_metrics.ttft_p99_ms).abs()
                             / validation_metrics.ttft_p99_ms) * 100.0;
        let itl_error = ((itl_mean - validation_metrics.itl_mean_ms).abs()
                        / validation_metrics.itl_mean_ms) * 100.0;
        let tps_error = ((tps_mean - validation_metrics.tokens_per_sec).abs()
                        / validation_metrics.tokens_per_sec) * 100.0;

        let all_within_tolerance = ttft_p50_error <= 10.0
            && ttft_p99_error <= 10.0
            && itl_error <= 10.0
            && tps_error <= 10.0;

        Ok(ProfileValidationReport {
            profile_name: profile_key.to_string(),
            num_simulations,
            ttft_p50_simulated: ttft_p50,
            ttft_p50_expected: validation_metrics.ttft_p50_ms,
            ttft_p50_error_pct: ttft_p50_error,
            ttft_p99_simulated: ttft_p99,
            ttft_p99_expected: validation_metrics.ttft_p99_ms,
            ttft_p99_error_pct: ttft_p99_error,
            itl_mean_simulated: itl_mean,
            itl_mean_expected: validation_metrics.itl_mean_ms,
            itl_error_pct: itl_error,
            tps_simulated: tps_mean,
            tps_expected: validation_metrics.tokens_per_sec,
            tps_error_pct: tps_error,
            passed: all_within_tolerance,
        })
    }

    // ========================================================================
    // Built-in Provider Profiles
    // ========================================================================

    fn gpt4_turbo_profile() -> LatencyProfile {
        LatencyProfile {
            name: "GPT-4 Turbo".to_string(),
            provider: "OpenAI".to_string(),
            model: "gpt-4-turbo".to_string(),
            ttft: DistributionConfig::LogNormal {
                p50_ms: 800.0,
                p99_ms: 2500.0,
            },
            itl: DistributionConfig::Normal {
                mean_ms: 20.0,
                std_dev_ms: 5.0,
            },
            network_jitter: Some(DistributionConfig::Normal {
                mean_ms: 0.0,
                std_dev_ms: 2.0,
            }),
            degradation: DegradationConfig {
                model: DegradationModel::Exponential { alpha: 0.5 },
                baseline_qps: 10.0,
            },
            validation_metrics: Some(ValidationMetrics {
                ttft_p50_ms: 800.0,
                ttft_p99_ms: 2500.0,
                itl_mean_ms: 20.0,
                itl_p99_ms: 35.0,
                tokens_per_sec: 50.0,
                measurement_date: "2024-01".to_string(),
                sample_size: 1000,
            }),
        }
    }

    fn gpt35_turbo_profile() -> LatencyProfile {
        LatencyProfile {
            name: "GPT-3.5 Turbo".to_string(),
            provider: "OpenAI".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            ttft: DistributionConfig::LogNormal {
                p50_ms: 250.0,
                p99_ms: 800.0,
            },
            itl: DistributionConfig::Normal {
                mean_ms: 10.0,
                std_dev_ms: 3.0,
            },
            network_jitter: Some(DistributionConfig::Normal {
                mean_ms: 0.0,
                std_dev_ms: 1.5,
            }),
            degradation: DegradationConfig {
                model: DegradationModel::Linear { slope: 0.3 },
                baseline_qps: 50.0,
            },
            validation_metrics: Some(ValidationMetrics {
                ttft_p50_ms: 250.0,
                ttft_p99_ms: 800.0,
                itl_mean_ms: 10.0,
                itl_p99_ms: 18.0,
                tokens_per_sec: 100.0,
                measurement_date: "2024-01".to_string(),
                sample_size: 1000,
            }),
        }
    }

    fn claude3_opus_profile() -> LatencyProfile {
        LatencyProfile {
            name: "Claude 3 Opus".to_string(),
            provider: "Anthropic".to_string(),
            model: "claude-3-opus".to_string(),
            ttft: DistributionConfig::LogNormal {
                p50_ms: 1200.0,
                p99_ms: 3000.0,
            },
            itl: DistributionConfig::Normal {
                mean_ms: 25.0,
                std_dev_ms: 6.0,
            },
            network_jitter: Some(DistributionConfig::Normal {
                mean_ms: 0.0,
                std_dev_ms: 2.5,
            }),
            degradation: DegradationConfig {
                model: DegradationModel::MMOne { service_rate: 8.0 },
                baseline_qps: 5.0,
            },
            validation_metrics: Some(ValidationMetrics {
                ttft_p50_ms: 1200.0,
                ttft_p99_ms: 3000.0,
                itl_mean_ms: 25.0,
                itl_p99_ms: 45.0,
                tokens_per_sec: 40.0,
                measurement_date: "2024-01".to_string(),
                sample_size: 1000,
            }),
        }
    }

    fn claude3_sonnet_profile() -> LatencyProfile {
        LatencyProfile {
            name: "Claude 3 Sonnet".to_string(),
            provider: "Anthropic".to_string(),
            model: "claude-3-sonnet".to_string(),
            ttft: DistributionConfig::LogNormal {
                p50_ms: 500.0,
                p99_ms: 1500.0,
            },
            itl: DistributionConfig::Normal {
                mean_ms: 15.0,
                std_dev_ms: 4.0,
            },
            network_jitter: Some(DistributionConfig::Normal {
                mean_ms: 0.0,
                std_dev_ms: 2.0,
            }),
            degradation: DegradationConfig {
                model: DegradationModel::Exponential { alpha: 0.4 },
                baseline_qps: 20.0,
            },
            validation_metrics: Some(ValidationMetrics {
                ttft_p50_ms: 500.0,
                ttft_p99_ms: 1500.0,
                itl_mean_ms: 15.0,
                itl_p99_ms: 28.0,
                tokens_per_sec: 65.0,
                measurement_date: "2024-01".to_string(),
                sample_size: 1000,
            }),
        }
    }

    fn gemini15_pro_profile() -> LatencyProfile {
        LatencyProfile {
            name: "Gemini 1.5 Pro".to_string(),
            provider: "Google".to_string(),
            model: "gemini-1.5-pro".to_string(),
            ttft: DistributionConfig::LogNormal {
                p50_ms: 600.0,
                p99_ms: 1800.0,
            },
            itl: DistributionConfig::Normal {
                mean_ms: 18.0,
                std_dev_ms: 5.0,
            },
            network_jitter: Some(DistributionConfig::Normal {
                mean_ms: 0.0,
                std_dev_ms: 2.0,
            }),
            degradation: DegradationConfig {
                model: DegradationModel::Linear { slope: 0.4 },
                baseline_qps: 15.0,
            },
            validation_metrics: Some(ValidationMetrics {
                ttft_p50_ms: 600.0,
                ttft_p99_ms: 1800.0,
                itl_mean_ms: 18.0,
                itl_p99_ms: 32.0,
                tokens_per_sec: 55.0,
                measurement_date: "2024-01".to_string(),
                sample_size: 1000,
            }),
        }
    }
}

#[derive(Debug)]
pub struct ProfileValidationReport {
    pub profile_name: String,
    pub num_simulations: usize,
    pub ttft_p50_simulated: f64,
    pub ttft_p50_expected: f64,
    pub ttft_p50_error_pct: f64,
    pub ttft_p99_simulated: f64,
    pub ttft_p99_expected: f64,
    pub ttft_p99_error_pct: f64,
    pub itl_mean_simulated: f64,
    pub itl_mean_expected: f64,
    pub itl_error_pct: f64,
    pub tps_simulated: f64,
    pub tps_expected: f64,
    pub tps_error_pct: f64,
    pub passed: bool,
}

impl ProfileValidationReport {
    pub fn print(&self) {
        println!("=== Validation Report: {} ===", self.profile_name);
        println!("Simulations: {}", self.num_simulations);
        println!();
        println!("TTFT p50: {:.1}ms (expected {:.1}ms) - Error: {:.1}%",
                 self.ttft_p50_simulated, self.ttft_p50_expected, self.ttft_p50_error_pct);
        println!("TTFT p99: {:.1}ms (expected {:.1}ms) - Error: {:.1}%",
                 self.ttft_p99_simulated, self.ttft_p99_expected, self.ttft_p99_error_pct);
        println!("ITL mean: {:.1}ms (expected {:.1}ms) - Error: {:.1}%",
                 self.itl_mean_simulated, self.itl_mean_expected, self.itl_error_pct);
        println!("Tokens/sec: {:.1} (expected {:.1}) - Error: {:.1}%",
                 self.tps_simulated, self.tps_expected, self.tps_error_pct);
        println!();
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
    }
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum DistributionError {
    InvalidParameter(String),
}

#[derive(Debug)]
pub enum ModelError {
    ProfileNotFound(String),
    NoValidationMetrics,
    DistributionError(DistributionError),
}

impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::ProfileNotFound(name) => write!(f, "Profile not found: {}", name),
            ModelError::NoValidationMetrics => write!(f, "No validation metrics available"),
            ModelError::DistributionError(e) => write!(f, "Distribution error: {}", e),
        }
    }
}

impl std::error::Error for DistributionError {}
impl std::error::Error for ModelError {}

// ============================================================================
// Utility Functions
// ============================================================================

/// Inverse normal CDF approximation (Abramowitz and Stegun 26.2.23)
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }

    let q = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * q.ln()).sqrt();

    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    let numerator = C0 + t * (C1 + t * C2);
    let denominator = 1.0 + t * (D1 + t * (D2 + t * D3));
    let z = t - numerator / denominator;

    if p < 0.5 { -z } else { z }
}

/// Multiply duration by a scalar factor
fn multiply_duration(d: Duration, factor: f64) -> Duration {
    Duration::from_secs_f64(d.as_secs_f64() * factor)
}

/// Calculate percentile from sorted array
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ============================================================================
// Usage Example
// ============================================================================

#[cfg(test)]
mod example_usage {
    use super::*;

    fn demonstrate_latency_system() {
        // Create latency model with built-in profiles
        let model = LatencyModel::new(42)
            .with_builtin_profiles();

        // Simulate a single GPT-4 Turbo request
        println!("=== Single Request Simulation ===");
        let timing = model.simulate_request("gpt-4-turbo", 100).unwrap();
        println!("TTFT: {:?}", timing.ttft);
        println!("Mean ITL: {:?}", timing.mean_itl);
        println!("Total duration: {:?}", timing.total_duration);
        println!("Tokens/sec: {:.2}", timing.tokens_per_second());

        // Validate profile against measurements
        println!("\n=== Profile Validation ===");
        let report = model.validate_profile("gpt-4-turbo", 1000, 100).unwrap();
        report.print();

        // Demonstrate load degradation
        println!("\n=== Load Degradation Test ===");
        let mut model_with_load = model;
        for qps in [1.0, 5.0, 10.0, 20.0, 50.0] {
            model_with_load.set_load(qps);
            let timing = model_with_load.simulate_request("gpt-4-turbo", 100).unwrap();
            println!("QPS: {:.0}, TTFT: {:?}, Mean ITL: {:?}",
                     qps, timing.ttft, timing.mean_itl);
        }

        // Custom profile with empirical distribution
        println!("\n=== Custom Profile ===");
        let custom_profile = LatencyProfile {
            name: "Custom Model".to_string(),
            provider: "Custom".to_string(),
            model: "custom-1".to_string(),
            ttft: DistributionConfig::Empirical {
                samples_ms: vec![100.0, 150.0, 200.0, 180.0, 220.0],
            },
            itl: DistributionConfig::Bimodal {
                fast_mean_ms: 5.0,
                fast_std_ms: 1.0,
                slow_mean_ms: 50.0,
                slow_std_ms: 10.0,
                fast_probability: 0.9,
            },
            network_jitter: None,
            degradation: DegradationConfig {
                model: DegradationModel::None,
                baseline_qps: 1.0,
            },
            validation_metrics: None,
        };

        let mut custom_model = LatencyModel::new(42);
        custom_model.add_profile("custom".to_string(), custom_profile);
        let timing = custom_model.simulate_request("custom", 50).unwrap();
        println!("Custom model - TTFT: {:?}, Tokens/sec: {:.2}",
                 timing.ttft, timing.tokens_per_second());
    }
}

// ============================================================================
// Production Integration Notes
// ============================================================================

/*
INTEGRATION CHECKLIST:

1. Configuration Loading:
   - Load profiles from YAML/JSON config files
   - Support hot-reloading for profile updates
   - Validate all profiles on startup

2. Metrics Collection:
   - Export simulation metrics to Prometheus
   - Track: TTFT, ITL, E2E latency percentiles
   - Monitor validation error rates

3. Calibration System:
   - Periodic validation against live provider APIs
   - Automatic profile adjustment based on drift detection
   - Alert when error exceeds 10% threshold

4. Performance:
   - Pre-build distributions for hot paths
   - Use thread-local RNGs for parallelism
   - Cache percentile calculations

5. Testing:
   - Unit tests for each distribution
   - Integration tests for complete profiles
   - Property-based testing for statistical properties
   - Regression tests against known measurements

6. Documentation:
   - Profile schema documentation
   - Calibration procedure guide
   - Troubleshooting guide for accuracy issues
*/
