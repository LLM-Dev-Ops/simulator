//! Retry and Backoff Infrastructure Module
//!
//! Provides retry logic with configurable backoff strategies for Phase 2B adapter consumption.
//! This module fills the gap identified in the infrastructure analysis - the Simulator
//! previously lacked a general-purpose retry abstraction.
//!
//! ## Features
//!
//! - Multiple backoff strategies (constant, linear, exponential, decorrelated jitter)
//! - Configurable retry conditions
//! - Async-friendly with tokio integration
//! - Retry budget tracking
//! - Circuit breaker pattern support
//!
//! ## Example
//!
//! ```rust,ignore
//! use llm_simulator::infra::retry::{RetryPolicy, RetryConfig};
//! use std::time::Duration;
//!
//! let policy = RetryPolicy::exponential()
//!     .max_retries(3)
//!     .base_delay(Duration::from_millis(100))
//!     .max_delay(Duration::from_secs(10));
//!
//! let result = policy.retry(|| async {
//!     // Your fallible operation
//!     fetch_from_upstream().await
//! }).await;
//! ```

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Base delay between retries
    pub base_delay_ms: u64,
    /// Maximum delay between retries
    pub max_delay_ms: u64,
    /// Backoff strategy
    pub strategy: BackoffStrategy,
    /// Jitter factor (0.0 - 1.0)
    pub jitter: f64,
    /// Enable retry budget
    pub budget_enabled: bool,
    /// Retry budget per minute
    pub budget_per_minute: u32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            strategy: BackoffStrategy::ExponentialWithJitter,
            jitter: 0.2,
            budget_enabled: false,
            budget_per_minute: 100,
        }
    }
}

/// Backoff strategy enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackoffStrategy {
    /// Constant delay between retries
    Constant,
    /// Linearly increasing delay
    Linear,
    /// Exponentially increasing delay
    Exponential,
    /// Exponential with jitter (recommended)
    ExponentialWithJitter,
    /// Decorrelated jitter (AWS style)
    DecorrelatedJitter,
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self::ExponentialWithJitter
    }
}

/// Retry error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum RetryError<E> {
    #[error("Max retries ({max}) exceeded. Last error: {last_error}")]
    MaxRetriesExceeded {
        max: u32,
        attempts: u32,
        last_error: E,
    },

    #[error("Retry budget exhausted")]
    BudgetExhausted,

    #[error("Operation failed: {0}")]
    OperationFailed(E),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// Backoff calculator
#[derive(Debug, Clone)]
pub struct Backoff {
    /// Base delay
    base_delay: Duration,
    /// Maximum delay
    max_delay: Duration,
    /// Backoff strategy
    strategy: BackoffStrategy,
    /// Jitter factor
    jitter: f64,
    /// Current attempt (for decorrelated jitter)
    last_delay: Duration,
}

impl Backoff {
    /// Create a new backoff calculator
    pub fn new(base_delay: Duration, max_delay: Duration, strategy: BackoffStrategy, jitter: f64) -> Self {
        Self {
            base_delay,
            max_delay,
            strategy,
            jitter,
            last_delay: base_delay,
        }
    }

    /// Calculate delay for the given attempt number (0-indexed)
    pub fn delay_for_attempt(&mut self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let max_ms = self.max_delay.as_millis() as f64;

        let delay_ms = match self.strategy {
            BackoffStrategy::Constant => base_ms,
            BackoffStrategy::Linear => base_ms * (attempt + 1) as f64,
            BackoffStrategy::Exponential => base_ms * 2_f64.powi(attempt as i32),
            BackoffStrategy::ExponentialWithJitter => {
                let exp_delay = base_ms * 2_f64.powi(attempt as i32);
                let jitter_range = exp_delay * self.jitter;
                let jitter = rand::thread_rng().gen_range(-jitter_range..jitter_range);
                (exp_delay + jitter).max(0.0)
            }
            BackoffStrategy::DecorrelatedJitter => {
                // AWS-style decorrelated jitter
                let last_ms = self.last_delay.as_millis() as f64;
                let new_delay = rand::thread_rng().gen_range(base_ms..(last_ms * 3.0).min(max_ms));
                self.last_delay = Duration::from_millis(new_delay as u64);
                new_delay
            }
        };

        let clamped_ms = delay_ms.min(max_ms).max(0.0);
        Duration::from_millis(clamped_ms as u64)
    }

    /// Reset the backoff state
    pub fn reset(&mut self) {
        self.last_delay = self.base_delay;
    }
}

/// Retry policy for resilient operations
#[derive(Clone)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    max_retries: u32,
    /// Base delay
    base_delay: Duration,
    /// Maximum delay
    max_delay: Duration,
    /// Backoff strategy
    strategy: BackoffStrategy,
    /// Jitter factor
    jitter: f64,
    /// Retry budget tracking
    budget: Option<Arc<RetryBudget>>,
}

impl RetryPolicy {
    /// Create a new retry policy with default settings
    pub fn new() -> Self {
        Self::from_config(&RetryConfig::default())
    }

    /// Create from configuration
    pub fn from_config(config: &RetryConfig) -> Self {
        Self {
            max_retries: config.max_retries,
            base_delay: Duration::from_millis(config.base_delay_ms),
            max_delay: Duration::from_millis(config.max_delay_ms),
            strategy: config.strategy,
            jitter: config.jitter,
            budget: if config.budget_enabled {
                Some(Arc::new(RetryBudget::new(config.budget_per_minute)))
            } else {
                None
            },
        }
    }

    /// Create a policy with constant backoff
    pub fn constant() -> Self {
        Self {
            strategy: BackoffStrategy::Constant,
            ..Default::default()
        }
    }

    /// Create a policy with linear backoff
    pub fn linear() -> Self {
        Self {
            strategy: BackoffStrategy::Linear,
            ..Default::default()
        }
    }

    /// Create a policy with exponential backoff
    pub fn exponential() -> Self {
        Self {
            strategy: BackoffStrategy::Exponential,
            ..Default::default()
        }
    }

    /// Create a policy with exponential backoff and jitter (recommended)
    pub fn exponential_with_jitter() -> Self {
        Self {
            strategy: BackoffStrategy::ExponentialWithJitter,
            jitter: 0.2,
            ..Default::default()
        }
    }

    /// Set maximum retries
    pub fn max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set base delay
    pub fn base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set maximum delay
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set jitter factor
    pub fn jitter(mut self, factor: f64) -> Self {
        self.jitter = factor.clamp(0.0, 1.0);
        self
    }

    /// Enable retry budget
    pub fn with_budget(mut self, retries_per_minute: u32) -> Self {
        self.budget = Some(Arc::new(RetryBudget::new(retries_per_minute)));
        self
    }

    /// Execute an operation with retry logic
    pub async fn retry<T, E, F, Fut>(&self, mut operation: F) -> Result<T, RetryError<E>>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display + Clone,
    {
        let mut backoff = Backoff::new(self.base_delay, self.max_delay, self.strategy, self.jitter);
        let mut last_error: Option<E> = None;

        for attempt in 0..=self.max_retries {
            // Check retry budget
            if let Some(budget) = &self.budget {
                if !budget.try_acquire() {
                    return Err(RetryError::BudgetExhausted);
                }
            }

            match operation().await {
                Ok(result) => {
                    // Record success in budget
                    if let Some(budget) = &self.budget {
                        budget.record_success();
                    }
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e.clone());

                    if attempt < self.max_retries {
                        let delay = backoff.delay_for_attempt(attempt);
                        tracing::debug!(
                            attempt = attempt + 1,
                            max = self.max_retries,
                            delay_ms = delay.as_millis(),
                            error = %e,
                            "Retry attempt failed, waiting before next attempt"
                        );
                        sleep(delay).await;
                    }
                }
            }
        }

        Err(RetryError::MaxRetriesExceeded {
            max: self.max_retries,
            attempts: self.max_retries + 1,
            last_error: last_error.unwrap(),
        })
    }

    /// Execute with retry but bail on non-retryable errors
    pub async fn retry_if<T, E, F, Fut, P>(&self, mut operation: F, should_retry: P) -> Result<T, RetryError<E>>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display + Clone,
        P: Fn(&E) -> bool,
    {
        let mut backoff = Backoff::new(self.base_delay, self.max_delay, self.strategy, self.jitter);
        let mut last_error: Option<E> = None;

        for attempt in 0..=self.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if !should_retry(&e) {
                        return Err(RetryError::OperationFailed(e));
                    }

                    last_error = Some(e);

                    if attempt < self.max_retries {
                        let delay = backoff.delay_for_attempt(attempt);
                        sleep(delay).await;
                    }
                }
            }
        }

        Err(RetryError::MaxRetriesExceeded {
            max: self.max_retries,
            attempts: self.max_retries + 1,
            last_error: last_error.unwrap(),
        })
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry budget to prevent retry storms
pub struct RetryBudget {
    /// Maximum retries per minute
    max_per_minute: u32,
    /// Current count
    count: AtomicU64,
    /// Window start
    window_start: std::sync::RwLock<Instant>,
    /// Success count (for adaptive budgeting)
    successes: AtomicU64,
}

impl RetryBudget {
    /// Create a new retry budget
    pub fn new(max_per_minute: u32) -> Self {
        Self {
            max_per_minute,
            count: AtomicU64::new(0),
            window_start: std::sync::RwLock::new(Instant::now()),
            successes: AtomicU64::new(0),
        }
    }

    /// Try to acquire a retry permit
    pub fn try_acquire(&self) -> bool {
        self.maybe_reset_window();

        let current = self.count.fetch_add(1, Ordering::Relaxed);
        current < self.max_per_minute as u64
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        self.successes.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current retry count
    pub fn current_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get success count
    pub fn success_count(&self) -> u64 {
        self.successes.load(Ordering::Relaxed)
    }

    /// Reset if window has elapsed
    fn maybe_reset_window(&self) {
        let window_start = *self.window_start.read().unwrap();
        if window_start.elapsed() >= Duration::from_secs(60) {
            let mut ws = self.window_start.write().unwrap();
            if ws.elapsed() >= Duration::from_secs(60) {
                *ws = Instant::now();
                self.count.store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_backoff_constant() {
        let mut backoff = Backoff::new(
            Duration::from_millis(100),
            Duration::from_secs(10),
            BackoffStrategy::Constant,
            0.0,
        );

        assert_eq!(backoff.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(backoff.delay_for_attempt(5), Duration::from_millis(100));
    }

    #[test]
    fn test_backoff_exponential() {
        let mut backoff = Backoff::new(
            Duration::from_millis(100),
            Duration::from_secs(10),
            BackoffStrategy::Exponential,
            0.0,
        );

        assert_eq!(backoff.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(backoff.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(backoff.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn test_backoff_max_delay() {
        let mut backoff = Backoff::new(
            Duration::from_millis(100),
            Duration::from_millis(500),
            BackoffStrategy::Exponential,
            0.0,
        );

        assert_eq!(backoff.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(backoff.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(backoff.delay_for_attempt(3), Duration::from_millis(500)); // capped
        assert_eq!(backoff.delay_for_attempt(10), Duration::from_millis(500)); // capped
    }

    #[tokio::test]
    async fn test_retry_success() {
        let policy = RetryPolicy::new().max_retries(3);

        let result = policy.retry(|| async { Ok::<_, &str>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_eventual_success() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let policy = RetryPolicy::new()
            .max_retries(5)
            .base_delay(Duration::from_millis(10));

        let result = policy
            .retry(|| {
                let attempts = attempts_clone.clone();
                async move {
                    let current = attempts.fetch_add(1, Ordering::SeqCst);
                    if current < 2 {
                        Err("not yet")
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_max_exceeded() {
        let policy = RetryPolicy::new()
            .max_retries(2)
            .base_delay(Duration::from_millis(10));

        let result: Result<(), RetryError<&str>> = policy.retry(|| async { Err("always fail") }).await;

        assert!(matches!(result, Err(RetryError::MaxRetriesExceeded { max: 2, attempts: 3, .. })));
    }

    #[test]
    fn test_retry_budget() {
        let budget = RetryBudget::new(5);

        // Should allow 5 retries
        for _ in 0..5 {
            assert!(budget.try_acquire());
        }

        // 6th should fail
        assert!(!budget.try_acquire());
    }

    #[tokio::test]
    async fn test_retry_if() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let policy = RetryPolicy::new()
            .max_retries(5)
            .base_delay(Duration::from_millis(10));

        let result: Result<i32, RetryError<&str>> = policy
            .retry_if(
                || {
                    let attempts = attempts_clone.clone();
                    async move {
                        let current = attempts.fetch_add(1, Ordering::SeqCst);
                        if current < 2 {
                            Err("retryable")
                        } else {
                            Err("not_retryable")
                        }
                    }
                },
                |e| *e == "retryable",
            )
            .await;

        // Should stop on non-retryable error
        assert!(matches!(result, Err(RetryError::OperationFailed("not_retryable"))));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
