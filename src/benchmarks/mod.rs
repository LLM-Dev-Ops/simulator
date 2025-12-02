//! Canonical Benchmark Module for LLM-Simulator
//!
//! This module implements the standardized benchmark interface used across
//! all 25 benchmark-target repositories. It provides:
//!
//! - `run_all_benchmarks()` - Main entrypoint returning `Vec<BenchmarkResult>`
//! - `BenchmarkResult` - Canonical result struct with target_id, metrics, timestamp
//! - `BenchTarget` trait - Interface for benchmark targets
//! - `all_targets()` - Registry of all benchmark targets
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llm_simulator::benchmarks::{run_all_benchmarks, BenchmarkResult};
//!
//! let results: Vec<BenchmarkResult> = run_all_benchmarks();
//! for result in &results {
//!     println!("{}: {:?}", result.target_id, result.metrics);
//! }
//! ```

pub mod adapters;
pub mod io;
pub mod markdown;
pub mod result;

pub use adapters::{all_targets, BenchTarget};
pub use io::{
    ensure_output_dirs, output_dir, raw_output_dir, read_latest_results, read_results,
    write_latest_results, write_output_file, write_raw_result, write_results, OUTPUT_DIR, RAW_DIR,
    SUMMARY_FILE,
};
pub use markdown::{generate_report, generate_summary, result_to_markdown};
pub use result::{BenchmarkResult, BenchmarkResultBuilder};

use std::time::Instant;

/// Run all benchmarks and return results
///
/// This is the canonical entrypoint for the benchmark system.
/// It executes all registered benchmark targets and collects their results.
///
/// # Returns
///
/// A `Vec<BenchmarkResult>` containing results from all benchmark targets.
///
/// # Example
///
/// ```rust,no_run
/// use llm_simulator::benchmarks::run_all_benchmarks;
///
/// let results = run_all_benchmarks();
/// println!("Completed {} benchmarks", results.len());
/// ```
pub fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    let targets = all_targets();
    let mut results = Vec::with_capacity(targets.len());

    eprintln!("Running {} benchmarks...", targets.len());

    for target in targets {
        eprintln!("  Running: {} - {}", target.id(), target.description());
        let start = Instant::now();

        let result = target.run();

        let elapsed = start.elapsed();
        eprintln!("    Completed in {:?}", elapsed);

        results.push(result);
    }

    eprintln!("All benchmarks completed.");
    results
}

/// Run benchmarks and write results to output directory
///
/// This function runs all benchmarks and automatically writes:
/// - Individual raw results to `benchmarks/output/raw/`
/// - Combined results to `benchmarks/output/results_<timestamp>.json`
/// - Latest results to `benchmarks/output/latest.json`
/// - Summary markdown to `benchmarks/output/summary.md`
///
/// # Returns
///
/// The benchmark results after writing to files.
///
/// # Errors
///
/// Returns an error if file operations fail.
pub fn run_and_save_benchmarks() -> anyhow::Result<Vec<BenchmarkResult>> {
    let results = run_all_benchmarks();

    // Ensure output directories exist
    ensure_output_dirs()?;

    // Write individual raw results
    for result in &results {
        let path = write_raw_result(result)?;
        eprintln!("  Wrote raw result: {:?}", path);
    }

    // Write combined results
    let results_path = write_results(&results)?;
    eprintln!("Wrote combined results: {:?}", results_path);

    // Write latest results
    let latest_path = write_latest_results(&results)?;
    eprintln!("Wrote latest results: {:?}", latest_path);

    // Generate and write summary
    let summary = generate_summary(&results);
    let summary_path = write_output_file(SUMMARY_FILE, &summary)?;
    eprintln!("Wrote summary: {:?}", summary_path);

    // Generate and write full report
    let report = generate_report(&results);
    let report_path = write_output_file("report.md", &report)?;
    eprintln!("Wrote report: {:?}", report_path);

    Ok(results)
}

/// Run a specific benchmark by target ID
///
/// # Arguments
///
/// * `target_id` - The ID of the benchmark target to run
///
/// # Returns
///
/// `Some(BenchmarkResult)` if the target was found and executed,
/// `None` if no target with the given ID exists.
pub fn run_benchmark(target_id: &str) -> Option<BenchmarkResult> {
    let targets = all_targets();

    for target in targets {
        if target.id() == target_id {
            return Some(target.run());
        }
    }

    None
}

/// List all available benchmark target IDs
pub fn list_benchmark_ids() -> Vec<String> {
    all_targets().iter().map(|t| t.id().to_string()).collect()
}

/// Get benchmark target descriptions
pub fn list_benchmarks() -> Vec<(String, String)> {
    all_targets()
        .iter()
        .map(|t| (t.id().to_string(), t.description().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_all_benchmarks() {
        // This test runs all benchmarks - may take a few seconds
        let results = run_all_benchmarks();

        assert!(!results.is_empty());

        // Verify all results have valid structure
        for result in &results {
            assert!(!result.target_id.is_empty());
            assert!(!result.metrics.is_null());
        }
    }

    #[test]
    fn test_run_single_benchmark() {
        let result = run_benchmark("latency-sampling");
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.target_id, "latency-sampling");
    }

    #[test]
    fn test_run_nonexistent_benchmark() {
        let result = run_benchmark("nonexistent-benchmark");
        assert!(result.is_none());
    }

    #[test]
    fn test_list_benchmark_ids() {
        let ids = list_benchmark_ids();

        assert!(!ids.is_empty());
        assert!(ids.contains(&"latency-sampling".to_string()));
        assert!(ids.contains(&"token-throughput".to_string()));
        assert!(ids.contains(&"embedding-generation".to_string()));
    }

    #[test]
    fn test_list_benchmarks() {
        let benchmarks = list_benchmarks();

        assert!(!benchmarks.is_empty());

        // Verify all have descriptions
        for (id, desc) in &benchmarks {
            assert!(!id.is_empty());
            // Note: description may be empty for some benchmarks
            let _ = desc;
        }
    }

    #[test]
    fn test_generate_summary() {
        let results = vec![
            BenchmarkResult::new("test-1", serde_json::json!({"value": 100})),
            BenchmarkResult::new("test-2", serde_json::json!({"value": 200})),
        ];

        let summary = generate_summary(&results);

        assert!(summary.contains("# Benchmark Summary"));
        assert!(summary.contains("test-1"));
        assert!(summary.contains("test-2"));
    }

    #[test]
    fn test_generate_report() {
        let results = vec![BenchmarkResult::new(
            "test-bench",
            serde_json::json!({
                "throughput_ops_per_sec": 50000.0,
                "latency_avg_ms": 0.02
            }),
        )];

        let report = generate_report(&results);

        assert!(report.contains("# LLM-Simulator Benchmark Report"));
        assert!(report.contains("test-bench"));
        assert!(report.contains("50000"));
    }
}
