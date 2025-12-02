//! Benchmark I/O operations
//!
//! Handles reading and writing benchmark results to the filesystem.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;

use super::BenchmarkResult;

/// Default output directory relative to project root
pub const OUTPUT_DIR: &str = "src/benchmarks/output";

/// Raw results subdirectory
pub const RAW_DIR: &str = "raw";

/// Summary file name
pub const SUMMARY_FILE: &str = "summary.md";

/// Get the canonical output directory path
pub fn output_dir() -> PathBuf {
    PathBuf::from(OUTPUT_DIR)
}

/// Get the raw output directory path
pub fn raw_output_dir() -> PathBuf {
    output_dir().join(RAW_DIR)
}

/// Ensure output directories exist
pub fn ensure_output_dirs() -> Result<()> {
    let output = output_dir();
    let raw = raw_output_dir();

    fs::create_dir_all(&output)
        .with_context(|| format!("Failed to create output directory: {:?}", output))?;
    fs::create_dir_all(&raw)
        .with_context(|| format!("Failed to create raw output directory: {:?}", raw))?;

    Ok(())
}

/// Write a single benchmark result to the raw output directory
pub fn write_raw_result(result: &BenchmarkResult) -> Result<PathBuf> {
    ensure_output_dirs()?;

    let filename = format!(
        "{}_{}.json",
        result.target_id,
        result.timestamp.format("%Y%m%d_%H%M%S")
    );
    let path = raw_output_dir().join(&filename);

    let file = File::create(&path)
        .with_context(|| format!("Failed to create result file: {:?}", path))?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, result)
        .with_context(|| format!("Failed to write result to: {:?}", path))?;

    Ok(path)
}

/// Write all benchmark results to a combined JSON file
pub fn write_results(results: &[BenchmarkResult]) -> Result<PathBuf> {
    ensure_output_dirs()?;

    let filename = format!("results_{}.json", Utc::now().format("%Y%m%d_%H%M%S"));
    let path = output_dir().join(&filename);

    let file = File::create(&path)
        .with_context(|| format!("Failed to create results file: {:?}", path))?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, results)
        .with_context(|| format!("Failed to write results to: {:?}", path))?;

    Ok(path)
}

/// Write latest results to a fixed filename for easy access
pub fn write_latest_results(results: &[BenchmarkResult]) -> Result<PathBuf> {
    ensure_output_dirs()?;

    let path = output_dir().join("latest.json");

    let file = File::create(&path)
        .with_context(|| format!("Failed to create latest results file: {:?}", path))?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, results)
        .with_context(|| format!("Failed to write latest results to: {:?}", path))?;

    Ok(path)
}

/// Read benchmark results from a JSON file
pub fn read_results(path: &Path) -> Result<Vec<BenchmarkResult>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open results file: {:?}", path))?;
    let reader = BufReader::new(file);

    let results: Vec<BenchmarkResult> = serde_json::from_reader(reader)
        .with_context(|| format!("Failed to parse results from: {:?}", path))?;

    Ok(results)
}

/// Read the latest results file
pub fn read_latest_results() -> Result<Vec<BenchmarkResult>> {
    let path = output_dir().join("latest.json");
    read_results(&path)
}

/// List all result files in the output directory
pub fn list_result_files() -> Result<Vec<PathBuf>> {
    let output = output_dir();
    if !output.exists() {
        return Ok(vec![]);
    }

    let mut files: Vec<PathBuf> = fs::read_dir(&output)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().map_or(false, |ext| ext == "json")
                && path.file_name().map_or(false, |name| name != "latest.json")
        })
        .collect();

    files.sort_by(|a, b| b.cmp(a)); // Sort descending (newest first)
    Ok(files)
}

/// List all raw result files
pub fn list_raw_result_files() -> Result<Vec<PathBuf>> {
    let raw = raw_output_dir();
    if !raw.exists() {
        return Ok(vec![]);
    }

    let mut files: Vec<PathBuf> = fs::read_dir(&raw)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().map_or(false, |ext| ext == "json"))
        .collect();

    files.sort_by(|a, b| b.cmp(a));
    Ok(files)
}

/// Write content to a file in the output directory
pub fn write_output_file(filename: &str, content: &str) -> Result<PathBuf> {
    ensure_output_dirs()?;

    let path = output_dir().join(filename);
    let mut file = File::create(&path)
        .with_context(|| format!("Failed to create file: {:?}", path))?;

    file.write_all(content.as_bytes())
        .with_context(|| format!("Failed to write to: {:?}", path))?;

    Ok(path)
}

/// Clean up old result files, keeping only the N most recent
pub fn cleanup_old_results(keep_count: usize) -> Result<usize> {
    let files = list_result_files()?;
    let mut removed = 0;

    for file in files.into_iter().skip(keep_count) {
        fs::remove_file(&file)?;
        removed += 1;
    }

    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_output_dir_paths() {
        let output = output_dir();
        let raw = raw_output_dir();

        assert!(output.ends_with("output"));
        assert!(raw.ends_with("raw"));
    }

    #[test]
    fn test_write_and_read_results() {
        let results = vec![
            BenchmarkResult::new("test-1", serde_json::json!({"value": 1})),
            BenchmarkResult::new("test-2", serde_json::json!({"value": 2})),
        ];

        // Create temp directory for test
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_results.json");

        let file = File::create(&path).unwrap();
        serde_json::to_writer_pretty(file, &results).unwrap();

        let loaded = read_results(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].target_id(), "test-1");
    }
}
