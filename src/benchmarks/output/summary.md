# Benchmark Summary

Last Run: Not yet executed

## Results: 0/0 Passed

### Benchmark Targets

Run `llm-simulator run --output` to execute benchmarks and generate results.

Available benchmark targets:

- **latency-sampling**: Measures latency distribution sampling throughput
- **ttft-sampling**: Measures Time-To-First-Token sampling performance across profiles
- **itl-sampling**: Measures Inter-Token-Latency sampling performance
- **schedule-generation**: Measures latency schedule generation for streaming responses
- **distribution-sampling**: Measures sampling performance across all distribution types
- **token-throughput**: Measures token generation throughput
- **response-generation**: Measures response generation across different strategies
- **tokenization**: Measures text tokenization performance for streaming
- **concurrent-requests**: Simulates concurrent request processing stress
- **large-context**: Measures handling of large context windows
- **embedding-generation**: Measures embedding vector generation performance
- **batch-embedding**: Measures batch embedding generation performance
- **profile-lookup**: Measures latency profile lookup performance

## Usage

```bash
# List available benchmarks
llm-simulator run --list

# Run all benchmarks
llm-simulator run

# Run specific benchmarks
llm-simulator run --targets latency-sampling,token-throughput

# Run and save results
llm-simulator run --output

# Output as JSON
llm-simulator run --format json

# Output as Markdown
llm-simulator run --format markdown
```

---

*This file will be updated when benchmarks are executed with `--output` flag*
