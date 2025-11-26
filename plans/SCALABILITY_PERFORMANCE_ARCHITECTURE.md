# LLM-Simulator: Scalability and Performance Architecture

## Executive Summary

This document provides enterprise-grade architecture documentation for achieving high-performance scalability in LLM-Simulator, targeting 10,000+ requests per second with sub-5ms latency overhead.

**Performance Targets:**
- **Throughput:** 10,000+ RPS per node
- **Latency Overhead:** <5ms per request
- **P99 Latency:** <10ms overhead
- **Memory per Request:** <10KB
- **Cold Start:** <1 second
- **Concurrent Sessions:** 100,000+

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Horizontal Scaling Architecture](#2-horizontal-scaling-architecture)
3. [Vertical Scaling Optimization](#3-vertical-scaling-optimization)
4. [Performance Optimization Strategies](#4-performance-optimization-strategies)
5. [Resource Management](#5-resource-management)
6. [Caching Architecture](#6-caching-architecture)
7. [Load Balancing](#7-load-balancing)
8. [Benchmarking Methodology](#8-benchmarking-methodology)
9. [Capacity Planning](#9-capacity-planning)
10. [Performance Anti-Patterns](#10-performance-anti-patterns)
11. [Profiling and Tuning](#11-profiling-and-tuning)

---

## 1. Architecture Overview

### 1.1 System Architecture Diagram

```
                                    ┌─────────────────────┐
                                    │   Load Balancer     │
                                    │   (Layer 7/HTTP/2)  │
                                    │   - HAProxy/Envoy   │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
         ┌──────────▼──────────┐    ┌─────────▼──────────┐    ┌─────────▼──────────┐
         │  Simulator Node 1   │    │  Simulator Node 2  │    │  Simulator Node N  │
         │  ┌───────────────┐  │    │  ┌───────────────┐ │    │  ┌───────────────┐ │
         │  │ Tokio Runtime │  │    │  │ Tokio Runtime │ │    │  │ Tokio Runtime │ │
         │  │ (Work Stealing│  │    │  │ (Work Stealing│ │    │  │ (Work Stealing│ │
         │  │  Thread Pool) │  │    │  │  Thread Pool) │ │    │  │  Thread Pool) │ │
         │  └───────┬───────┘  │    │  └───────┬───────┘ │    │  └───────┬───────┘ │
         │          │          │    │          │         │    │          │         │
         │  ┌───────▼───────┐  │    │  ┌───────▼───────┐ │    │  ┌───────▼───────┐ │
         │  │  Axum Server  │  │    │  │  Axum Server  │ │    │  │  Axum Server  │ │
         │  │   (HTTP/2)    │  │    │  │   (HTTP/2)    │ │    │  │   (HTTP/2)    │ │
         │  └───────┬───────┘  │    │  └───────┬───────┘ │    │  └───────┬───────┘ │
         │          │          │    │          │         │    │          │         │
         │  ┌───────▼───────┐  │    │  ┌───────▼───────┐ │    │  ┌───────▼───────┐ │
         │  │ Request Layer │  │    │  │ Request Layer │ │    │  │ Request Layer │ │
         │  │ - Middleware  │  │    │  │ - Middleware  │ │    │  │ - Middleware  │ │
         │  │ - Routing     │  │    │  │ - Routing     │ │    │  │ - Routing     │ │
         │  └───────┬───────┘  │    │  └───────┬───────┘ │    │  └───────┬───────┘ │
         │          │          │    │          │         │    │          │         │
         │  ┌───────▼───────┐  │    │  ┌───────▼───────┐ │    │  ┌───────▼───────┐ │
         │  │ Lock-Free     │  │    │  │ Lock-Free     │ │    │  │ Lock-Free     │ │
         │  │ Data Structs  │  │    │  │ Data Structs  │ │    │  │ Data Structs  │ │
         │  └───────┬───────┘  │    │  └───────┬───────┘ │    │  └───────┬───────┘ │
         │          │          │    │          │         │    │          │         │
         │  ┌───────▼───────┐  │    │  ┌───────▼───────┐ │    │  ┌───────▼───────┐ │
         │  │   Simulation  │  │    │  │   Simulation  │ │    │  │   Simulation  │ │
         │  │     Engine    │  │    │  │     Engine    │ │    │  │     Engine    │ │
         │  └───────────────┘  │    │  └───────────────┘ │    │  └───────────────┘ │
         └─────────────────────┘    └────────────────────┘    └────────────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Shared Services   │
                                    │  - Redis (Metrics)  │
                                    │  - Prometheus       │
                                    │  - Config Store     │
                                    └─────────────────────┘
```

### 1.2 Performance Characteristics

| Component | Latency Contribution | Optimization Strategy |
|-----------|---------------------|----------------------|
| HTTP Parsing | 0.1-0.5ms | Zero-copy parsing, HTTP/2 |
| Request Routing | 0.05-0.1ms | Pre-compiled regex, hash maps |
| Middleware Stack | 0.2-0.5ms | Async, minimal allocations |
| Simulation Engine | 2-3ms | Lock-free, CPU cache optimization |
| Response Serialization | 0.3-0.8ms | Pre-allocated buffers, SIMD |
| Network I/O | 0.5-1ms | TCP_NODELAY, SO_REUSEPORT |
| **Total Overhead** | **3.15-5.9ms** | **Target: <5ms average** |

---

## 2. Horizontal Scaling Architecture

### 2.1 Stateless Design

**Design Principles:**
- Each simulator node is fully stateless
- No session affinity required
- Configuration loaded from shared store
- Metrics pushed to centralized collectors

**Scaling Formula:**
```
Total Capacity = N × (Node_RPS × Efficiency_Factor)

Where:
  N = Number of nodes
  Node_RPS = 10,000-15,000 RPS per node (8 CPU cores)
  Efficiency_Factor = 0.85-0.95 (accounting for load balancer overhead)

Example:
  10 nodes × 12,000 RPS × 0.90 = 108,000 total RPS
```

### 2.2 Node Deployment Patterns

#### Pattern 1: Homogeneous Cluster
```yaml
# All nodes identical configuration
deployment:
  type: homogeneous
  node_count: 10
  cpu_per_node: 8
  memory_per_node: 16GB

scaling:
  metric: cpu_utilization
  target: 70%
  min_nodes: 3
  max_nodes: 50
```

#### Pattern 2: Heterogeneous Cluster
```yaml
# Specialized nodes for different workloads
deployment:
  pools:
    - name: high_throughput
      node_count: 5
      cpu: 16
      memory: 32GB
      target: streaming_requests

    - name: low_latency
      node_count: 5
      cpu: 8
      memory: 16GB
      target: non_streaming_requests
```

### 2.3 Service Discovery

```rust
// Rust implementation for node registration
use consul::Client as ConsulClient;

pub struct NodeRegistry {
    consul: ConsulClient,
    node_id: String,
    health_check_interval: Duration,
}

impl NodeRegistry {
    pub async fn register(&self) -> Result<(), RegistryError> {
        let service = ServiceDefinition {
            id: self.node_id.clone(),
            name: "llm-simulator",
            address: self.get_local_ip(),
            port: 8080,
            tags: vec!["http", "simulator", "v1"],
            check: HealthCheck {
                http: format!("http://{}:8080/health", self.get_local_ip()),
                interval: self.health_check_interval,
                timeout: Duration::from_secs(5),
            },
        };

        self.consul.register_service(service).await
    }

    pub async fn heartbeat(&self) -> Result<(), RegistryError> {
        // Update TTL to keep registration alive
        self.consul.agent_check_pass(&format!("service:{}", self.node_id)).await
    }
}
```

### 2.4 Load Distribution Strategy

**Request Distribution Algorithm:**
```
1. Round-robin with health-aware routing
2. Least-connections for long-lived connections
3. Consistent hashing for session affinity (if needed)
4. Weighted distribution based on node capacity
```

**Implementation:**
```rust
pub enum LoadBalanceStrategy {
    RoundRobin,
    LeastConnections,
    ConsistentHash { hash_key: HashKey },
    Weighted { weights: HashMap<String, f64> },
    LatencyBased { window: Duration },
}

impl LoadBalancer {
    pub fn select_node(&mut self, request: &Request) -> Result<NodeId, LBError> {
        match &self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed);
                let node = &self.healthy_nodes[idx % self.healthy_nodes.len()];
                Ok(node.id.clone())
            }

            LoadBalanceStrategy::LeastConnections => {
                self.healthy_nodes
                    .iter()
                    .min_by_key(|n| n.active_connections.load(Ordering::Relaxed))
                    .map(|n| n.id.clone())
                    .ok_or(LBError::NoHealthyNodes)
            }

            LoadBalanceStrategy::LatencyBased { window } => {
                // Select node with lowest p99 latency in recent window
                self.select_by_performance(*window)
            }

            _ => todo!("Other strategies"),
        }
    }
}
```

---

## 3. Vertical Scaling Optimization

### 3.1 CPU Optimization

**Tokio Runtime Configuration:**
```rust
use tokio::runtime::{Builder, Runtime};

pub fn create_optimized_runtime(num_cores: usize) -> Runtime {
    Builder::new_multi_thread()
        // Worker threads = CPU cores
        .worker_threads(num_cores)

        // Thread stack size (reduce for memory efficiency)
        .thread_stack_size(2 * 1024 * 1024) // 2MB

        // Thread naming for profiling
        .thread_name("simulator-worker")

        // Enable I/O driver
        .enable_io()

        // Enable time driver for delays
        .enable_time()

        // Max blocking threads for sync operations
        .max_blocking_threads(512)

        // Thread park timeout
        .thread_keep_alive(Duration::from_secs(10))

        // Event interval for better fairness
        .event_interval(61)

        .build()
        .expect("Failed to create runtime")
}
```

**CPU Affinity and NUMA Awareness:**
```rust
use core_affinity::{set_for_current, CoreId};

pub fn pin_worker_threads(num_cores: usize) {
    let core_ids: Vec<CoreId> = core_affinity::get_core_ids()
        .unwrap()
        .into_iter()
        .take(num_cores)
        .collect();

    for (idx, core_id) in core_ids.iter().enumerate() {
        std::thread::Builder::new()
            .name(format!("worker-{}", idx))
            .spawn(move || {
                // Pin this thread to specific core
                set_for_current(*core_id);

                // Worker loop
                run_worker_loop();
            })
            .unwrap();
    }
}
```

### 3.2 Memory Optimization

**Stack vs Heap Allocation:**
```rust
// GOOD: Stack allocation for hot path
pub fn process_request_fast(req: &Request) -> Response {
    // Fixed-size buffer on stack
    let mut buffer = [0u8; 4096];

    // Process inline without heap allocation
    let response_size = serialize_response(&req, &mut buffer);

    Response::from_slice(&buffer[..response_size])
}

// BAD: Heap allocation in hot path
pub fn process_request_slow(req: &Request) -> Response {
    // Heap allocation on every request
    let buffer = Vec::with_capacity(4096);

    // Additional allocation for String
    let json = serde_json::to_string(&req).unwrap();

    Response::new(json)
}
```

**Memory Pool for Request Buffers:**
```rust
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

pub struct BufferPool {
    pool: Arc<ArrayQueue<Vec<u8>>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl BufferPool {
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        let pool = Arc::new(ArrayQueue::new(max_buffers));

        // Pre-allocate buffers
        for _ in 0..max_buffers {
            let buffer = Vec::with_capacity(buffer_size);
            pool.push(buffer).ok();
        }

        Self { pool, buffer_size, max_buffers }
    }

    pub fn acquire(&self) -> PooledBuffer {
        let buffer = self.pool.pop().unwrap_or_else(|| {
            // Pool exhausted, allocate new
            Vec::with_capacity(self.buffer_size)
        });

        PooledBuffer {
            buffer,
            pool: Arc::clone(&self.pool),
        }
    }
}

pub struct PooledBuffer {
    buffer: Vec<u8>,
    pool: Arc<ArrayQueue<Vec<u8>>>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool
        let mut buffer = std::mem::take(&mut self.buffer);
        buffer.clear();
        self.pool.push(buffer).ok();
    }
}
```

**Memory Layout Optimization:**
```rust
// Cache-line aligned structures for hot data
#[repr(align(64))]  // 64-byte cache line
pub struct RequestContext {
    // Hot fields first
    pub request_id: u64,
    pub start_time: Instant,
    pub priority: u8,

    // Padding to fill cache line
    _padding1: [u8; 64 - 17],

    // Cold fields in next cache line
    pub metadata: RequestMetadata,
}

// Compact representation
#[derive(Clone, Copy)]
pub struct CompactRequest {
    // Pack fields efficiently
    pub flags: u32,        // 4 bytes
    pub token_count: u16,  // 2 bytes
    pub priority: u8,      // 1 byte
    pub _pad: u8,          // 1 byte (alignment)
}
```

### 3.3 Resource Limits per Node

**Recommended Configuration:**
```yaml
resources:
  cpu:
    cores: 8-16
    governor: performance  # CPU frequency scaling

  memory:
    limit: 16-32GB
    swap: disabled  # Disable swap for predictable latency
    transparent_hugepages: madvise

  network:
    max_connections: 100000
    tcp_backlog: 4096
    rmem_max: 134217728  # 128MB
    wmem_max: 134217728  # 128MB

  file_descriptors:
    soft_limit: 1048576
    hard_limit: 1048576
```

---

## 4. Performance Optimization Strategies

### 4.1 Performance Strategy Matrix

| Strategy | Benefit | Cost | Priority | Implementation Complexity |
|----------|---------|------|----------|--------------------------|
| Lock-free data structures | 30-50% throughput↑ | Medium | High | High |
| Zero-copy I/O | 20-30% latency↓ | Low | High | Medium |
| Connection pooling | 40-60% overhead↓ | Low | High | Low |
| SIMD operations | 2-4x specific ops↑ | Medium | Medium | High |
| Pre-allocated buffers | 15-25% latency↓ | Low | High | Low |
| Async I/O (Tokio) | 10x concurrency↑ | Low | High | Medium |
| Cache-line optimization | 10-20% throughput↑ | Medium | Medium | High |
| Minimal allocations | 20-30% latency↓ | Low | High | Medium |

### 4.2 Lock-Free Data Structures

**Concurrent Request Queue:**
```rust
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct RequestQueue {
    queue: SegQueue<PendingRequest>,
    enqueued: AtomicU64,
    dequeued: AtomicU64,
}

impl RequestQueue {
    pub fn push(&self, req: PendingRequest) {
        self.queue.push(req);
        self.enqueued.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pop(&self) -> Option<PendingRequest> {
        let req = self.queue.pop();
        if req.is_some() {
            self.dequeued.fetch_add(1, Ordering::Relaxed);
        }
        req
    }

    pub fn depth(&self) -> u64 {
        self.enqueued.load(Ordering::Relaxed)
            - self.dequeued.load(Ordering::Relaxed)
    }
}
```

**Lock-Free Metrics Collection:**
```rust
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(align(64))]  // Prevent false sharing
pub struct MetricsCounter {
    value: AtomicU64,
    _padding: [u8; 56],
}

pub struct Metrics {
    requests_total: MetricsCounter,
    requests_success: MetricsCounter,
    requests_error: MetricsCounter,
    latency_sum_ns: AtomicU64,
}

impl Metrics {
    pub fn record_request(&self, duration: Duration, success: bool) {
        self.requests_total.value.fetch_add(1, Ordering::Relaxed);

        if success {
            self.requests_success.value.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_error.value.fetch_add(1, Ordering::Relaxed);
        }

        self.latency_sum_ns.fetch_add(
            duration.as_nanos() as u64,
            Ordering::Relaxed
        );
    }
}
```

### 4.3 Zero-Copy Techniques

**Direct Buffer Manipulation:**
```rust
use bytes::{Bytes, BytesMut, Buf, BufMut};

pub fn serialize_response_zerocopy(
    response: &SimulationResponse,
    mut buffer: BytesMut,
) -> Bytes {
    // Reserve exact space needed
    buffer.reserve(estimate_size(response));

    // Write header directly
    buffer.put_slice(b"HTTP/1.1 200 OK\r\n");
    buffer.put_slice(b"Content-Type: application/json\r\n\r\n");

    // Serialize directly to buffer (no intermediate allocation)
    serde_json::to_writer(
        (&mut buffer).writer(),
        response
    ).unwrap();

    // Freeze to make immutable Bytes (zero-copy)
    buffer.freeze()
}
```

**Shared Reference Counting:**
```rust
use bytes::Bytes;
use std::sync::Arc;

pub struct SharedResponse {
    // Reference-counted, allows zero-copy sharing
    body: Bytes,
    headers: Arc<HeaderMap>,
}

impl Clone for SharedResponse {
    fn clone(&self) -> Self {
        // Shallow copy - no data duplication
        Self {
            body: self.body.clone(),     // Bytes::clone is cheap (rc bump)
            headers: Arc::clone(&self.headers),  // Arc::clone is cheap
        }
    }
}
```

### 4.4 SIMD Optimizations

**Vectorized Token Processing:**
```rust
use std::simd::{u32x8, SimdUint};

pub fn count_tokens_simd(text: &[u8]) -> usize {
    let spaces = b' ';
    let newlines = b'\n';
    let tabs = b'\t';

    let mut count = 0;
    let chunks = text.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Load 8 bytes into SIMD register
        let bytes = u32x8::from_array([
            chunk[0] as u32, chunk[1] as u32, chunk[2] as u32, chunk[3] as u32,
            chunk[4] as u32, chunk[5] as u32, chunk[6] as u32, chunk[7] as u32,
        ]);

        // Parallel comparison
        let is_space = bytes.simd_eq(u32x8::splat(spaces as u32));
        let is_newline = bytes.simd_eq(u32x8::splat(newlines as u32));
        let is_tab = bytes.simd_eq(u32x8::splat(tabs as u32));

        // Count matches
        count += (is_space | is_newline | is_tab).to_bitmask().count_ones();
    }

    // Handle remainder
    count += remainder.iter()
        .filter(|&&b| b == spaces || b == newlines || b == tabs)
        .count();

    count
}
```

---

## 5. Resource Management

### 5.1 Connection Pooling

**HTTP Client Connection Pool:**
```rust
use hyper::{Client, Body};
use hyper_tls::HttpsConnector;
use std::time::Duration;

pub struct ConnectionPool {
    client: Client<HttpsConnector<hyper::client::HttpConnector>>,
}

impl ConnectionPool {
    pub fn new(max_idle_per_host: usize) -> Self {
        let https = HttpsConnector::new();

        let client = Client::builder()
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(max_idle_per_host)
            .http2_only(true)
            .http2_initial_stream_window_size(65_535)
            .http2_initial_connection_window_size(1_048_576)
            .http2_adaptive_window(true)
            .http2_max_frame_size(16_384)
            .build(https);

        Self { client }
    }
}
```

### 5.2 Concurrency Control

**Semaphore-Based Rate Limiting:**
```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
    max_permits: usize,
}

impl ConcurrencyLimiter {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_permits: max_concurrent,
        }
    }

    pub async fn acquire(&self) -> Result<SemaphorePermit, LimiterError> {
        // Non-blocking try acquire with timeout
        tokio::time::timeout(
            Duration::from_millis(100),
            self.semaphore.acquire()
        )
        .await
        .map_err(|_| LimiterError::Timeout)?
        .map_err(|_| LimiterError::Closed)
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn utilization(&self) -> f64 {
        let available = self.available_permits();
        1.0 - (available as f64 / self.max_permits as f64)
    }
}
```

### 5.3 Memory Management

**Arena Allocator for Request Processing:**
```rust
use bumpalo::Bump;

pub struct RequestArena {
    arena: Bump,
}

impl RequestArena {
    pub fn new() -> Self {
        Self {
            arena: Bump::with_capacity(64 * 1024), // 64KB
        }
    }

    pub fn allocate<T>(&self, value: T) -> &mut T {
        self.arena.alloc(value)
    }

    pub fn reset(&mut self) {
        // Fast bulk deallocation
        self.arena.reset();
    }
}

// Usage in request handler
pub async fn handle_request(req: Request) -> Response {
    let mut arena = RequestArena::new();

    // All allocations go to arena
    let context = arena.allocate(RequestContext::new());
    let result = process_in_arena(&mut arena, context);

    // Arena automatically deallocated on drop
    result
}
```

### 5.4 Resource Scaling Formulas

**CPU Scaling:**
```
Required_Cores = (Target_RPS × Avg_CPU_Time_Per_Request) / CPU_Utilization_Target

Example:
  Target: 10,000 RPS
  CPU time per request: 2ms = 0.002s
  Target utilization: 70%

  Required_Cores = (10,000 × 0.002) / 0.70 = 28.6 cores
  Recommended: 32 cores (next power of 2)
```

**Memory Scaling:**
```
Required_Memory = Base_Memory + (Concurrent_Requests × Memory_Per_Request)

Example:
  Base memory (binary + caches): 2GB
  Concurrent requests at 10K RPS with 20ms avg latency: 10,000 × 0.02 = 200
  Memory per request: 10KB

  Required_Memory = 2GB + (200 × 10KB) = 2GB + 2MB ≈ 2.002GB
  Recommended: 4GB (with 2x safety margin)
```

**Network Bandwidth:**
```
Required_Bandwidth = (Avg_Request_Size + Avg_Response_Size) × Target_RPS

Example:
  Avg request: 1KB
  Avg response: 5KB
  Target: 10,000 RPS

  Required_Bandwidth = (1KB + 5KB) × 10,000 = 60MB/s
  Recommended: 100MB/s (1Gbps link with overhead)
```

---

## 6. Caching Architecture

### 6.1 Multi-Layer Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                     L1: In-Process Cache                     │
│                      (Thread-Local)                          │
│                    100ns access latency                      │
└────────────────────────────┬────────────────────────────────┘
                             │ Miss
┌────────────────────────────▼────────────────────────────────┐
│                     L2: Shared Memory                        │
│                    (DashMap/Arc<RwLock>)                     │
│                    500ns-1μs access latency                  │
└────────────────────────────┬────────────────────────────────┘
                             │ Miss
┌────────────────────────────▼────────────────────────────────┐
│                      L3: Redis Cache                         │
│                   (Distributed, Optional)                    │
│                    1-5ms access latency                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Cache Implementation

**Thread-Local L1 Cache:**
```rust
use std::cell::RefCell;
use lru::LruCache;

thread_local! {
    static PROFILE_CACHE: RefCell<LruCache<String, Arc<LatencyProfile>>> =
        RefCell::new(LruCache::new(std::num::NonZeroUsize::new(256).unwrap()));
}

pub fn get_profile_cached(key: &str) -> Option<Arc<LatencyProfile>> {
    PROFILE_CACHE.with(|cache| {
        cache.borrow_mut().get(key).cloned()
    })
}

pub fn cache_profile(key: String, profile: Arc<LatencyProfile>) {
    PROFILE_CACHE.with(|cache| {
        cache.borrow_mut().put(key, profile);
    });
}
```

**Shared L2 Cache:**
```rust
use dashmap::DashMap;
use std::sync::Arc;

pub struct SharedCache<K, V> {
    map: DashMap<K, CachedValue<V>>,
    ttl: Duration,
    max_size: usize,
}

struct CachedValue<V> {
    value: Arc<V>,
    inserted_at: Instant,
}

impl<K, V> SharedCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        self.map.get(key).and_then(|entry| {
            let cached = entry.value();

            // Check TTL
            if cached.inserted_at.elapsed() > self.ttl {
                drop(entry);
                self.map.remove(key);
                return None;
            }

            Some(Arc::clone(&cached.value))
        })
    }

    pub fn insert(&self, key: K, value: V) {
        // Evict if at capacity
        if self.map.len() >= self.max_size {
            self.evict_lru();
        }

        self.map.insert(key, CachedValue {
            value: Arc::new(value),
            inserted_at: Instant::now(),
        });
    }

    fn evict_lru(&self) {
        // Find oldest entry
        let oldest = self.map.iter()
            .min_by_key(|entry| entry.value().inserted_at)
            .map(|entry| entry.key().clone());

        if let Some(key) = oldest {
            self.map.remove(&key);
        }
    }
}
```

### 6.3 Cache Warming Strategy

**Startup Cache Preloading:**
```rust
pub async fn warm_caches(config: &SimulatorConfig) -> Result<(), WarmupError> {
    println!("Warming caches...");

    // Pre-load all latency profiles
    for (key, profile) in &config.providers {
        let latency_profile = LatencyProfile::from_config(&profile.latency)?;
        cache_profile(key.clone(), Arc::new(latency_profile));
    }

    // Pre-compile regex patterns
    for route in &config.routes {
        compile_and_cache_route_pattern(&route.pattern)?;
    }

    // Pre-allocate buffer pools
    initialize_buffer_pools(1000)?;

    println!("Cache warming complete");
    Ok(())
}
```

### 6.4 Cache Metrics

```rust
pub struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}
```

---

## 7. Load Balancing

### 7.1 Load Balancing Algorithms

**Algorithm Comparison:**

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| Round Robin | Simple, even distribution | Ignores node load | Homogeneous nodes |
| Least Connections | Load-aware | Requires tracking | Variable request times |
| Weighted Round Robin | Handles heterogeneous nodes | Static weights | Known capacity differences |
| Latency-Based | Performance-aware | Higher overhead | Quality-sensitive workloads |
| Consistent Hashing | Session affinity, cache hits | Uneven distribution | Stateful scenarios |

### 7.2 HAProxy Configuration

```haproxy
global
    maxconn 100000
    nbproc 4
    cpu-map auto:1/1-4 0-3

    # Performance tuning
    tune.ssl.default-dh-param 2048
    tune.bufsize 32768
    tune.maxrewrite 1024

defaults
    mode http
    timeout connect 5s
    timeout client 300s
    timeout server 300s

    # HTTP/2 support
    option http-use-htx

    # Health checks
    option httpchk GET /health
    http-check expect status 200

frontend llm_simulator
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/simulator.pem alpn h2,http/1.1

    # Request buffering
    option http-buffer-request

    # Compression
    compression algo gzip
    compression type application/json text/plain

    # Rate limiting
    stick-table type ip size 1m expire 10s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 1000 }

    default_backend simulator_nodes

backend simulator_nodes
    balance leastconn

    # Health checks
    option httpchk GET /health
    http-check expect status 200

    # Connection settings
    option http-server-close
    option forwardfor

    # Servers
    server node1 10.0.1.10:8080 check inter 5s fall 3 rise 2 maxconn 10000
    server node2 10.0.1.11:8080 check inter 5s fall 3 rise 2 maxconn 10000
    server node3 10.0.1.12:8080 check inter 5s fall 3 rise 2 maxconn 10000
    server node4 10.0.1.13:8080 check inter 5s fall 3 rise 2 maxconn 10000
```

### 7.3 Envoy Proxy Configuration

```yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 80

    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http

          # HTTP/2 settings
          http2_protocol_options:
            max_concurrent_streams: 1000
            initial_stream_window_size: 65536
            initial_connection_window_size: 1048576

          # Route configuration
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: simulator_cluster
                  timeout: 300s

          # Filters
          http_filters:
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
  - name: simulator_cluster
    connect_timeout: 5s
    type: STRICT_DNS

    # Load balancing
    lb_policy: LEAST_REQUEST
    lb_subset_config:
      fallback_policy: ANY_ENDPOINT

    # Health checks
    health_checks:
    - timeout: 5s
      interval: 10s
      unhealthy_threshold: 3
      healthy_threshold: 2
      http_health_check:
        path: "/health"
        expected_statuses:
        - start: 200
          end: 200

    # Circuit breaker
    circuit_breakers:
      thresholds:
      - priority: DEFAULT
        max_connections: 10000
        max_pending_requests: 10000
        max_requests: 100000
        max_retries: 3

    # Endpoints
    load_assignment:
      cluster_name: simulator_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.0.1.10
                port_value: 8080
        - endpoint:
            address:
              socket_address:
                address: 10.0.1.11
                port_value: 8080
```

---

## 8. Benchmarking Methodology

### 8.1 Benchmark Suite

**Load Testing Tools:**
```bash
# wrk - HTTP benchmarking
wrk -t12 -c400 -d30s --latency http://localhost:8080/v1/chat/completions

# vegeta - Constant rate testing
echo "POST http://localhost:8080/v1/chat/completions" | \
  vegeta attack -rate=10000/s -duration=60s -body=request.json | \
  vegeta report -type=text

# k6 - Scenario-based testing
k6 run --vus 1000 --duration 5m benchmark.js

# Gatling - Complex scenarios
gatling.sh -s SimulatorBenchmark
```

**Custom Rust Benchmark:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tokio::runtime::Runtime;

fn bench_chat_completion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = create_test_client();

    let mut group = c.benchmark_group("chat_completion");
    group.throughput(Throughput::Elements(1));

    group.bench_function("non_streaming", |b| {
        b.iter(|| {
            rt.block_on(async {
                let response = client
                    .chat_completion(black_box(create_test_request()))
                    .await
                    .unwrap();
                black_box(response);
            })
        })
    });

    group.bench_function("streaming", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut stream = client
                    .chat_completion_stream(black_box(create_test_request()))
                    .await
                    .unwrap();

                while let Some(chunk) = stream.next().await {
                    black_box(chunk.unwrap());
                }
            })
        })
    });

    group.finish();
}

criterion_group!(benches, bench_chat_completion);
criterion_main!(benches);
```

### 8.2 Performance Test Scenarios

**Scenario 1: Sustained Throughput**
```yaml
test: sustained_throughput
description: Measure maximum sustained RPS

configuration:
  duration: 300s
  ramp_up: 30s
  rate: 10000/s

metrics:
  - requests_per_second
  - p50_latency
  - p99_latency
  - error_rate
  - cpu_utilization
  - memory_usage

success_criteria:
  throughput: ">= 10000 RPS"
  p99_latency: "<= 10ms"
  error_rate: "<= 0.1%"
```

**Scenario 2: Burst Capacity**
```yaml
test: burst_capacity
description: Handle traffic spikes

configuration:
  baseline_rate: 1000/s
  burst_rate: 20000/s
  burst_duration: 10s
  cooldown: 30s
  iterations: 10

success_criteria:
  peak_throughput: ">= 20000 RPS"
  p99_latency_during_burst: "<= 50ms"
  recovery_time: "<= 5s"
```

**Scenario 3: Endurance Test**
```yaml
test: endurance
description: Long-running stability test

configuration:
  duration: 86400s  # 24 hours
  rate: 5000/s

monitors:
  - memory_leaks
  - connection_leaks
  - performance_degradation
  - error_rate_trend

success_criteria:
  memory_growth: "<= 10% over 24h"
  latency_degradation: "<= 5% over 24h"
  zero_crashes: true
```

### 8.3 Benchmark Results Template

```markdown
# Benchmark Results: LLM-Simulator v1.0

**Test Date:** 2024-01-15
**Configuration:** 8-core, 16GB RAM
**Load Balancer:** HAProxy 2.8

## Sustained Throughput Test

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Throughput | 10,000 RPS | 12,450 RPS | ✅ Pass |
| P50 Latency | <5ms | 3.2ms | ✅ Pass |
| P95 Latency | <8ms | 6.8ms | ✅ Pass |
| P99 Latency | <10ms | 9.2ms | ✅ Pass |
| Error Rate | <0.1% | 0.02% | ✅ Pass |
| CPU Utilization | 70-80% | 74% | ✅ Pass |
| Memory Usage | <8GB | 6.2GB | ✅ Pass |

## Latency Distribution
```
Percentile  Latency
50%         3.2ms
75%         4.8ms
90%         7.1ms
95%         6.8ms
99%         9.2ms
99.9%       15.3ms
```

## Resource Utilization
- Average CPU: 74%
- Peak CPU: 89%
- Average Memory: 6.2GB
- Peak Memory: 7.1GB
- Network I/O: 62MB/s avg, 95MB/s peak
```

---

## 9. Capacity Planning

### 9.1 Capacity Planning Framework

**Step 1: Define Requirements**
```yaml
requirements:
  target_rps: 50000
  peak_multiplier: 2.0  # 2x for peak traffic
  latency_sla:
    p50: 5ms
    p99: 10ms
  availability: 99.9%
  geographic_regions: 3
```

**Step 2: Calculate Base Capacity**
```
Base_Nodes = (Target_RPS × Peak_Multiplier) / (Node_RPS × Efficiency)

Example:
  Target: 50,000 RPS
  Peak: 2.0x = 100,000 RPS
  Node capacity: 12,000 RPS
  Efficiency: 0.85

  Base_Nodes = 100,000 / (12,000 × 0.85)
             = 100,000 / 10,200
             = 9.8 nodes

  Rounded up: 10 nodes per region
  Total (3 regions): 30 nodes
```

**Step 3: Add Redundancy**
```
Total_Nodes = Base_Nodes × (1 + Redundancy_Factor)

Example:
  Base: 10 nodes
  N+1 redundancy: 11 nodes
  N+2 redundancy: 12 nodes

  Recommended: 12 nodes per region (20% overhead)
  Total: 36 nodes
```

### 9.2 Growth Projection

**Traffic Growth Model:**
```python
import numpy as np
import matplotlib.pyplot as plt

def project_capacity(
    initial_rps: int,
    growth_rate: float,
    months: int
) -> list:
    """
    Project capacity needs based on growth rate

    Args:
        initial_rps: Current RPS
        growth_rate: Monthly growth (e.g., 0.15 for 15%)
        months: Projection period

    Returns:
        List of (month, required_rps, required_nodes)
    """
    projections = []

    for month in range(months):
        rps = initial_rps * (1 + growth_rate) ** month
        peak_rps = rps * 2.0  # Peak multiplier
        nodes_needed = np.ceil(peak_rps / 10200)  # 12K RPS × 0.85 eff

        projections.append((month, rps, nodes_needed))

    return projections

# Example usage
projections = project_capacity(
    initial_rps=50000,
    growth_rate=0.15,  # 15% monthly growth
    months=12
)

# Output:
# Month 0: 50,000 RPS → 10 nodes
# Month 6: 115,659 RPS → 23 nodes
# Month 12: 267,447 RPS → 53 nodes
```

### 9.3 Cost Optimization

**Node Cost Analysis:**
```yaml
cloud_provider: AWS

instance_types:
  - type: c6i.2xlarge
    vcpu: 8
    memory: 16GB
    cost_per_hour: $0.34
    estimated_rps: 12000
    cost_per_million_requests: $0.79

  - type: c6i.4xlarge
    vcpu: 16
    memory: 32GB
    cost_per_hour: $0.68
    estimated_rps: 20000
    cost_per_million_requests: $0.94

  - type: c6i.8xlarge
    vcpu: 32
    memory: 64GB
    cost_per_hour: $1.36
    estimated_rps: 35000
    cost_per_million_requests: $1.08

recommendation:
  optimal: c6i.2xlarge
  reason: "Best cost per million requests"
  monthly_cost_50k_rps: "$2,448/month (10 nodes × $0.34 × 730 hours)"
```

### 9.4 Scaling Decision Matrix

| Current Utilization | Action | Timeline | Priority |
|---------------------|--------|----------|----------|
| < 50% | Monitor | Quarterly review | Low |
| 50-70% | Plan scale | 1 month | Medium |
| 70-85% | Prepare scale | 2 weeks | High |
| 85-95% | Scale immediately | 24-48 hours | Critical |
| > 95% | Emergency scale | Immediate | Emergency |

**Auto-Scaling Rules:**
```yaml
autoscaling:
  enabled: true

  scale_up:
    - metric: cpu_utilization
      threshold: 75%
      duration: 300s
      action: add_1_node

    - metric: request_queue_depth
      threshold: 1000
      duration: 60s
      action: add_2_nodes

    - metric: p99_latency
      threshold: 15ms
      duration: 120s
      action: add_1_node

  scale_down:
    - metric: cpu_utilization
      threshold: 40%
      duration: 600s
      action: remove_1_node
      min_nodes: 3

  cooldown:
    scale_up: 300s
    scale_down: 600s
```

---

## 10. Performance Anti-Patterns

### 10.1 Anti-Patterns to Avoid

| Anti-Pattern | Problem | Impact | Solution |
|--------------|---------|--------|----------|
| **Excessive Mutex Locking** | Contention on hot paths | 50-80% throughput loss | Lock-free structures, RwLock |
| **Allocation in Hot Path** | Memory allocator overhead | 20-40% latency increase | Buffer pooling, arena allocation |
| **Blocking I/O** | Thread starvation | 90%+ capacity loss | Async I/O with Tokio |
| **Large Stack Frames** | Stack overflow, cache misses | 10-20% slowdown | Box large structs |
| **String Concatenation** | Multiple allocations | 30-50% slower | Pre-allocated buffers |
| **Clone Heavy Structures** | Memory bandwidth saturation | 40-60% throughput loss | Arc, Rc, or references |
| **Unbounded Queues** | Memory exhaustion | OOM crashes | Bounded queues with backpressure |
| **Missing Connection Pooling** | Connection overhead | 3-5x latency | Connection pools |
| **Synchronous Logging** | I/O blocking | 20-30% slowdown | Async logging |
| **No Request Timeout** | Resource leaks | Gradual degradation | Timeouts at all layers |

### 10.2 Code Examples

**❌ ANTI-PATTERN: Mutex in Hot Path**
```rust
use std::sync::Mutex;

// BAD: Mutex contention kills performance
pub struct BadCounter {
    value: Mutex<u64>,
}

impl BadCounter {
    pub fn increment(&self) {
        let mut value = self.value.lock().unwrap();
        *value += 1;
    }  // Lock held across increment
}
```

**✅ CORRECT: Lock-Free Atomic**
```rust
use std::sync::atomic::{AtomicU64, Ordering};

// GOOD: Lock-free, scales linearly
pub struct GoodCounter {
    value: AtomicU64,
}

impl GoodCounter {
    pub fn increment(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
}
```

**❌ ANTI-PATTERN: String Concatenation**
```rust
// BAD: Multiple allocations
pub fn build_response_bad(parts: &[&str]) -> String {
    let mut result = String::new();
    for part in parts {
        result = result + part;  // Allocation on each iteration
    }
    result
}
```

**✅ CORRECT: Pre-Allocated Buffer**
```rust
// GOOD: Single allocation
pub fn build_response_good(parts: &[&str]) -> String {
    let total_len: usize = parts.iter().map(|s| s.len()).sum();
    let mut result = String::with_capacity(total_len);
    for part in parts {
        result.push_str(part);
    }
    result
}
```

**❌ ANTI-PATTERN: Blocking Database Call**
```rust
// BAD: Blocks the entire async runtime thread
pub async fn get_config_bad(db: &Database) -> Config {
    let result = db.query_sync("SELECT * FROM config");  // BLOCKS!
    parse_config(result)
}
```

**✅ CORRECT: Async Database Call**
```rust
// GOOD: Yields to runtime, allows other tasks
pub async fn get_config_good(db: &Database) -> Config {
    let result = db.query("SELECT * FROM config").await;  // Async
    parse_config(result)
}
```

---

## 11. Profiling and Tuning

### 11.1 Profiling Tools

**CPU Profiling with perf:**
```bash
# Record CPU profile
perf record -F 99 -g -p $(pgrep llm-simulator) -- sleep 60

# Generate flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > cpu-flamegraph.svg

# View hotspots
perf report --stdio
```

**Memory Profiling with heaptrack:**
```bash
# Run with heaptrack
heaptrack ./llm-simulator

# Analyze results
heaptrack_gui heaptrack.llm-simulator.*.gz
```

**Async Profiling with tokio-console:**
```rust
// Add to Cargo.toml
[dependencies]
console-subscriber = "0.1"

// In main.rs
#[tokio::main]
async fn main() {
    console_subscriber::init();
    // ... rest of application
}
```

```bash
# Run tokio-console
tokio-console
```

### 11.2 System Tuning Parameters

**Linux Kernel Parameters:**
```bash
# /etc/sysctl.conf

# Network tuning
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# TCP tuning
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 0
net.ipv4.tcp_timestamps = 1
net.ipv4.ip_local_port_range = 10000 65535

# Buffer sizes
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# File descriptors
fs.file-max = 2097152
fs.nr_open = 2097152

# Apply changes
sysctl -p
```

**Process Limits:**
```bash
# /etc/security/limits.conf
*    soft    nofile    1048576
*    hard    nofile    1048576
*    soft    nproc     unlimited
*    hard    nproc     unlimited
```

### 11.3 Performance Monitoring

**Prometheus Metrics:**
```rust
use prometheus::{
    Counter, Histogram, HistogramOpts, Opts, Registry,
    exponential_buckets,
};

pub struct PerformanceMetrics {
    request_duration: Histogram,
    request_total: Counter,
    memory_allocated: Counter,
    cpu_time: Counter,
}

impl PerformanceMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let request_duration = Histogram::with_opts(
            HistogramOpts::new(
                "request_duration_seconds",
                "Request duration in seconds"
            ).buckets(exponential_buckets(0.001, 2.0, 15)?)
        )?;

        let request_total = Counter::with_opts(
            Opts::new("requests_total", "Total requests processed")
        )?;

        let memory_allocated = Counter::with_opts(
            Opts::new("memory_allocated_bytes", "Total memory allocated")
        )?;

        let cpu_time = Counter::with_opts(
            Opts::new("cpu_time_seconds", "Total CPU time consumed")
        )?;

        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(request_total.clone()))?;
        registry.register(Box::new(memory_allocated.clone()))?;
        registry.register(Box::new(cpu_time.clone()))?;

        Ok(Self {
            request_duration,
            request_total,
            memory_allocated,
            cpu_time,
        })
    }

    pub fn observe_request(&self, duration: Duration) {
        self.request_duration.observe(duration.as_secs_f64());
        self.request_total.inc();
    }
}
```

**Custom Performance Dashboard:**
```yaml
# Grafana dashboard JSON excerpt
dashboard:
  title: LLM-Simulator Performance

  panels:
    - title: Request Rate
      type: graph
      targets:
        - expr: rate(requests_total[1m])
          legend: "RPS"
      thresholds:
        - value: 10000
          color: green

    - title: Latency Percentiles
      type: graph
      targets:
        - expr: histogram_quantile(0.50, rate(request_duration_seconds_bucket[5m]))
          legend: "p50"
        - expr: histogram_quantile(0.99, rate(request_duration_seconds_bucket[5m]))
          legend: "p99"
      thresholds:
        - value: 0.005  # 5ms
          color: green
        - value: 0.010  # 10ms
          color: yellow

    - title: CPU Utilization
      type: gauge
      targets:
        - expr: rate(cpu_time_seconds[1m]) * 100
      thresholds:
        - value: 70
          color: green
        - value: 85
          color: yellow
        - value: 95
          color: red
```

### 11.4 Continuous Performance Testing

**CI/CD Performance Gate:**
```yaml
# .github/workflows/performance.yml
name: Performance Regression

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run benchmarks
        run: cargo bench --bench performance -- --save-baseline PR

      - name: Compare with main
        run: |
          cargo bench --bench performance -- --baseline main --baseline PR

      - name: Check for regressions
        run: |
          critcmp main PR | tee regression-report.txt

          # Fail if >10% regression in any benchmark
          if grep -q "+[1-9][0-9]\.[0-9]\\+%" regression-report.txt; then
            echo "Performance regression detected!"
            exit 1
          fi
```

---

## Appendix A: Quick Reference

### Performance Targets Summary

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Throughput | 10,000+ RPS | `wrk -t12 -c400 -d30s` |
| Latency (P50) | <5ms | Prometheus histogram |
| Latency (P99) | <10ms | Prometheus histogram |
| Memory/Request | <10KB | `heaptrack` analysis |
| Cold Start | <1s | Time to first request |
| Concurrent Sessions | 100,000+ | `ulimit -n` check |
| CPU Utilization | 70-85% | `top`, Prometheus |
| Error Rate | <0.1% | Prometheus counter |

### Optimization Checklist

- [ ] Tokio runtime properly configured
- [ ] Lock-free data structures on hot paths
- [ ] Connection pooling enabled
- [ ] Buffer pools initialized
- [ ] Zero-copy I/O where possible
- [ ] Thread-local caching implemented
- [ ] SIMD optimizations applied
- [ ] Memory arena allocation for requests
- [ ] Async I/O throughout
- [ ] Request timeouts configured
- [ ] Rate limiting in place
- [ ] Metrics collection efficient
- [ ] Load balancer optimized
- [ ] Auto-scaling configured
- [ ] Performance monitoring active
- [ ] Benchmark suite running

### Resource Scaling Quick Formulas

```
Nodes = Target_RPS / (12000 × 0.85)
Memory = 2GB + (Concurrent_Requests × 10KB)
Cores = (Target_RPS × 0.002) / 0.70
Bandwidth = (Request_Size + Response_Size) × Target_RPS
```

---

## Appendix B: Troubleshooting Guide

### Common Performance Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High P99, normal P50 | Tail latency spikes | Check GC pauses, CPU steal time |
| Gradual slowdown | Memory leak | Profile with heaptrack |
| CPU at 100%, low RPS | Lock contention | Profile with perf, use lock-free |
| High memory usage | Unbounded queues | Add backpressure, bounded queues |
| Connection timeouts | FD exhaustion | Increase ulimit, check for leaks |
| Erratic performance | CPU throttling | Check `dmesg`, thermal issues |
| Low RPS despite resources | Configuration issue | Review worker threads, check limits |

---

## Document Control

- **Version:** 1.0
- **Last Updated:** 2024-01-26
- **Author:** Principal Systems Architect
- **Review Cycle:** Quarterly
- **Next Review:** 2024-04-26
