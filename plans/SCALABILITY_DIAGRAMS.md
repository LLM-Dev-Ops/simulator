# LLM-Simulator: Scalability & Performance Diagrams

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Data Flow](#2-data-flow)
3. [Scaling Patterns](#3-scaling-patterns)
4. [Performance Optimization](#4-performance-optimization)
5. [Resource Management](#5-resource-management)
6. [Deployment Topologies](#6-deployment-topologies)

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
                        ┌──────────────────────────────────┐
                        │      Internet / Clients          │
                        └────────────┬─────────────────────┘
                                     │
                        ┌────────────▼─────────────────────┐
                        │    CDN / Edge Cache (Optional)   │
                        │    - Static content              │
                        │    - Rate limiting               │
                        └────────────┬─────────────────────┘
                                     │
                        ┌────────────▼─────────────────────┐
                        │      Global Load Balancer        │
                        │      - Geographic routing        │
                        │      - DDoS protection           │
                        └────────────┬─────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
    ┌──────▼──────┐         ┌───────▼────────┐      ┌────────▼──────┐
    │   Region 1  │         │   Region 2     │      │   Region 3    │
    │  (us-east)  │         │  (us-west)     │      │  (eu-west)    │
    └──────┬──────┘         └────────┬───────┘      └────────┬──────┘
           │                         │                        │
    ┌──────▼──────────────────────────────────────────────────▼──────┐
    │                                                                 │
    │                    REGIONAL ARCHITECTURE                        │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │          Layer 7 Load Balancer (HAProxy/Envoy)          │   │
    │  │          - Health checks                                │   │
    │  │          - SSL termination                              │   │
    │  │          - Request routing                              │   │
    │  └──────────────────────┬──────────────────────────────────┘   │
    │                         │                                       │
    │    ┌────────────────────┼────────────────────┐                 │
    │    │                    │                    │                 │
    │  ┌─▼──────┐      ┌──────▼─┐         ┌──────▼─┐                │
    │  │ Node 1 │      │ Node 2 │   ...   │ Node N │                │
    │  │ 12K RPS│      │ 12K RPS│         │ 12K RPS│                │
    │  └────┬───┘      └────┬───┘         └────┬───┘                │
    │       │               │                   │                    │
    │       └───────────────┼───────────────────┘                    │
    │                       │                                        │
    │  ┌────────────────────▼──────────────────────────┐             │
    │  │          Shared Services Layer                │             │
    │  │  ┌──────────┐  ┌─────────┐  ┌────────────┐   │             │
    │  │  │  Redis   │  │Prometheus│  │Config Store│   │             │
    │  │  │ (Metrics)│  │ (Metrics)│  │  (Consul)  │   │             │
    │  │  └──────────┘  └─────────┘  └────────────┘   │             │
    │  └──────────────────────────────────────────────┘             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### 1.2 Single Node Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SIMULATOR NODE                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   TOKIO ASYNC RUNTIME                     │  │
│  │                                                           │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N │     │  │
│  │  │ (CPU 0) │  │ (CPU 1) │  │ (CPU 2) │  │(CPU N-1)│     │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │  │
│  │       │            │            │            │           │  │
│  │       └────────────┴────────────┴────────────┘           │  │
│  │                    Work-Stealing Queue                    │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │                              │
│  ┌───────────────────────────────▼───────────────────────────┐  │
│  │                      HTTP/2 SERVER (Axum)                 │  │
│  │  - Connection pooling                                     │  │
│  │  - Request multiplexing                                   │  │
│  │  - Stream management                                      │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │                              │
│  ┌───────────────────────────────▼───────────────────────────┐  │
│  │                    MIDDLEWARE STACK                       │  │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐  │  │
│  │  │   Auth   │→│   Rate   │→│  Logging  │→│  Metrics   │  │  │
│  │  │          │ │  Limit   │ │           │ │            │  │  │
│  │  └──────────┘ └──────────┘ └───────────┘ └────────────┘  │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │                              │
│  ┌───────────────────────────────▼───────────────────────────┐  │
│  │                    REQUEST ROUTER                         │  │
│  │  - Pre-compiled regex patterns                            │  │
│  │  - Hash-based route lookup                                │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │                              │
│  ┌───────────────────────────────▼───────────────────────────┐  │
│  │                    REQUEST HANDLERS                       │  │
│  │  ┌───────────┐  ┌───────────┐  ┌────────────┐            │  │
│  │  │   Chat    │  │Completion │  │ Embeddings │            │  │
│  │  │Completion │  │           │  │            │            │  │
│  │  └─────┬─────┘  └─────┬─────┘  └──────┬─────┘            │  │
│  │        │              │                │                  │  │
│  └────────┼──────────────┼────────────────┼──────────────────┘  │
│           │              │                │                     │
│  ┌────────▼──────────────▼────────────────▼──────────────────┐  │
│  │                 SIMULATION ENGINE                         │  │
│  │                                                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │  │
│  │  │   Latency    │  │    Error     │  │   Response     │  │  │
│  │  │  Simulator   │  │  Injector    │  │   Generator    │  │  │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │         Lock-Free Data Structures                │    │  │
│  │  │  - AtomicU64 counters                            │    │  │
│  │  │  - SegQueue for request queues                   │    │  │
│  │  │  - DashMap for shared state                      │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    MEMORY MANAGEMENT                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │  │
│  │  │Buffer Pool   │  │Arena Allocator│  │Thread-Local   │  │  │
│  │  │(4KB buffers) │  │(Bump alloc)   │  │Caches (LRU)   │  │  │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 Request Processing Pipeline

```
┌────────────┐
│   Client   │
└──────┬─────┘
       │ 1. HTTP/2 Request
       │
┌──────▼─────────────────────────────────────────────────────┐
│                    INGRESS LAYER                           │
│  Time: 0.1-0.5ms                                           │
│  - TCP connection established (or reused)                  │
│  - TLS handshake (if needed, ~1-5ms first time)            │
│  - HTTP/2 frame parsing (zero-copy)                        │
└──────┬─────────────────────────────────────────────────────┘
       │
┌──────▼─────────────────────────────────────────────────────┐
│                  MIDDLEWARE LAYER                          │
│  Time: 0.2-0.5ms                                           │
│                                                            │
│  ┌────────────────┐     ┌──────────────┐                  │
│  │ 1. Auth Check  │────▶│ Extract key  │                  │
│  │    ~0.05ms     │     │ Validate     │                  │
│  └────────┬───────┘     └──────────────┘                  │
│           │                                                │
│  ┌────────▼───────┐     ┌──────────────┐                  │
│  │ 2. Rate Limit  │────▶│ Check quota  │                  │
│  │    ~0.08ms     │     │ (lock-free)  │                  │
│  └────────┬───────┘     └──────────────┘                  │
│           │                                                │
│  ┌────────▼───────┐     ┌──────────────┐                  │
│  │ 3. Logging     │────▶│ Async write  │                  │
│  │    ~0.02ms     │     │ (buffered)   │                  │
│  └────────┬───────┘     └──────────────┘                  │
│           │                                                │
│  ┌────────▼───────┐     ┌──────────────┐                  │
│  │ 4. Metrics     │────▶│ Atomic inc   │                  │
│  │    ~0.05ms     │     │ (lock-free)  │                  │
│  └────────────────┘     └──────────────┘                  │
└──────┬─────────────────────────────────────────────────────┘
       │
┌──────▼─────────────────────────────────────────────────────┐
│                    ROUTING LAYER                           │
│  Time: 0.05-0.1ms                                          │
│  - Pre-compiled regex match                                │
│  - Handler lookup (hash map, O(1))                         │
│  - Parameter extraction                                    │
└──────┬─────────────────────────────────────────────────────┘
       │
┌──────▼─────────────────────────────────────────────────────┐
│                   HANDLER LAYER                            │
│  Time: 0.3-0.8ms                                           │
│                                                            │
│  ┌─────────────────┐                                       │
│  │ 1. Deserialize  │ Parse JSON (zero-copy where possible) │
│  │    ~0.2ms       │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 2. Validate     │ Check required fields                 │
│  │    ~0.05ms      │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 3. Acquire      │ Get buffer from pool                  │
│  │    Resources    │ Acquire semaphore permit              │
│  │    ~0.05ms      │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 4. Call Engine  │                                       │
│  └────────┬────────┘                                       │
└───────────┼────────────────────────────────────────────────┘
            │
┌───────────▼────────────────────────────────────────────────┐
│                 SIMULATION ENGINE                          │
│  Time: 2-3ms (main latency contributor)                    │
│                                                            │
│  ┌─────────────────┐                                       │
│  │ 1. Profile      │ Get latency profile (cached)          │
│  │    Lookup       │ Thread-local cache hit: 100ns         │
│  │    ~0.1-1ms     │ Shared cache hit: 1μs                 │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 2. Generate     │ Sample from distribution              │
│  │    Timing       │ Calculate token arrivals              │
│  │    ~0.5ms       │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 3. Error Check  │ Should inject error?                  │
│  │    ~0.1ms       │ Probability check (lock-free)         │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 4. Generate     │ Create response tokens                │
│  │    Response     │ Pre-allocated buffer                  │
│  │    ~1-1.5ms     │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 5. Simulate     │ tokio::time::sleep(duration)          │
│  │    Latency      │ Non-blocking (yields to runtime)      │
│  └────────┬────────┘                                       │
└───────────┼────────────────────────────────────────────────┘
            │
┌───────────▼────────────────────────────────────────────────┐
│                 RESPONSE LAYER                             │
│  Time: 0.3-0.8ms                                           │
│                                                            │
│  ┌─────────────────┐                                       │
│  │ 1. Serialize    │ JSON encoding (pre-allocated buffer)  │
│  │    ~0.3ms       │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 2. Add Headers  │ Content-Type, X-Request-ID, etc.      │
│  │    ~0.1ms       │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 3. Compress     │ gzip (if client supports)             │
│  │    ~0.2-0.4ms   │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│  ┌────────▼────────┐                                       │
│  │ 4. Record       │ Update metrics (atomic)               │
│  │    Metrics      │                                       │
│  │    ~0.05ms      │                                       │
│  └────────┬────────┘                                       │
└───────────┼────────────────────────────────────────────────┘
            │
┌───────────▼────────────────────────────────────────────────┐
│                   EGRESS LAYER                             │
│  Time: 0.5-1ms                                             │
│  - HTTP/2 frame encoding                                   │
│  - TCP send (buffered)                                     │
│  - Connection kept alive for reuse                         │
└───────────┬────────────────────────────────────────────────┘
            │
       ┌────▼────┐
       │ Client  │
       └─────────┘

TOTAL OVERHEAD: 3.15 - 5.9ms (Target: <5ms average)
```

### 2.2 Streaming Response Flow

```
Client                   Simulator                   Simulation Engine
  │                          │                               │
  │  POST /v1/chat/         │                               │
  │  completions            │                               │
  │  stream=true            │                               │
  ├─────────────────────────▶│                               │
  │                          │                               │
  │                          │  1. Initialize simulator      │
  │                          ├──────────────────────────────▶│
  │                          │                               │
  │                          │  2. Generate token timings    │
  │                          │◀──────────────────────────────┤
  │                          │  [t1, t2, t3, ..., tN]        │
  │                          │                               │
  │  HTTP 200 + headers      │                               │
  │  Transfer-Encoding:      │                               │
  │  chunked                 │                               │
  │◀─────────────────────────┤                               │
  │                          │                               │
  │                          │                               │
  │  ┌─ SSE Stream ──────────────────────────────────────┐   │
  │  │                      │                            │   │
  │  │  Wait(TTFT)          │                            │   │
  │  │  ├───────────────────┤                            │   │
  │  │                      │                            │   │
  │  │  data: {"delta":     │                            │   │
  │  │  "Hello"}            │                            │   │
  │  │◀─────────────────────┤                            │   │
  │  │                      │                            │   │
  │  │  Wait(ITL_1)         │                            │   │
  │  │  ├───────────────────┤                            │   │
  │  │                      │                            │   │
  │  │  data: {"delta":     │                            │   │
  │  │  " world"}           │                            │   │
  │  │◀─────────────────────┤                            │   │
  │  │                      │                            │   │
  │  │  Wait(ITL_2)         │                            │   │
  │  │  ├───────────────────┤                            │   │
  │  │                      │                            │   │
  │  │  data: {"delta":     │                            │   │
  │  │  "!"}                │                            │   │
  │  │◀─────────────────────┤                            │   │
  │  │                      │                            │   │
  │  │  ...                 │                            │   │
  │  │                      │                            │   │
  │  │  data: [DONE]        │                            │   │
  │  │◀─────────────────────┤                            │   │
  │  │                      │                            │   │
  │  └──────────────────────────────────────────────────┘   │
  │                          │                               │
  │  Connection closed       │                               │
  │◀─────────────────────────┤                               │
  │                          │                               │


Timing Breakdown:
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  TTFT (Time to First Token): 250-800ms                   │
│  ├─────────────────────────────────────────┤             │
│                                                           │
│  ITL_1 (Inter-Token Latency): 10-20ms                    │
│  ├──┤                                                     │
│                                                           │
│  ITL_2: 10-20ms                                           │
│  ├──┤                                                     │
│                                                           │
│  ...                                                      │
│                                                           │
│  Total Duration = TTFT + (N-1) × mean_ITL                │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Scaling Patterns

### 3.1 Horizontal Scaling (Linear Growth)

```
Throughput vs. Node Count

 RPS
 │
200K├                                      ●
    │                                    ╱
180K├                                  ●
    │                                ╱
160K├                              ●
    │                            ╱
140K├                          ●
    │                        ╱
120K├                      ●
    │                    ╱
100K├                  ●
    │                ╱
 80K├              ●
    │            ╱
 60K├          ●
    │        ╱
 40K├      ●
    │    ╱
 20K├  ●
    │╱
   0├─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─► Nodes
    0 2 4 6 8 10 12 14 16 18 20

Formula: Total_RPS = N × 10,200 (assumes 12K RPS × 85% efficiency)

Scaling Efficiency:
- 2 nodes:   20,400 RPS (100% efficiency)
- 10 nodes: 102,000 RPS (100% efficiency)
- 20 nodes: 204,000 RPS (100% efficiency)

Linear scaling maintained through:
- Stateless architecture
- Lock-free data structures
- No shared mutable state
- Independent request processing
```

### 3.2 Vertical Scaling (CPU Cores)

```
Throughput vs. CPU Cores (Single Node)

 RPS
 │
24K├                          ┌───────  Saturation
   │                        ╱
20K├                      ●
   │                    ╱
16K├                  ●
   │                ╱
12K├              ●
   │            ╱
 8K├          ●
   │        ╱
 4K├      ●
   │    ╱
 2K├  ●
   │╱
  0├─┬─┬──┬──┬──┬──┬──┬──┬──┬──► CPU Cores
   0 2 4  8  12 16 20 24 28 32

Optimal Configuration:
- 8 cores:  ~12,000 RPS (sweet spot for c6i.2xlarge)
- 16 cores: ~20,000 RPS (c6i.4xlarge)
- 32 cores: ~24,000 RPS (diminishing returns due to coordination)

Bottleneck Analysis:
- 1-8 cores:   CPU-bound (linear scaling)
- 8-16 cores:  Near-linear (90% efficiency)
- 16+ cores:   Synchronization overhead becomes significant
```

### 3.3 Auto-Scaling Behavior

```
Traffic Pattern Over Time

Load
│
│    ┌─────────┐
│    │  Peak   │
│    │ Traffic │
│ ┌──┘         └───┐
│ │   Baseline     │
│─┴────────────────┴─────────────────────────────► Time
│
│ Node Count
│    ┌────┬────┐
│    │ 15 │ 15 │
│ ┌──┘    └────┴───┐
│ │  10   Nodes 10 │
│─┴────────────────┴─────────────────────────────► Time
│
│ Response Time (P99)
│
│       ┌─┐
│     ┌─┘ └─┐
│ ────┘     └────────────────────────────────────► Time
│    Good   Good

Auto-Scaling Timeline:

00:00 - Baseline: 10 nodes, 50K RPS, CPU 50%
05:00 - Traffic increases
05:03 - CPU hits 75% threshold → Trigger scale-up
05:04 - Provisioning new nodes (1-2 min)
05:06 - New nodes added → 15 nodes
05:07 - CPU drops to 50%, P99 stable
10:00 - Peak traffic: 75K RPS
12:00 - Traffic decreases
12:10 - CPU drops to 40% for 10+ minutes
12:11 - Trigger scale-down
12:13 - Remove 5 nodes → 10 nodes
12:14 - Back to baseline

Scaling Policies:
- Scale-up:  CPU >75% for 5 min OR P99 >15ms for 2 min
- Scale-down: CPU <40% for 10 min AND P99 <8ms
- Cooldown: 5 min between scale operations
- Min nodes: 3 (HA)
- Max nodes: 50 (cost limit)
```

---

## 4. Performance Optimization

### 4.1 Latency Breakdown

```
Request Latency Components (Target: <5ms)

┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Network (0.5-1ms)     ████                                │
│                                                            │
│  HTTP Parsing (0.1ms)  █                                   │
│                                                            │
│  Routing (0.05ms)      █                                   │
│                                                            │
│  Auth (0.05ms)         █                                   │
│                                                            │
│  Rate Limit (0.08ms)   █                                   │
│                                                            │
│  Handler (0.3ms)       ███                                 │
│                                                            │
│  Simulation (2.5ms)    ██████████████████████              │
│                                                            │
│  Serialization (0.3ms) ███                                 │
│                                                            │
│  Response (0.5ms)      ████                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
  0ms                  1ms                 2ms          3ms  4ms

Total: 3.15ms average, 5.9ms worst-case

Optimization Opportunities:
1. Simulation (2.5ms) - 50% of total
   → Cache latency profiles
   → Pre-generate random numbers
   → Use faster RNG

2. Network (1ms) - 20% of total
   → HTTP/2 with connection reuse
   → TCP_NODELAY enabled
   → Optimize buffer sizes

3. Serialization (0.3ms) - 6% of total
   → Pre-allocated buffers
   → Zero-copy where possible
   → SIMD for JSON encoding
```

### 4.2 Memory Layout Optimization

```
Cache-Line Aware Data Structures

┌─────────────────────────────────────────────────────────────┐
│                      CPU Cache                              │
│                                                             │
│  L1 Cache (32KB per core)    Access: ~1ns                   │
│  ├─ Hot data (frequently accessed)                          │
│  ├─ Request context                                         │
│  └─ Counters (atomic)                                       │
│                                                             │
│  L2 Cache (256KB per core)   Access: ~4ns                   │
│  ├─ Latency profiles (cached)                               │
│  ├─ Configuration                                           │
│  └─ Lookup tables                                           │
│                                                             │
│  L3 Cache (Shared, 16-32MB)  Access: ~10-20ns               │
│  ├─ Shared state (DashMap)                                  │
│  ├─ Buffer pools                                            │
│  └─ Metrics aggregation                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Main Memory (RAM)                        │
│                    Access: ~100ns                           │
│  ├─ Large allocations                                       │
│  ├─ Response buffers                                        │
│  └─ Metrics history                                         │
└─────────────────────────────────────────────────────────────┘

Struct Layout Example:

// BAD: Not cache-aligned
struct RequestContext {
    id: u64,              // 8 bytes
    small_field: u8,      // 1 byte
    metadata: [u8; 100],  // 100 bytes
    timestamp: Instant,   // 16 bytes
}
// Total: 125 bytes (spans 2 cache lines, causes false sharing)

// GOOD: Cache-line aligned
#[repr(align(64))]
struct RequestContext {
    // Hot fields (cache line 1)
    id: u64,
    timestamp: Instant,
    small_field: u8,
    _pad1: [u8; 39],      // Padding to 64 bytes

    // Cold fields (cache line 2)
    metadata: [u8; 100],
    _pad2: [u8; 28],      // Padding to 128 bytes
}
```

### 4.3 Lock-Free vs Mutex Performance

```
Throughput Comparison (Operations/Second)

AtomicU64 (Lock-Free)    ████████████████████████████  1,000,000,000 ops/s
RwLock (Read-Heavy)      ████████████                    500,000,000 ops/s
Mutex                    ████                            200,000,000 ops/s

                         0        250M      500M      750M      1B

Latency Comparison (Nanoseconds per Operation)

AtomicU64                █  1ns
RwLock (Read)            ██ 2ns
RwLock (Write)           ████ 4ns
Mutex                    ████████ 8ns

                         0    2    4    6    8   10

Contention Impact (4 threads, high contention)

AtomicU64                ████████████  Still ~500M ops/s
RwLock                   ████           100M ops/s
Mutex                    █              25M ops/s

Recommendation:
- Hot path counters: AtomicU64
- Read-heavy shared state: RwLock
- Complex shared state: DashMap (lock-free hash map)
- Avoid Mutex on critical paths
```

---

## 5. Resource Management

### 5.1 Memory Management Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Binary + Static Data: 50-100MB                │
│  - Compiled code                                            │
│  - Static configuration                                     │
│  - Latency profile data                                     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Runtime Allocations: 1-2GB                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Thread-Local Storage (TLS): ~10MB per thread         │  │
│  │  - Request context                                    │  │
│  │  - L1 cache (LRU)                                     │  │
│  │  - Thread-local RNG                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Buffer Pools: 500MB                                  │  │
│  │  - Pre-allocated 4KB buffers                          │  │
│  │  - Pool size: 100,000 buffers                         │  │
│  │  - Lock-free MPMC queue                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Shared State: 200-500MB                              │  │
│  │  - DashMap (config, profiles)                         │  │
│  │  - Metrics aggregation                                │  │
│  │  - Request tracking                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Request Processing: 200-500MB                        │  │
│  │  - Active requests (200 concurrent × 10KB = 2MB)      │  │
│  │  - Response buffers                                   │  │
│  │  - Temporary allocations                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Tokio Runtime: 100-200MB                             │  │
│  │  - Task queues                                        │  │
│  │  - I/O buffers                                        │  │
│  │  - Timer wheel                                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Total Memory: 2-4GB                          │
│  Recommended instance: 16GB (4x safety margin)              │
└─────────────────────────────────────────────────────────────┘

Memory Growth Over Time (Healthy vs Leak):

Memory
│
│ Leak (Bad)
│       ╱
│     ╱
│   ╱
│ ╱
│────────────────  Healthy (stable after warmup)
│
└──────────────────────────────────────────────────► Time
  0h    1h     6h    12h    24h
```

### 5.2 Connection Pooling

```
┌─────────────────────────────────────────────────────────────┐
│              HTTP/2 CONNECTION POOLING                      │
└─────────────────────────────────────────────────────────────┘

Client Connections (Inbound):

┌──────────┐  Connection 1  ┌─────────────────────┐
│ Client 1 │◀──────────────▶│  100 multiplexed    │
└──────────┘                │  streams (HTTP/2)   │
                            │                     │
┌──────────┐  Connection 2  │                     │
│ Client 2 │◀──────────────▶│  Pool: 10,000       │
└──────────┘                │  connections        │
                            │                     │
┌──────────┐  Connection N  │                     │
│ Client N │◀──────────────▶│  Max streams/conn:  │
└──────────┘                │  100                │
                            │                     │
                            │  Idle timeout: 90s  │
                            │                     │
                            │  Keepalive: 60s     │
                            └─────────────────────┘

Effective Capacity:
10,000 connections × 100 streams = 1,000,000 concurrent requests

Memory per connection: ~4KB
Total connection memory: 10,000 × 4KB = 40MB


Connection Lifecycle:

┌────────────┐
│  New       │
│  Request   │
└─────┬──────┘
      │
      │  Connection exists?
      │
      ├─ YES ─▶ ┌──────────────┐
      │         │  Reuse       │
      │         │  Connection  │
      │         └──────┬───────┘
      │                │
      ├─ NO ──▶ ┌─────▼────────┐
      │         │  Create New  │
      │         │  Connection  │
      │         └──────┬───────┘
      │                │
      │         ┌──────▼───────┐
      └────────▶│  Process     │
                │  Request     │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Keep-Alive? │
                └──────┬───────┘
                       │
              YES ─────┼───── NO
                       │        │
                ┌──────▼──┐  ┌──▼─────┐
                │  Pool   │  │ Close  │
                │  (Reuse)│  └────────┘
                └─────────┘
```

### 5.3 Concurrency Control

```
┌─────────────────────────────────────────────────────────────┐
│              SEMAPHORE-BASED RATE LIMITING                  │
└─────────────────────────────────────────────────────────────┘

Semaphore State:

Available Permits: [●●●●●●●○○○○○○○○○○○○○]  6/20 available

Incoming Requests:

Request 1 ──▶ Try Acquire ──▶ SUCCESS ──▶ Process
Request 2 ──▶ Try Acquire ──▶ SUCCESS ──▶ Process
Request 3 ──▶ Try Acquire ──▶ SUCCESS ──▶ Process
Request 4 ──▶ Try Acquire ──▶ SUCCESS ──▶ Process
Request 5 ──▶ Try Acquire ──▶ SUCCESS ──▶ Process
Request 6 ──▶ Try Acquire ──▶ SUCCESS ──▶ Process
Request 7 ──▶ Try Acquire ──▶ WAIT     (queued)
Request 8 ──▶ Try Acquire ──▶ WAIT     (queued)
Request 9 ──▶ Try Acquire ──▶ TIMEOUT  ──▶ 503 Error

Available Permits: [○○○○○○○○○○○○○○○○○○○○]  0/20 available

After Request Completion:

Request 1 completes ──▶ Release permit ──▶ Request 7 proceeds

Available Permits: [●○○○○○○○○○○○○○○○○○○○]  1/20 available


Concurrency Metrics:

Utilization = (Max_Permits - Available) / Max_Permits

100% ├                     ●●●●●
     │                   ●●    ●●
     │                  ●        ●
 75% ├               ●●            ●●
     │             ●●                ●●
 50% ├          ●●●                    ●●●
     │        ●●                          ●●
 25% ├     ●●●                              ●●●
     │   ●●                                    ●●
  0% └──●──────────────────────────────────────●─► Time
        Idle    Busy Period                  Idle

Target Utilization: 70-85%
Alert if >95% for >5 minutes
```

---

## 6. Deployment Topologies

### 6.1 Single-Region Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS US-EAST-1                            │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              VPC (10.0.0.0/16)                        │  │
│  │                                                       │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │     Public Subnet (10.0.1.0/24)              │    │  │
│  │  │                                              │    │  │
│  │  │  ┌──────────────────────────────────────┐    │    │  │
│  │  │  │  Application Load Balancer           │    │    │  │
│  │  │  │  - Internet-facing                   │    │    │  │
│  │  │  │  - SSL termination                   │    │    │  │
│  │  │  │  - Health checks                     │    │    │  │
│  │  │  └────────────┬─────────────────────────┘    │    │  │
│  │  └───────────────┼──────────────────────────────┘    │  │
│  │                  │                                    │  │
│  │  ┌───────────────▼──────────────────────────────┐    │  │
│  │  │     Private Subnet A (10.0.10.0/24)          │    │  │
│  │  │                                              │    │  │
│  │  │  ┌─────────────┐  ┌─────────────┐           │    │  │
│  │  │  │  Simulator  │  │  Simulator  │           │    │  │
│  │  │  │  Instance 1 │  │  Instance 2 │  ...      │    │  │
│  │  │  │  c6i.2xlarge│  │  c6i.2xlarge│           │    │  │
│  │  │  └─────────────┘  └─────────────┘           │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  │                                                       │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │     Private Subnet B (10.0.11.0/24)          │    │  │
│  │  │                                              │    │  │
│  │  │  ┌─────────────┐  ┌─────────────┐           │    │  │
│  │  │  │  Simulator  │  │  Simulator  │           │    │  │
│  │  │  │  Instance 3 │  │  Instance 4 │  ...      │    │  │
│  │  │  │  c6i.2xlarge│  │  c6i.2xlarge│           │    │  │
│  │  │  └─────────────┘  └─────────────┘           │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  │                                                       │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │     Private Subnet C (10.0.12.0/24)          │    │  │
│  │  │                                              │    │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │    │  │
│  │  │  │  Redis   │  │Prometheus│  │  Consul   │  │    │  │
│  │  │  │(Metrics) │  │ (Monitor)│  │  (Config) │  │    │  │
│  │  │  └──────────┘  └──────────┘  └───────────┘  │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Specifications:
- AZs: 2 (Availability Zones A & B)
- Nodes: 10 total (5 per AZ)
- Capacity: ~102,000 RPS
- Availability: 99.9% (SLA)
- Monthly Cost: ~$2,500
```

### 6.2 Multi-Region Deployment

```
                    ┌──────────────────────┐
                    │   Route 53 (DNS)     │
                    │   - Latency routing  │
                    │   - Health checks    │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼───────┐    ┌─────────▼────────┐   ┌────────▼────────┐
│  US-EAST-1    │    │   US-WEST-2      │   │   EU-WEST-1     │
│               │    │                  │   │                 │
│  ┌─────────┐  │    │  ┌─────────┐     │   │  ┌─────────┐    │
│  │   ALB   │  │    │  │   ALB   │     │   │  │   ALB   │    │
│  └────┬────┘  │    │  └────┬────┘     │   │  └────┬────┘    │
│       │       │    │       │          │   │       │         │
│  ┌────▼────┐  │    │  ┌────▼────┐     │   │  ┌────▼────┐    │
│  │10 Nodes │  │    │  │10 Nodes │     │   │  │10 Nodes │    │
│  │102K RPS │  │    │  │102K RPS │     │   │  │102K RPS │    │
│  └─────────┘  │    │  └─────────┘     │   │  └─────────┘    │
│               │    │                  │   │                 │
└───────────────┘    └──────────────────┘   └─────────────────┘

Total Capacity: 306,000 RPS
Geographic Distribution:
- Americas: 40% traffic → US-EAST-1, US-WEST-2
- Europe: 35% traffic → EU-WEST-1
- Asia: 25% traffic → Closest region (routed via latency)

Failover:
- Regional failure → Route 53 automatically routes to healthy region
- RTO (Recovery Time Objective): 2-5 minutes
- RPO (Recovery Point Objective): 0 (stateless)
```

---

**Document Version:** 1.0
**Last Updated:** 2024-01-26
