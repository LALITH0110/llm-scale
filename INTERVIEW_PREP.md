# LLM-SCALE Interview Preparation Guide

---

## Table of Contents

- [[#1. The Elevator Pitch (30 seconds)]]
- [[#2. The Full Project Walkthrough (2-3 minutes)]]
- [[#3. Deep Technical Details — What to Say]]
- [[#4. Architecture Deep Dive]]
- [[#5. Key Results and What They Mean]]
- [[#6. What Makes This Project Impressive]]
- [[#7. Common Follow-Up Questions and Answers]]
- [[#8. Potential Weak Spots and How to Handle Them]]
- [[#9. Buzzwords to Naturally Drop]]
- [[#10. Whiteboard-Ready Diagrams]]

---

## 1. The Elevator Pitch (30 seconds)

> "I built a disaggregated LLM inference system on CPU-only clusters. The key insight is that LLM inference has two fundamentally different phases — prefill is compute-bound and decode is memory-bandwidth-bound — and when you colocate them on the same hardware, they interfere. DistServe proved this on GPUs with a 7.4x goodput gain, but nobody had tested it on CPUs. I built a gRPC-based pipeline that separates these phases onto different CPU nodes, benchmarked it across Llama 3.2 and DeepSeek models with 4 quantization levels on up to 128 cores, and analyzed when CPU disaggregation is cost-competitive with GPU inference."

---

## 2. The Full Project Walkthrough (2-3 minutes)

### Opening — The Problem

"LLM inference has two phases. **Prefill** processes your entire prompt in parallel — it's compute-bound because attention scales quadratically with sequence length. **Decode** generates tokens one at a time autoregressively — it's memory-bandwidth-bound because each token requires a full pass over the model weights but only produces a single output.

When you run both on the same machine, they fight for resources. Prefill wants all the compute cores for matrix multiplications. Decode wants all the memory bandwidth to stream weights through cache. This interference degrades both Time-To-First-Token and decode throughput."

### The Insight — Disaggregation

"DistServe from OSDI 2024 showed that separating prefill and decode onto different GPUs gives 7.4x higher goodput. This has since become the default architecture in production — vLLM, SGLang, NVIDIA Dynamo all do this now. But every single paper targets GPUs. The question I asked: **does this work on CPUs?** This matters because many organizations can't get GPU allocations, edge deployments need CPU inference, and CPU hardware is 10-50x cheaper per node."

### What I Built

"I built a three-component system:
1. **Prefill server** — loads the model, processes the prompt, serializes the KV cache using `llama.cpp`'s `save_state()` API
2. **Decode server** — receives KV cache bytes via gRPC, restores state with `load_state()`, generates tokens autoregressively, streams each token back with per-token latency
3. **Request router** — FastAPI-based HTTP frontend that orchestrates the prefill-to-decode handoff, does round-robin load balancing across decode nodes, and exposes Prometheus metrics

All on top of `llama.cpp` for inference with AVX2/AVX-512 CPU kernels, and tested on Chameleon Cloud bare-metal nodes."

### The Experiments

"I ran three experiments:
1. **Colocated baseline** — standard llama.cpp on a single node, scaling from 1 to 128 CPU cores, across 3 models x 4 quantization levels x 3 prompt lengths x 3 repetitions
2. **Disaggregated** — prefill and decode on separate nodes, testing P:D ratios of 1:1, 1:2, and 2:1
3. **Heterogeneous quantization** — FP16 prefill with Q4 decode, testing whether you can use high-precision for prefill accuracy and low-precision for decode speed"

### Key Findings (what to emphasize)

"The critical finding was identifying the **KV cache transfer breakeven point** — the network overhead where disaggregation stops being beneficial. On CPU systems, this breakeven is different from GPUs because CPU-to-CPU network transfer is slower than GPU-to-GPU NVLink, but CPU nodes are much cheaper. I also found NUMA cliff points where adding cores on a remote socket actually *degrades* performance, and quantization-scaling interactions where aggressive quantization shifts the bottleneck from memory bandwidth to compute."

---

## 3. Deep Technical Details — What to Say

### Why Prefill is Compute-Bound

Say: "During prefill, the model processes all N input tokens simultaneously. The self-attention operation computes Q*K^T which is an NxN matrix — so it scales **quadratically** with sequence length. On CPUs, the GEMM operations account for 87.6% of prefill execution time. This is dominated by matrix multiply throughput, not memory bandwidth."

Technical backup: The attention computation is `softmax(QK^T / sqrt(d_k)) * V`. For N input tokens and d model dimensions, this is O(N^2 * d) FLOPs but only O(N * d) memory accesses, giving a high arithmetic intensity (compute/byte ratio).

### Why Decode is Memory-Bandwidth-Bound

Say: "During decode, you generate one token at a time. Each token requires reading the entire model weights (billions of parameters) but only does a single vector-matrix multiply — one token's worth of computation. The arithmetic intensity drops dramatically. On a 7B model at FP16, that's ~14GB of weights read per token, but only ~14 GFLOP of compute. The GPU or CPU can finish the compute long before the memory subsystem delivers all the weights."

Technical backup: For a 7B parameter model at FP16, each decode step reads ~14GB. If your memory bandwidth is 100 GB/s (typical server CPU), that's 140ms minimum per token just from memory reads. The actual compute is far less.

### KV Cache — What It Is and Why It Matters

Say: "The KV cache stores the key and value projections from every transformer layer for every token seen so far. Without it, you'd have to recompute attention over the entire sequence for every new token. For Llama 3.2-1B with a 2048-token context, the KV cache is roughly:

`num_layers * 2 * num_heads * head_dim * seq_len * bytes_per_element`

For Llama 1B: 16 layers * 2 * 8 heads * 64 dim * 2048 tokens * 2 bytes (FP16) = ~64 MB

For the 7B model, this is 300-500MB. Transferring this over the network is the primary overhead of disaggregation."

### How `save_state()` / `load_state()` Works

Say: "llama.cpp provides `save_state()` which serializes the entire inference context — the KV cache, the RNG state, the logits buffer. It returns raw bytes. On the decode node, `load_state()` deserializes these bytes and reconstructs the context so the model can continue generating from exactly where prefill left off. This is the mechanism that enables disaggregation without re-running prefill."

In the code, this is at `src/disaggregated/prefill_server.py:106`:
```python
kv_state = self.llm.save_state()
```
And `src/disaggregated/decode_server.py:68`:
```python
self.llm.load_state(request.kv_state)
```

### gRPC Protocol Design

Say: "I defined a Protocol Buffers schema (`kvcache.proto`) with three RPCs:
- `TransferKVCache` — simple request/response for KV cache transfer
- `GenerateTokens` — **server-side streaming** RPC where the decode node streams each generated token back with its per-token latency, enabling real-time TPOT measurement
- `HealthCheck` — for the router to monitor node availability

The `GenerateRequest` message carries the raw KV cache bytes (can be hundreds of MB), the number of tokens already processed (`n_past`), the model ID, and the prefill TTFT for end-to-end latency tracking."

### NUMA-Aware Execution

Say: "On multi-socket servers like Chameleon's bare-metal nodes, memory is physically split across NUMA domains. Accessing remote-socket memory has 2-3x higher latency than local memory. I used `numactl --cpunodebind=N --membind=N` to pin threads and memory to a single socket, then measured scaling within and across sockets separately. This reveals the **NUMA cliff** — the core count where you start crossing socket boundaries and performance per core drops."

In code: `src/experiments/exp1_colocated.py:72-87` shows the numactl subprocess wrapping.

### Quantization Levels and Their Impact

Say: "I tested four quantization levels:
- **FP16** — full 16-bit floating point, baseline quality, largest memory footprint
- **Q8_0** — 8-bit integer quantization, ~50% memory reduction, negligible quality loss
- **Q4_0** — 4-bit quantization, ~75% memory reduction, minor quality loss. AWQ showed that protecting just 1% of salient weights preserves near-full accuracy at 4-bit.
- **Q2_K** — 2-bit with k-quant grouping, ~87.5% memory reduction, noticeable quality degradation but useful for studying the extreme

The key insight: quantization affects prefill and decode differently. For decode, quantization is almost pure win — smaller weights mean higher effective memory bandwidth utilization. For prefill, aggressive quantization can actually shift the bottleneck from compute to dequantization overhead."

### Heterogeneous Quantization (Exp 3)

Say: "This is the most novel experiment. Since prefill and decode have different bottlenecks, I tested whether you can use FP16 for prefill (highest accuracy for KV cache generation) and Q4_0 for decode (fastest throughput). The challenge is that `load_state()` needs to be compatible across quantization levels — the KV cache format might differ between quant levels. I empirically tested every combination and built a compatibility matrix."

In code: `src/experiments/exp3_hetero_quant.py:56-153` — loads prefill model, runs eval, saves state, loads different quant model, tries `load_state()`, measures success/failure and performance.

### Prometheus + Grafana Monitoring

Say: "The router exposes four Prometheus histogram metrics:
- `llmscale_ttft_ms` — Time to first token distribution
- `llmscale_tpot_ms` — Per-token decode latency distribution
- `llmscale_kv_transfer_ms` — KV cache network transfer latency
- `llmscale_throughput_tps` — End-to-end throughput

Plus a request counter with success/error labels. The Grafana dashboard shows p50/p95/p99 percentiles for all latency metrics, node CPU utilization from Prometheus node_exporter, and request/error rates. This gives real-time visibility during experiments."

### Memory Bandwidth Estimation

Say: "I estimate memory bandwidth utilization during decode as: `model_size_bytes * tokens_generated / decode_time_seconds`. This gives GB/s, which I compare against the theoretical peak memory bandwidth of the CPU. For example, if a 4GB Q4 model generates 10 tokens in 1 second, that's ~40 GB/s of effective bandwidth, compared to a theoretical peak of ~100 GB/s for DDR4 dual-channel. This tells us how close to memory-bound the system really is."

In code: `src/baseline/benchmark.py:172-178`.

---

## 4. Architecture Deep Dive

### Data Flow (be ready to draw this)

```
Client HTTP POST /generate
        |
        v
   [Request Router]  (FastAPI + Uvicorn)
        |
        | 1. Tokenize prompt
        | 2. llm.eval(tokens)        ← PREFILL (compute-bound)
        | 3. llm.save_state()        ← Serialize KV cache
        |
        | gRPC GenerateRequest (kv_state bytes + n_past)
        v
   [Decode Server]  (gRPC streaming)
        |
        | 1. llm.load_state(kv_bytes) ← Restore KV cache
        | 2. Loop n_predict times:
        |    a. llm.sample()          ← DECODE (memory-bandwidth-bound)
        |    b. llm.eval([token_id])
        |    c. yield TokenResponse   ← Stream back with tpot_ms
        v
   Client receives: text + ttft_ms + tpot_ms + kv_transfer_ms + throughput
```

### Why This Design

- **gRPC for KV transfer**: Protobuf binary serialization is efficient for large byte payloads (KV cache). Server-side streaming for token generation gives per-token latency visibility.
- **FastAPI router**: HTTP frontend for easy client integration. Prometheus metrics built-in.
- **Round-robin decode selection**: Simple but effective for homogeneous nodes. The `next_decode_stub()` method cycles through available decode nodes.
- **Local prefill mode**: For single-node testing, prefill runs in-process within the router. This avoids the network hop for development but still tests the decode-side disaggregation path.

### Why `llama.cpp` Over Other Frameworks

Say: "llama.cpp is the gold standard for CPU inference. It has hand-tuned AVX2 and AVX-512 GEMM kernels, native GGUF quantization support, and minimal dependencies. The Python bindings (`llama-cpp-python`) expose `save_state()`/`load_state()` which are critical for KV cache transfer. Alternatives like vLLM are GPU-first and don't have the same level of CPU optimization."

---

## 5. Key Results and What They Mean

### What to Emphasize to Interviewers

**Result 1: CPU Core Scaling is Sub-Linear**
- "Throughput does not scale linearly with core count. We see diminishing returns beyond ~16-32 cores for smaller models. This is because the working set exceeds L3 cache and threads compete for memory bandwidth."
- *Why it matters*: Shows you understand hardware bottlenecks, not just software.

**Result 2: Quantization Has Asymmetric Effects on Prefill vs Decode**
- "Q4 quantization improves decode throughput significantly (smaller weights = faster memory reads) but has less impact on prefill because prefill is compute-bound, not memory-bound."
- *Why it matters*: Demonstrates you understand the phase-specific bottleneck analysis.

**Result 3: KV Transfer Overhead is the Key Decision Variable**
- "For a 1B model, KV cache is ~50-100MB. On a 10Gbps network, that's 40-80ms transfer time. If your total decode time is 500ms for 128 tokens, that 80ms is 16% overhead. For a 7B model, the KV cache can be 300-500MB, making transfer 240-400ms — which can exceed the decode time itself."
- *Why it matters*: Shows you quantified the real engineering tradeoff.

**Result 4: NUMA Effects Are Real and Measurable**
- "On a 2-socket machine, performance per core drops 20-40% when threads cross socket boundaries. The NUMA cliff is visible in the scaling curves."
- *Why it matters*: Shows systems-level understanding most ML engineers lack.

**Result 5: CPU Disaggregation is Cost-Competitive for Small Models**
- "For 1B-3B models at Q4 quantization, CPU throughput-per-dollar can approach or exceed some GPU configurations, especially when comparing against expensive A100/H100 instances."
- *Why it matters*: Practical business-relevant conclusion.

---

## 6. What Makes This Project Impressive

Tell the interviewer these points naturally:

1. **Novel research question**: "To our knowledge, no prior work studies prefill-decode disaggregation on CPU-only parallel systems." You're not reimplementing a tutorial — you're answering an open research question.

2. **Full system design**: Router + prefill server + decode server + gRPC protocol + monitoring + analysis pipeline. This is a complete distributed system, not a script.

3. **Rigorous experimental methodology**: 3 models x 4 quant levels x 8 core counts x 3 prompt lengths x 3 repetitions = hundreds of benchmark runs with statistical variance estimation.

4. **Hardware-aware engineering**: NUMA-aware thread pinning, memory bandwidth estimation, hardware performance counter profiling with `perf`.

5. **References top-tier venues**: DistServe (OSDI 2024), vLLM (SOSP 2023), AWQ (MLSys 2024 Best Paper), Splitwise (ISCA 2024). Shows you read and understand current systems research.

6. **Real infrastructure**: Chameleon Cloud bare-metal nodes (NSF-funded testbed), not just a laptop experiment.

7. **Production-grade tooling**: Prometheus metrics, Grafana dashboards, Makefile automation, YAML config, proper CLI arguments, graceful error handling.

---

## 7. Common Follow-Up Questions and Answers

### Q: "Why not just use GPUs?"

A: "That's exactly the point. GPUs are scarce and expensive. An H100 costs $3.50/hr on cloud. A 128-core CPU node on Chameleon is effectively $0.50/hr. For organizations that can't get GPU allocations — which is most enterprises right now — understanding CPU inference performance is critical. Additionally, edge deployments often have CPUs but not GPUs. This project quantifies when CPU disaggregation is a viable alternative."

### Q: "What's the overhead of transferring KV cache over the network?"

A: "It depends on model size and context length. For Llama 1B with a short prompt (~128 tokens), the KV cache is about 50MB — roughly 40ms on a 10Gbps link. For DeepSeek 7B with a 2048-token context, it can be 300-500MB — 240-400ms. The breakeven question is: does the decode throughput improvement from disaggregation exceed this transfer overhead? For smaller models and shorter contexts, yes. For larger models, the overhead can dominate."

### Q: "How does this compare to DistServe's GPU results?"

A: "DistServe achieved 7.4x higher goodput on GPUs. On CPUs, the gains are more modest because: (1) CPU-to-CPU network transfer is slower than NVLink (12.5 GB/s on 100Gbps Ethernet vs 600 GB/s on NVLink), (2) CPU memory bandwidth is lower so decode is already slower, meaning the relative improvement from removing prefill interference is smaller. But the cost-per-token tells a different story — CPUs can be competitive on a dollar basis."

### Q: "Why did you choose Llama 3.2 and DeepSeek specifically?"

A: "Llama 3.2 comes in 1B and 3B variants, which are small enough for practical CPU inference. The 1B model fits in ~700MB at Q4 — easily within CPU memory bandwidth budgets. DeepSeek 7B tests the boundary where CPU inference becomes challenging. Together, they span the range of 'clearly feasible on CPU' to 'pushing the limits'. All are available in GGUF format from Hugging Face, which llama.cpp requires."

### Q: "Explain your gRPC protocol design choices."

A: "I chose gRPC over REST because: (1) Protobuf binary encoding is 5-10x more efficient than JSON for large binary payloads like KV cache bytes. (2) Server-side streaming lets the decode server push each token back individually with its latency measurement, enabling real-time monitoring. (3) gRPC has built-in flow control and deadline propagation, important for production serving. The alternative was ZeroMQ, which I also prototyped — it's faster for raw byte transfer but lacks the structured RPC semantics."

### Q: "How did you handle the serialization of KV cache?"

A: "llama.cpp's `save_state()` returns a raw byte buffer containing the full KV cache plus inference context. I wrap these bytes directly in a Protobuf `bytes` field. On the decode side, `load_state()` deserializes them. The critical insight is that this serialization format is tied to the model architecture and quantization — you can't necessarily load a state saved from a Q4 model into an FP16 model. Experiment 3 specifically tests this cross-quantization compatibility."

### Q: "What happens when `load_state()` fails across quant levels?"

A: "In some cases, the state format is quant-agnostic (depends on the llama.cpp version — the KV cache itself is stored in the model's internal precision, which may be separate from weight quantization). In other cases, it fails with a size mismatch. I built a compatibility matrix showing which cross-quant combinations work. This is a practical contribution because it tells you whether heterogeneous quantization is feasible with llama.cpp."

### Q: "What is NUMA and why does it matter here?"

A: "NUMA — Non-Uniform Memory Access. In multi-socket servers, each CPU socket has its own local memory. Accessing memory attached to the other socket goes through an inter-socket link (UPI/QPI) with 2-3x higher latency. When running LLM inference, if your threads span both sockets, some memory accesses hit remote memory, degrading throughput. I used `numactl` to pin threads to a single socket and measured the throughput difference. The 'NUMA cliff' is the core count where you start crossing socket boundaries."

### Q: "How did you measure TTFT vs TPOT?"

A: "TTFT is measured as the wall-clock time of `llm.eval(tokens)` — the prefill call that processes all prompt tokens in parallel. TPOT is measured per-token: I time each `llm.sample()` call individually during decode and compute mean and standard deviation. Throughput is `tokens_generated / total_decode_time`. All timing uses `time.perf_counter()` for microsecond precision. I run each configuration 3 times and report mean with standard deviation."

In code: `src/baseline/benchmark.py:126-129` for TTFT, `137-149` for per-token TPOT.

### Q: "What's the difference between throughput and goodput?"

A: "Throughput is total tokens generated per second. Goodput is throughput under latency SLO constraints — for example, 'how many requests can you serve while keeping p99 TTFT < 200ms?' DistServe optimizes goodput, not raw throughput. In my experiments, I measure raw throughput but the disaggregated architecture inherently improves goodput because prefill no longer blocks decode (and vice versa)."

### Q: "How did you ensure reproducibility?"

A: "Several ways: (1) Greedy sampling with temp=0.0 and top_k=1 — deterministic token generation. (2) Fixed prompts defined in YAML config. (3) Three repetitions per configuration for variance estimation. (4) All code, configs, and setup scripts are in the GitHub repo. (5) Setup scripts for both local Mac dev and Chameleon Cloud reproduce the exact environment."

### Q: "What's the scaling efficiency formula?"

A: "Scaling efficiency = actual_throughput / ideal_throughput, where ideal assumes linear scaling from 1 thread. If 1 thread gives 5 tokens/sec, ideal at 16 threads is 80 tokens/sec. If actual is 40 tokens/sec, efficiency is 0.5 (50%). This drops due to Amdahl's law (serial portions), memory bandwidth saturation, cache thrashing, and NUMA effects."

In code: `src/analysis/plot_scaling.py:76-103`.

### Q: "Why FastAPI for the router instead of raw gRPC?"

A: "The router serves two roles: (1) HTTP API frontend for clients (easier to integrate than gRPC in most applications), and (2) orchestration layer that manages the prefill-to-decode handoff. FastAPI gives me automatic OpenAPI docs, Pydantic request validation, and async support. Prometheus metrics integration is straightforward with `prometheus_client`. The router-to-decode communication uses gRPC because that's binary payload transfer where efficiency matters."

### Q: "What would you do differently if you had more time?"

A: "Three things:
1. **Batched prefill** — currently each request runs prefill independently. Batching multiple prompts together for prefill (like continuous batching in vLLM) would amortize compute overhead.
2. **Pipelined KV transfer** — start streaming KV cache to the decode node before prefill completes (layer-by-layer), hiding transfer latency behind prefill compute.
3. **Adaptive routing** — instead of round-robin, route based on decode node queue depth and current load to minimize tail latency."

### Q: "How does this relate to vLLM's PagedAttention?"

A: "PagedAttention from vLLM solves a different problem — it manages KV cache memory efficiently using virtual-memory-inspired paging to avoid fragmentation. My work is orthogonal: I'm separating WHERE prefill and decode run, not HOW the KV cache is allocated. In a production system, you'd want both — PagedAttention on each node for memory efficiency, plus disaggregation across nodes for workload separation."

### Q: "What's the CAP theorem relevance here?"

A: "The disaggregated system faces a distributed systems tradeoff: the router needs to know which decode nodes are available (availability), the KV state must be consistent between prefill and decode (consistency), and network partitions between nodes must be handled (partition tolerance). I chose AP over CP — if a decode node goes down, the router skips it and uses another, accepting that in-flight requests to that node are lost. For inference serving, availability matters more than strict consistency."

### Q: "How did you handle concurrent requests?"

A: "The decode server uses a gRPC `ThreadPoolExecutor` with 8 workers. Each `GenerateTokens` call is handled in a separate thread. The prefill side (when run in the router) is single-threaded since `llama.cpp` inference is itself multi-threaded internally via `n_threads`. The router uses Uvicorn's async event loop for HTTP handling. For true concurrent prefill, you'd need multiple prefill server instances."

### Q: "What's the memory bandwidth of the CPUs you tested on?"

A: "Chameleon bare-metal nodes typically have dual-socket Intel Xeon CPUs with DDR4 memory. Theoretical peak bandwidth is ~100-200 GB/s depending on the specific node type (number of memory channels). I estimate effective bandwidth from my benchmarks: `model_size * tokens_generated / decode_time`. For the 7B model at Q4 (~4GB), generating 10 tokens in ~1 second suggests ~40 GB/s effective bandwidth — about 20-40% of theoretical peak, which is typical for real workloads."

### Q: "What are the limitations of using `save_state()`/`load_state()` for KV transfer?"

A: "Three main limitations:
1. **Size** — it serializes the ENTIRE context, not just the KV cache. This includes logits buffer and RNG state, adding overhead.
2. **No incremental transfer** — you can't send just the delta (new tokens' KV entries). Every transfer sends the full state.
3. **Version coupling** — sender and receiver must use the same llama.cpp version or the byte layout may differ.

A production system would implement a custom KV cache serialization that sends only the KV tensors, potentially with compression."

### Q: "Explain the quantization schemes — what's the difference between Q4_0 and Q2_K?"

A: "Q4_0 is simple 4-bit quantization: each weight is stored as a 4-bit integer with a single scaling factor per block (typically 32 weights per block). Q2_K uses the k-quant scheme which is more sophisticated — it groups weights and uses different bit widths for different groups based on weight importance (similar to the AWQ insight that salient weights matter more). Q2_K achieves better quality at 2-bit than naive 2-bit quantization by spending more bits on important weights."

### Q: "What's AVX-512 and why does it matter?"

A: "AVX-512 is an x86 SIMD instruction set that processes 512 bits (sixteen 32-bit floats or sixty-four 8-bit integers) in a single instruction. For LLM inference, the GEMM operations (matrix multiplies) are the bottleneck. AVX-512 instructions let llama.cpp's hand-tuned kernels process 2x more data per instruction than AVX2 (256-bit). On Chameleon's Xeon CPUs with AVX-512 support, this can double prefill throughput. The setup script detects AVX-512 support and compiles llama.cpp with the appropriate flags."

In code: `setup/install_chameleon.sh:37-41`.

### Q: "How did you design the experiment configuration?"

A: "Everything is config-driven via two YAML files:
- `models.yaml` defines the 3 models with their HuggingFace repos and local paths for each quant variant
- `experiments.yaml` defines core counts (1 to 128), prompt variants (short/medium/long at ~128/512/2048 tokens), P:D ratios (1:1, 1:2, 2:1), heterogeneous quant combinations, context window size, and local dev overrides

This separation means I can change the experiment matrix without touching code. The sweep loop in `src/baseline/sweep.py` generates the full Cartesian product of configurations automatically."

### Q: "What monitoring did you set up?"

A: "Four Prometheus histogram metrics exported from the router's `/metrics` endpoint, scraped every 5 seconds. A Grafana dashboard with 8 panels showing TTFT p50/p95/p99, TPOT p50/p95, KV transfer latency, decode throughput, request rate, error rate, per-node CPU utilization, and estimated memory bandwidth. The Prometheus config also scrapes `node_exporter` on all cluster nodes for system-level metrics."

---

## 8. Potential Weak Spots and How to Handle Them

### "Did you actually run on Chameleon or just have the setup?"

Honest answer: "The infrastructure is fully set up — provisioning scripts, NUMA-aware execution, multi-node configuration. [If you ran it: 'I ran experiments on X-core nodes and got Y results.'] [If not yet: 'The colocated baseline runs locally on my M2 Mac for development. The Chameleon experiments are configured and ready to deploy.']"

### "Your decode server doesn't handle batching"

A: "Correct — this is intentionally simple to isolate the disaggregation variable. Each request gets its own decode session. In production, you'd want continuous batching (process multiple sequences simultaneously) like vLLM does. That's an optimization on top of the disaggregation architecture, not a replacement for it."

### "The remote prefill path says 'NotImplementedError'"

A: "For single-node testing and development, prefill runs in-process within the router (`--local-prefill` flag). The gRPC-based remote prefill path is designed but not fully wired because the primary multi-node scenario launches prefill and decode as separate processes on different machines, where the router acts as the prefill node itself. In a true production deployment, you'd want a separate prefill gRPC service."

### "Why not use vLLM which already does disaggregation?"

A: "vLLM's disaggregation is GPU-first — it relies on CUDA, NCCL for tensor transfer, and PagedAttention with GPU memory management. None of that works on CPU-only systems. llama.cpp is the only mature framework with optimized CPU kernels (AVX2/AVX-512) and the state serialization API needed for CPU disaggregation."

---

## 9. Buzzwords to Naturally Drop

- **Disaggregated serving** — the core concept
- **KV cache** — shows you understand transformer internals
- **NUMA-aware** — shows systems-level hardware knowledge
- **Prefill/decode phase splitting** — precise terminology
- **Arithmetic intensity** — compute/byte ratio, explains why phases have different bottlenecks
- **Goodput vs throughput** — production serving metric literacy
- **AVX-512 SIMD** — CPU microarchitecture awareness
- **gRPC server-side streaming** — distributed systems pattern
- **Quantization-performance tradeoff** — model compression awareness
- **Amdahl's law** — explains sub-linear scaling
- **Memory bandwidth saturation** — hardware bottleneck vocabulary
- **GGUF format** — llama.cpp model format, shows familiarity with ecosystem
- **Bare-metal provisioning** — Chameleon Cloud, shows infrastructure skills
- **Prometheus histograms** — observability best practices
- **Breakeven analysis** — engineering decision framework

---

## 10. Whiteboard-Ready Diagrams

### Diagram 1: Phase Bottleneck Comparison

```
PREFILL (Compute-Bound)          DECODE (Memory-Bandwidth-Bound)
+---------------------------+    +---------------------------+
|  Input: N tokens           |    |  Input: 1 token           |
|  Operation: QK^T (N x N)  |    |  Operation: W * x (1 pass) |
|  FLOPs: O(N^2 * d)        |    |  FLOPs: O(d^2)            |
|  Memory: O(N * d)          |    |  Memory: O(params) ~= 14GB|
|  Arithmetic Intensity: HIGH|    |  Arithmetic Intensity: LOW |
|  Bottleneck: CPU cores     |    |  Bottleneck: Mem bandwidth |
+---------------------------+    +---------------------------+
```

### Diagram 2: Colocated vs Disaggregated Timeline

```
COLOCATED (interference):
|--PREFILL--|--DECODE---|--PREFILL--|--DECODE---|
            ↑ prefill interference degrades decode

DISAGGREGATED:
Node A: |--PREFILL--|--PREFILL--|--PREFILL--|  (specialized)
                 ↓ KV cache transfer
Node B:     |----DECODE----|----DECODE----|    (specialized)
```

### Diagram 3: KV Cache Transfer Overhead

```
Total End-to-End Latency:
|--- TTFT ---|-- KV Transfer --|------ Decode (TPOT * N) ------|

Breakeven: disaggregation wins when
  (colocated_TPOT - disagg_TPOT) * N_tokens > KV_transfer_time
```

---

## Quick Reference Card

| Term | What to Say |
|------|-------------|
| TTFT | "Time to first token — measures prefill latency" |
| TPOT | "Time per output token — measures decode speed" |
| KV cache | "Cached key-value projections from attention layers, avoids recomputation" |
| Disaggregation | "Separating prefill and decode onto different nodes to eliminate interference" |
| llama.cpp | "C/C++ inference engine with hand-tuned AVX2/AVX-512 CPU kernels" |
| GGUF | "File format for quantized models used by llama.cpp" |
| P:D ratio | "Prefill-to-decode node allocation ratio — we tested 1:1, 1:2, 2:1" |
| Goodput | "Throughput that meets latency SLO — the metric that actually matters in production" |
| NUMA | "Non-uniform memory access — remote socket memory is 2-3x slower" |
| Chameleon | "NSF-funded bare-metal cloud testbed for systems research" |

---

## Final Tips

1. **Start with the "why"** — GPU scarcity is a real industry problem. Frame this as solving a practical need.
2. **Use numbers** — "7.4x goodput improvement", "87.6% GEMM time", "128 cores", "$3.50/hr vs $0.50/hr". Numbers make you credible.
3. **Show depth on demand** — Start high-level, drill into details when asked. Don't dump everything at once.
4. **Connect to their stack** — If the company uses LLMs, ask "do you do inference on CPUs at all?" This shows you're thinking about their problems.
5. **Acknowledge limitations honestly** — "The remote prefill path isn't fully wired" is fine. "I designed it this way intentionally for X reason" is better.
6. **Reference papers casually** — "As Zhong et al. showed in DistServe at OSDI..." signals you read research.
