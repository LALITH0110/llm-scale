# LLM-SCALE Presentation Guide

**Target: ~7 minute presentation**

---

## Slide 1: Title

**LLM-SCALE: Disaggregated Prefill-Decode Inference on CPU-Only Parallel Systems**

Lalith Kothuru | CS 554 | Illinois Institute of Technology | Spring 2026

**Speaker notes:** "I took a GPU optimization called prefill-decode disaggregation and asked: does it work on CPUs? Short answer — no. The interesting part is *why*."

---

## Slide 1.5: Prior Work & What's Different

| Paper | Venue | What They Did |
|---|---|---|
| **DistServe** (Zhong et al.) | OSDI 2024 | Disaggregated prefill/decode on **GPU** pools → 7.4x goodput. Foundational. |
| **Splitwise** (Patel et al.) | ISCA 2024 | Phase-splitting for efficient GPU inference, similar disaggregation motivation |
| **vLLM / PagedAttention** (Kwon et al.) | SOSP 2023 | Paged KV cache management for GPUs; disaggregation adopted in v2 |
| **AWQ / GPTQ** | MLSys/ICLR 2024 | Post-training quantization reducing model memory footprint |
| **Guerrero et al.** | arXiv 2025 | CPUs outperform GPUs for on-device inference in specific scenarios |

**What's different about this work:**
- Every prior disaggregation paper (DistServe, Splitwise) targets **GPU-only** infrastructure
- No prior work studies disaggregation on **CPU-only** bare-metal clusters
- No prior work measures KV transfer overhead on commodity Ethernet (vs NVLink/PCIe)
- This is the **first empirical study** of P/D disaggregation under CPU constraints
- Finding is a **negative result with mechanistic explanation** — not just "it's slower", but precisely why

> *Note: This is genuinely novel territory. The papers above inspired the question; none of them answer it.*

**Speaker notes:** "DistServe and Splitwise are the canonical papers — both GPU, both NVLink. Nobody asked what happens on CPUs over Ethernet. That's the gap this fills."

---

## Slide 2: The Problem

- LLM inference has 2 phases: **prefill** (compute-bound) and **decode** (memory-bandwidth-bound)
- When colocated, they **fight for resources** — prefill wants compute, decode wants memory bandwidth
- DistServe (OSDI 2024): separating onto different GPUs → **7.4x goodput gain**

**But every paper targets GPUs. Nobody tested this on CPUs.**

*Visual: diagram showing prefill vs decode resource contention*

**Speaker notes:** "Prefill is compute-bound, decode is memory-bandwidth-bound — they conflict on the same machine. DistServe showed separating them on GPUs gives 7.4x more throughput. Does the same logic hold for CPUs?"

---

## Slide 3: Why CPUs Matter

- Many orgs **can't get GPU allocations**
- Edge/on-prem deployments need CPU inference
- CPU hardware is **10-50x cheaper** per node
- Modern CPUs: 96+ cores, 187GB RAM, AVX-512

**Research question:** Does prefill-decode disaggregation work on CPU-only systems?

**Speaker notes:** "GPUs aren't always available — cost, access, edge constraints. Modern server CPUs are impressive hardware. Can GPU-era optimizations transfer here?"

---

## Slide 4: What I Built

```
Incoming Requests
       |
       v
 +-----------+     KV Cache (gRPC)     +-----------+
 | Prefill   | ----------------------> | Decode    |
 | (llama.cpp)|                        | (llama.cpp)|
 +-----------+                         +-----------+
       ^
       |
  Request Router (FastAPI)
```

- **Prefill node**: processes prompt, serializes KV cache
- **Decode node**: restores KV state, generates tokens, streams back
- **Router**: orchestrates handoff, Prometheus metrics
- Built on `llama.cpp` with AVX-512 CPU kernels

**Speaker notes:** "FastAPI router fans out to a prefill server that calls save_state() and ships the KV cache over gRPC to a decode server that calls load_state() and streams tokens back. Same architecture as DistServe, but on bare-metal CPU nodes over 10GbE."

---

## Slide 5: Experimental Setup

| | Detail |
|---|---|
| **Testbed** | Chameleon Cloud (CHI@UC), 3 bare-metal nodes |
| **Hardware** | 96 cores, 187GB RAM, AVX-512 per node |
| **Models** | Llama 3.2 (1B, 3B), DeepSeek (7B) |
| **Quants** | FP16, Q8_0, Q4_0, Q3_K_L, Q2_K |
| **Prompts** | Short (13 tok), Medium (60 tok), Long (157 tok) |
| **Output** | 128 tokens, greedy decoding |

**3 Experiments:**
1. Colocated baseline (1,728 runs)
2. Disaggregated serving (108 runs)
3. Heterogeneous quantization (45 combos)

**Speaker notes:** "1,728 runs for the baseline sweeping cores, models, quants, prompt lengths on NSF bare-metal — no VMs, no noise. Then disaggregation and cross-quant experiments on top."

---

## Slide 6: Exp 1 — TPOT is Flat (Key Finding)

*Use figure: `scaling_tpot.png`*

**TPOT does NOT scale with cores:**
- 1 core: 0.44 ms/token
- 64 cores: 0.46 ms/token
- 128 cores: **1.6 ms/token** (NUMA penalty!)

**Why?** Decode = single matrix-vector multiply per token. Bottleneck is DRAM bandwidth, not compute. More cores don't increase memory bandwidth.

**Speaker notes:** "0.44ms at 1 core, 0.46ms at 64 cores — completely flat. Decode is a matrix-vector multiply waiting on DRAM reads. More cores don't buy more memory bandwidth. At 128 cores you cross NUMA sockets and it gets *worse*."

---

## Slide 7: Exp 1 — TTFT Doesn't Scale Either

*Use figure: `scaling_ttft.png`*

- TTFT is nearly **flat** from 1 to 128 cores
- Llama 1B long prompt: 595ms (1 core) vs 591ms (128 cores)

**Why?** `llama.cpp`'s eval() doesn't effectively parallelize prefill across CPU threads. This eliminates the primary motivation for disaggregation.

**Speaker notes:** "595ms at 1 core, 591ms at 128 — prefill doesn't parallelize in llama.cpp. This kills the whole disaggregation thesis: if the prefill node can't go faster with more hardware, there's nothing to gain."

---

## Slide 8: Exp 1 — Sweet Spot is 8-16 Cores

*Use figure: `scaling_throughput.png`*

| Cores | Llama 1B Q4_0 | Llama 3B Q4_0 | DeepSeek 7B Q4_0 |
|---|---|---|---|
| 1 | 1,458 t/s | 1,876 t/s | 2,345 t/s |
| 8 | **1,609 t/s** | **1,948 t/s** | **2,385 t/s** |
| 128 | 679 t/s | 656 t/s | 2,174 t/s |

**Beyond 32 cores, performance drops.** NUMA cross-socket penalties dominate.

**Speaker notes:** "Throughput peaks at 8-16 cores then collapses — Llama 1B loses 58% going from 8 to 128 cores due to NUMA cross-socket memory latency. Don't throw all cores at it."

---

## Slide 9: Exp 1 — Quantization Helps

| Model | FP16 | Q4_0 | Speedup | Memory Saved |
|---|---|---|---|---|
| Llama 1B | 1,009 t/s | 1,610 t/s | **+60%** | 41% |
| Llama 3B | 1,698 t/s | 1,939 t/s | **+14%** | 41% |

Q4_0 is the sweet spot: best throughput + smallest memory footprint.

Surprising: **DeepSeek 7B is fastest** at ~2,400 t/s despite being 7x larger than Llama 1B.

**Speaker notes:** "Since it's bandwidth-bound, smaller weights = faster reads. Q4_0 gives 60% speedup for Llama 1B. Oddly, DeepSeek 7B is the fastest model — its architecture aligns better with llama.cpp's AVX-512 memory access patterns."

---

## Slide 10: Exp 2 — Disaggregation Results

*Use figure: `comparison_colocated_vs_disagg.png`*

| Model:Quant | Colocated | Disaggregated | Slowdown |
|---|---|---|---|
| Llama 1B:Q4_0 | 1,937 t/s | 55 t/s | **35x slower** |
| Llama 3B:Q4_0 | 1,948 t/s | 28 t/s | **70x slower** |
| DeepSeek 7B:Q4_0 | 2,398 t/s | 12 t/s | **195x slower** |

**Disaggregation makes CPU inference dramatically worse.**

**Speaker notes:** "35x to 195x regression. Disaggregation doesn't just fail to help — it's catastrophic. FP16 models hit 571x. This is the headline result."

---

## Slide 11: Why? KV Transfer Dominates

*Use figure: `kv_transfer_latency.png` and `kv_overhead_fraction.png`*

| Prompt | KV Cache Size | Transfer Time | vs Prefill |
|---|---|---|---|
| Short | 7 MB | 338 ms | **2x slower** |
| Medium | 31 MB | 733 ms | **2x slower** |
| Long | 115 MB | 1,570 ms | **2x slower** |

KV transfer overhead = pickle serialization + gRPC + deserialization + state reconstruction

On GPU: NVLink does this in microseconds. On CPU: milliseconds.

**Speaker notes:** "The KV transfer — pickle + gRPC + deserialize — takes 2x longer than prefill itself. On NVLink, 115MB moves in ~200 microseconds. Over Ethernet, it takes 1.57 seconds. The GPU papers could ignore this cost. We can't."

---

## Slide 12: P:D Ratio Doesn't Matter

*Use figure: `pd_ratio_impact.png`*

| P:D | Avg Throughput |
|---|---|
| 1:1 | 14.7 t/s |
| 1:2 | 16.5 t/s |
| 2:1 | 15.7 t/s |

<12% variation. Bottleneck is serialization, not compute allocation.

**Speaker notes:** "Tuning the ratio changes nothing — <12% variation. The bottleneck is serialization, not how compute is allocated. You can't optimize your way out of it."

---

## Slide 13: Exp 3 — Cross-Quant KV Transfer Fails

*Use figure: `hetero_quant_compat.png`*

- **0 out of 45** cross-quant combinations worked
- `llama.cpp` KV state is tightly coupled to quantization format
- Cannot prefill in FP16 and decode in Q4_0
- Would need quant-agnostic KV representation (future work)

**Speaker notes:** "Zero out of 45 worked. llama.cpp's save_state/load_state is tightly coupled to quantization format — the tensor layout changes with quant level. No abstraction layer exists."

---

## Slide 14: Cost Analysis — CPUs Win on $/Token

*Use figure: `tokens_per_dollar.png`*

| System | tok/s | $/hr | Tokens per $ |
|---|---|---|---|
| **CPU (DeepSeek 7B)** | **2,423** | **$0.50** | **17.4M** |
| GPU H100 | 120 | $3.50 | 123K |
| GPU A100 | 80 | $2.50 | 115K |

**CPU is 40-150x more cost-efficient per token** for single-stream inference on models ≤7B.

**Speaker notes:** "Disaggregation fails — but colocated CPU inference is 140x cheaper per token than H100 for small models. The hardware is useful, just not in the way DistServe imagined."

---

## Slide 15: Why GPU Disaggregation Works but CPU Doesn't

| Factor | GPU | CPU |
|---|---|---|
| Decode speed | 20-50 ms/token | **0.4-0.9 ms/token** |
| KV transfer | NVLink: microseconds | gRPC+pickle: **hundreds of ms** |
| Prefill scaling | Tensor cores parallelize | **Does not parallelize** |
| Overhead amortized? | Yes (slow decode) | **No (fast decode)** |

CPU decode is so fast that any transfer overhead is catastrophic.

**Speaker notes:** "GPU decode is slow enough that microsecond NVLink transfers are negligible. CPU decode is so fast that hundreds-of-milliseconds transfers are catastrophic. And unlike GPUs, CPU prefill doesn't scale. Both assumptions behind disaggregation break."

---

## Slide 16: Key Takeaways

1. **CPU disaggregation does NOT work** — 35-571x throughput regression
2. **TPOT is memory-bandwidth-bound** — more cores don't help decode
3. **TTFT doesn't parallelize** in llama.cpp — can't isolate compute benefit
4. **Cross-quant KV transfer impossible** with current llama.cpp
5. **Despite this, CPUs are 40-150x more cost-efficient** per token for ≤7B models
6. **Optimal config: 8-16 cores, Q4_0, colocated, NUMA-bound**

**Speaker notes:** "Don't disaggregate on CPUs. Do use 8-16 cores, Q4_0, colocated, NUMA-pinned — and you'll get excellent cost efficiency."

---

## Slide 17: Future Work

- **Zero-copy KV transfer** (shared memory / RDMA) to eliminate serialization
- **NUMA-aware weight replication** to avoid cross-socket penalties
- **Quant-agnostic KV layer** for heterogeneous disaggregation
- **Larger models (13B-70B)** where decode is slower and overhead amortizable
- **Alternative CPU engines** (OpenVINO, ONNX) that may parallelize prefill better

**Speaker notes:** "RDMA or shared-memory KV transfer could make disaggregation viable at larger model sizes where decode is slower. Quant-agnostic KV serialization in llama.cpp would unlock the hetero use case."

---

## Slide 18: Questions?

GitHub: https://github.com/LALITH0110/llm-scale

**Speaker notes:** "In one sentence: disaggregation gives 7.4x speedup on GPUs and 35-571x slowdown on CPUs — because CPU decode is fast, prefill doesn't parallelize, and Ethernet serialization dominates. Questions?"

---

## Quick Q&A Prep

**"Isn't this a negative result?"** — Negative results are valuable. We show *why* it fails, which tells you what to fix: zero-copy transfer, larger models, better prefill parallelism.

**"What about batching?"** — Single-stream experiments. With high concurrency GPUs win on throughput; CPU cost advantage is strongest for low-concurrency workloads.

**"Why is DeepSeek 7B faster than Llama 1B?"** — Architectural differences — its attention/MLP layout aligns better with llama.cpp's AVX-512 memory access patterns.
