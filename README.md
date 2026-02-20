# LLM-SCALE: Disaggregated Prefill-Decode Inference Scaling on Highly Parallel CPU Systems

**CS 554 — Illinois Institute of Technology**
**Author:** Lalith Kothuru, Department of Computer Science, IIT Chicago, IL

---

## Abstract

Large Language Model (LLM) inference consists of two fundamentally different phases:

- **Prefill** — processes the entire input prompt in parallel; *compute-bound*
- **Decode** — generates tokens autoregressively; *memory-bandwidth-bound*

When colocated on the same hardware, these phases interfere with each other, degrading both Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT). Recent work on disaggregated serving (e.g., DistServe) demonstrated up to **7.4× higher goodput** on GPU clusters by separating these phases onto independent compute pools.

However, the applicability of prefill-decode disaggregation to **CPU-only, highly parallel systems** remains unexplored. This project, **LLM-SCALE**, is an empirical study that adapts disaggregated inference to multi-node CPU clusters on the Mystic and Chameleon testbeds.

---

## Overview

LLM-SCALE benchmarks **Llama 3.2 (1B, 3B)** and **DeepSeek (7B)** models using `llama.cpp` across colocated and disaggregated configurations, varying:

- CPU core counts (1 → up to 128 cores)
- Quantization levels: FP16, Q8\_0 (INT8), Q4\_0 (4-bit), Q2\_K (2-bit)

**Key metrics:** TTFT · TPOT · Throughput (tokens/s) · Memory bandwidth utilization · Throughput-per-dollar

The goal is to determine whether CPU-based disaggregated inference can offer a **cost-effective alternative** for organizations constrained by GPU availability.

---

## System Architecture

A two-tier CPU inference pipeline:

```
Incoming Requests
       │
       ▼
 ┌─────────────┐      KV Cache (gRPC/ZeroMQ)      ┌─────────────┐
 │ Prefill Node │  ────────────────────────────►  │ Decode Node  │
 │  (llama.cpp) │                                  │  (llama.cpp) │
 └─────────────┘                                  └─────────────┘
       ▲
       │
  Request Router
```

- **Prefill nodes** receive incoming requests, process the full prompt, and generate the KV cache.
- The **KV cache** is transferred over the network via gRPC or ZeroMQ to dedicated decode nodes.
- **Decode nodes** perform autoregressive token generation.
- A lightweight **request router** distributes requests and manages the prefill-to-decode handoff.
- The **baseline** colocates both phases on the same node using standard `llama.cpp` execution.

---

## Models and Quantization

| Model         | Parameters | Quantization Levels Tested           |
|---------------|------------|--------------------------------------|
| Llama 3.2     | 1B         | FP16, Q8\_0, Q4\_0, Q2\_K           |
| Llama 3.2     | 3B         | FP16, Q8\_0, Q4\_0, Q2\_K           |
| DeepSeek LLM  | 7B         | FP16, Q8\_0, Q4\_0, Q2\_K           |

All models use GGUF format via `llama.cpp`. Heterogeneous quantization (e.g., FP16 prefill + Q4 decode) is also evaluated in disaggregated configurations.

---

## Testbed and Infrastructure

| Cluster     | Use Case                                         |
|-------------|--------------------------------------------------|
| [Mystic](http://mystic.cs.iit.edu) | Multi-core single-node scaling experiments |
| [Chameleon Cloud](https://www.chameleoncloud.org) | Bare-metal multi-node disaggregated experiments |

Core counts are varied from 1 to the maximum available (up to 128 cores) to study scaling behavior.

---

## Software Stack

| Component     | Purpose                                                    |
|---------------|------------------------------------------------------------|
| `llama.cpp`   | Primary inference engine with AVX2/AVX-512 CPU kernels     |
| gRPC          | KV cache serialization and inter-node transfer             |
| Python orchestration | Disaggregated pipeline coordination               |
| Prometheus + Grafana | Real-time performance metrics collection          |
| `numactl` / `libnuma` | NUMA-aware thread pinning and memory allocation  |

---

## Experiments

### Experiment 1 — Colocated Baseline
Standard `llama.cpp` inference on a single node, scaling CPU cores from 1 to max, across all models and quantization levels. Measures TTFT, TPOT, and throughput.

### Experiment 2 — Disaggregated Inference
Prefill and decode separated onto different nodes. Varies the prefill-to-decode node ratio (1:1, 1:2, 2:1) and repeats all measurements.

### Experiment 3 — Heterogeneous Quantization
In the disaggregated setup, applies different quantization levels to prefill and decode nodes independently to determine if mixed-precision configurations improve overall performance.

---

## Metrics

| Metric                  | Description                                                            |
|-------------------------|------------------------------------------------------------------------|
| TTFT                    | Time-to-first-token (latency of prefill phase)                         |
| TPOT                    | Time-per-output-token (decode throughput)                              |
| Throughput              | Tokens/second under concurrent load                                    |
| Memory bandwidth        | Utilization during prefill and decode phases                           |
| Throughput-per-dollar   | Normalized throughput vs. hourly CPU cost; compared against H100/A100  |

---

## Project Timeline

| Weeks | Tasks                                                                                                       |
|-------|-------------------------------------------------------------------------------------------------------------|
| 1–2   | Environment setup: provision Mystic/Chameleon nodes, build `llama.cpp`, download and quantize models to GGUF |
| 3–4   | Implement colocated baseline benchmarks; run Experiment 1 across all core counts and quantization levels     |
| 5–6   | Design and implement disaggregated pipeline: gRPC KV cache transfer, request router, prefill/decode separation |
| 7–8   | Run Experiments 2 & 3; set up Prometheus/Grafana monitoring                                                  |
| 9–10  | Data analysis: generate scaling curves, compare colocated vs. disaggregated, identify KV transfer breakeven  |
| 11–12 | Write final IEEE-format report on Overleaf; prepare presentation; finalize code repo and documentation       |

---

## Deliverables

1. **Reproducible benchmarking framework** for colocated and disaggregated CPU-based LLM inference, hosted on GitHub.
2. **Disaggregated inference prototype** using `llama.cpp` with gRPC-based KV cache transfer.
3. **Final report** in IEEE format with comprehensive performance analysis.
4. **Presentation** covering methodology, results, and cost-efficiency conclusions.

---

## Background and Related Work

| Work          | Contribution                                                              |
|---------------|---------------------------------------------------------------------------|
| DistServe [2] | Disaggregates prefill and decode onto different GPUs; 7.4× goodput gain   |
| vLLM [1]      | PagedAttention for KV cache management; 2–4× throughput improvement       |
| Splitwise [9] | Phase splitting across compute- vs. memory-optimized machines             |
| AWQ [7]       | Activation-aware 4-bit quantization (MLSys 2024 Best Paper)               |
| GPTQ [8]      | Post-training quantization for generative models (ICLR 2023)              |
| Guerrero et al. [5] | CPUs can outperform GPUs for smaller models under certain conditions |

> To our knowledge, no prior work studies prefill-decode disaggregation on CPU-only parallel systems.

---

## References

1. W. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," *SOSP*, 2023.
2. Y. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," *OSDI*, 2024.
3. Hao AI Lab, "Disaggregated Inference: 18 Months Later," UCSD Technical Blog, Nov. 2025.
4. Z. Zhou et al., "A Survey on Efficient Inference for Large Language Models," *arXiv:2404.14294*, 2024.
5. J. Guerrero et al., "Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference," *arXiv:2505.06461*, 2025.
6. G. Gerganov, "llama.cpp: Inference of LLaMA model in pure C/C++," GitHub, 2023–2025.
7. J. Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression," *MLSys*, 2024. *(Best Paper)*
8. E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," *ICLR*, 2023.
9. P. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," *ISCA*, 2024.
10. Chameleon Cloud, NSF-funded testbed, 2025. https://www.chameleoncloud.org
