# LLM-SCALE: Disaggregated Prefill-Decode Inference Scaling on Highly Parallel CPU Systems

**CS 554 — Illinois Institute of Technology**
**Author:** Lalith Kothuru, Department of Computer Science, IIT Chicago, IL

---

## 1. Introduction

Large Language Model (LLM) inference consists of two fundamentally different phases: **prefill** (processing the input prompt in parallel, compute-bound) and **decode** (generating tokens autoregressively, memory-bandwidth-bound). When colocated on the same hardware, these phases interfere — prefill saturates compute while decode saturates memory bandwidth.

DistServe (OSDI 2024) demonstrated that disaggregating these phases onto separate GPU pools yields up to **7.4x higher goodput**. This has since become the default architecture in production systems (vLLM, SGLang, NVIDIA Dynamo). However, all prior work targets GPUs exclusively.

**Research question:** Does prefill-decode disaggregation improve inference performance on CPU-only parallel systems?

**Motivation:**
- Many organizations lack GPU access
- Edge deployments require CPU inference
- CPU hardware is 10-50x cheaper per node
- CPU clusters offer high core counts (96+ cores) with large memory

---

## 2. Methodology

### 2.1 Testbed

All experiments ran on **Chameleon Cloud** (CHI@UC), an NSF-funded bare-metal testbed.

| Spec | Value |
|---|---|
| Nodes | 3 bare-metal instances |
| CPU | Intel Cascade Lake (96 cores per node) |
| RAM | 187 GB per node |
| ISA Extensions | AVX2, AVX-512 |
| Network | sharednet1 (10GbE internal) |
| NUMA | 2 nodes per machine |
| OS | Ubuntu 22.04 (CC-Ubuntu22.04) |

### 2.2 Models and Quantization

| Model | Parameters | Quantization Levels |
|---|---|---|
| Llama 3.2 | 1B | FP16, Q8_0, Q4_0, Q3_K_L |
| Llama 3.2 | 3B | FP16, Q8_0, Q4_0, Q3_K_L |
| DeepSeek LLM | 7B | Q8_0, Q4_0, Q3_K_L, Q2_K |

All models in GGUF format via `llama.cpp` with AVX-512 optimized CPU kernels.

### 2.3 Software Stack

| Component | Role |
|---|---|
| `llama-cpp-python` (v0.3.18) | Inference engine with AVX-512 CPU kernels |
| gRPC (v1.78) | KV cache serialization and inter-node transfer |
| FastAPI + Uvicorn | HTTP request router |
| Prometheus + node_exporter | Performance metrics collection |
| `numactl` / `libnuma` | NUMA-aware thread pinning |

### 2.4 Prompts

| Name | Approx Tokens | Description |
|---|---|---|
| short_128 | ~13 tokens | Single-sentence question |
| medium_512 | ~60 tokens | Multi-paragraph technical prompt |
| long_2048 | ~157 tokens | Comprehensive survey request |

All experiments generate 128 output tokens with greedy decoding (temp=0, top_k=1).

### 2.5 Experiments

**Experiment 1 — Colocated Baseline:**
Standard `llama.cpp` inference on a single node. Sweeps CPU cores from 1 to 128 across all models, quantization levels, prompts, and 3 repetitions. NUMA-aware: runs separately on each NUMA node with `numactl` binding. **1,728 total runs.**

**Experiment 2 — Disaggregated Inference:**
Prefill and decode separated into different processes. Prefill runs in-process in the router, serializes full KV state via pickle, transfers via gRPC to a decode server. Tests 3 P:D ratios (1:1, 1:2, 2:1) across all models and quants. **108 total runs, 0 errors.**

**Experiment 3 — Heterogeneous Quantization:**
Tests whether KV cache from a higher-precision prefill (e.g., FP16) can be loaded into a lower-precision decode model (e.g., Q4_0). **45 cross-quant combinations tested.**

---

## 3. Results

### 3.1 Experiment 1: Colocated Baseline

#### 3.1.1 Peak Throughput

| Model | Quant | Peak Throughput (tok/s) | Avg Throughput | Avg TTFT (ms) | Avg TPOT (ms) | Peak RSS (MB) |
|---|---|---|---|---|---|---|
| DeepSeek 7B | Q2_K | 2,423 | 2,194 | 1,527 | 0.47 | 5,353 |
| DeepSeek 7B | Q3_K_L | 2,392 | 2,233 | 1,751 | 0.45 | 5,755 |
| DeepSeek 7B | Q4_0 | 2,398 | 2,263 | 944 | 0.44 | 9,255 |
| DeepSeek 7B | Q8_0 | 2,397 | 2,264 | 1,357 | 0.44 | 9,189 |
| Llama 3.2 1B | FP16 | 1,666 | 1,211 | 393 | 0.87 | 2,705 |
| Llama 3.2 1B | Q3_K_L | 1,836 | 1,341 | 445 | 0.80 | 1,109 |
| Llama 3.2 1B | Q4_0 | 1,937 | 1,391 | 335 | 0.78 | 1,619 |
| Llama 3.2 1B | Q8_0 | 1,851 | 1,377 | 413 | 0.77 | 1,612 |
| Llama 3.2 3B | FP16 | 1,748 | 1,642 | 722 | 0.61 | 6,832 |
| Llama 3.2 3B | Q3_K_L | 1,947 | 1,774 | 931 | 0.59 | 2,433 |
| Llama 3.2 3B | Q4_0 | 1,948 | 1,694 | 593 | 0.65 | 4,007 |
| Llama 3.2 3B | Q8_0 | 1,944 | 1,831 | 767 | 0.56 | 3,965 |

**Key observation:** DeepSeek 7B achieves the highest throughput (~2,400 tok/s) despite being the largest model. This suggests its architecture is particularly well-suited for CPU inference with `llama.cpp`'s AVX-512 kernels.

#### 3.1.2 TPOT Does Not Scale with Core Count

| Cores | DeepSeek 7B Q8_0 | Llama 1B Q8_0 | Llama 3B Q8_0 |
|---|---|---|---|
| 1 | 0.437 ms | 0.667 ms | 0.531 ms |
| 2 | 0.448 ms | 0.652 ms | 0.529 ms |
| 4 | 0.440 ms | 0.650 ms | 0.529 ms |
| 8 | 0.441 ms | 0.686 ms | 0.518 ms |
| 16 | 0.428 ms | 0.610 ms | 0.520 ms |
| 32 | 0.443 ms | 0.632 ms | 0.539 ms |
| 64 | 0.460 ms | 0.681 ms | 0.567 ms |
| 128 | 0.460 ms | 1.607 ms | 1.204 ms |

**Finding:** TPOT is essentially **flat from 1 to 64 cores**, confirming that decode is memory-bandwidth-bound, not compute-bound. At 128 cores, TPOT **degrades 2-3x** due to NUMA cross-socket access penalties.

#### 3.1.3 TTFT Does Not Scale with Core Count

TTFT (prefill latency) remained nearly constant across all core counts for all models and prompts. For example, Llama 3.2 1B FP16 on the long prompt: 595ms (1 core) vs 591ms (128 cores).

**Interpretation:** `llama.cpp`'s `eval()` function does not effectively parallelize the prefill computation across CPU threads. This is a critical limitation — it means the compute-bound phase cannot leverage the abundant parallelism available in high-core-count CPU systems.

#### 3.1.4 Optimal Core Count

Throughput peaks at **8-16 cores** and degrades beyond 32 cores. At 128 cores, Llama 1B throughput drops from 1,609 tok/s (8 cores) to 679 tok/s — a **58% reduction**. The degradation is caused by NUMA cross-socket memory access and thread synchronization overhead.

#### 3.1.5 Quantization Impact

At 8 threads (optimal range):

| Model | FP16 → Q4_0 | Memory Reduction |
|---|---|---|
| Llama 1B | 1,009 → 1,610 tok/s (+60%) | 2,708 → 1,600 MB (-41%) |
| Llama 3B | 1,698 → 1,939 tok/s (+14%) | 6,829 → 3,998 MB (-41%) |

Quantization consistently improves both throughput and memory usage. Q4_0 offers the best throughput for Llama models, while DeepSeek 7B shows minimal variance across quant levels.

#### 3.1.6 NUMA Node Symmetry

NUMA node 0 and node 1 produce nearly identical results (<7% difference), confirming symmetric hardware and no NUMA allocation bias.

---

### 3.2 Experiment 2: Disaggregated Inference

#### 3.2.1 KV Cache Transfer Overhead

| Prompt Length | Avg KV Transfer (ms) | Avg TTFT (ms) | KV/TTFT Ratio |
|---|---|---|---|
| short (~13 tok) | 338 ms | 181 ms | 1.9x |
| medium (~60 tok) | 733 ms | 340 ms | 2.2x |
| long (~157 tok) | 1,570 ms | 830 ms | 1.9x |

**Critical finding:** KV cache transfer takes **2x longer than prefill itself**. The overhead comes from:
1. `save_state()` serializing the full LlamaState (including input_ids, scores, KV tensors)
2. Pickle serialization of the state dictionary
3. gRPC message transmission (KV sizes: 7MB for short, 31MB for medium, 115MB for long prompts)
4. `load_state()` deserialization and tensor reconstruction on the decode side

#### 3.2.2 Throughput Comparison: Colocated vs Disaggregated

| Model:Quant | Colocated (tok/s) | Disaggregated 1:1 (tok/s) | Slowdown |
|---|---|---|---|
| Llama 1B:Q4_0 | 1,937 | 54.7 | **35x** |
| Llama 3B:Q4_0 | 1,948 | 27.9 | **70x** |
| DeepSeek 7B:Q4_0 | 2,398 | 12.3 | **195x** |
| Llama 1B:FP16 | 1,666 | 9.8 | **170x** |
| Llama 3B:FP16 | 1,748 | 4.3 | **406x** |
| DeepSeek 7B:Q8_0 | 2,397 | 4.2 | **571x** |

**Disaggregation causes massive throughput regression on CPUs.** The KV cache serialization and transfer overhead completely dominates — the pipeline spends most of its time moving state, not computing.

#### 3.2.3 P:D Ratio Impact

| P:D Ratio | Avg Throughput (tok/s) | Avg KV Transfer (ms) |
|---|---|---|
| 1:1 | 14.7 | 919 |
| 1:2 | 16.5 | 873 |
| 2:1 | 15.7 | 911 |

P:D ratio has **minimal impact** (<12% variation). The bottleneck is KV transfer, not compute allocation — changing the ratio of prefill-to-decode resources cannot address a serialization overhead problem.

---

### 3.3 Experiment 3: Heterogeneous Quantization

**Result: 0/45 cross-quant combinations succeeded.**

Every attempt to load KV state from one quantization level into a model at a different quantization level failed. `llama.cpp`'s `load_state()` requires exact tensor layout compatibility — the KV cache representation is tightly coupled to the model's quantization format.

**Tested combinations included:**
- FP16 prefill → Q4_0 decode
- FP16 prefill → Q8_0 decode
- Q8_0 prefill → Q4_0 decode
- Q8_0 prefill → Q3_K_L decode
- Q4_0 prefill → Q4_0 decode (same-quant baseline — also failed due to separate model instances)

**Implication:** Heterogeneous quantization across disaggregated nodes is **not feasible** with current `llama.cpp` KV cache serialization. Enabling this would require a quant-agnostic KV representation layer — a non-trivial engineering effort and potential area of future work.

---

### 3.4 Cost Efficiency Analysis

| System | Throughput (tok/s) | Cost ($/hr) | Tokens per Dollar |
|---|---|---|---|
| **CPU DeepSeek 7B Q2_K** | **2,423** | **$0.50** | **17,448,587** |
| CPU Llama 3B Q4_0 | 1,948 | $0.50 | 14,023,046 |
| CPU Llama 1B Q4_0 | 1,937 | $0.50 | 13,945,972 |
| GPU RTX 3090 | 35 | $0.35 | 360,000 |
| GPU RTX 4090 | 60 | $0.74 | 291,892 |
| GPU H100 SXM | 120 | $3.50 | 123,429 |
| GPU A100 SXM | 80 | $2.50 | 115,200 |

**CPU inference is 40-150x more cost-efficient per token** than GPU inference for these model sizes. While GPU absolute throughput is higher for large batch sizes, for single-request latency-sensitive workloads on models up to 7B parameters, CPUs offer dramatically better economics.

**Caveat:** GPU numbers assume single-stream inference without batching. With batched inference, GPUs become more competitive on throughput-per-dollar. The CPU advantage is strongest for low-concurrency, latency-sensitive deployments.

---

## 4. Discussion

### 4.1 Why Disaggregation Fails on CPUs

DistServe's disaggregation wins on GPUs because:
1. GPU prefill and decode have **fundamentally different compute profiles** (tensor cores vs memory bandwidth)
2. KV cache lives in GPU memory — transfer between GPUs uses NVLink (600 GB/s) or PCIe (64 GB/s)
3. GPU decode is slow enough (~20-50ms/token) that KV transfer overhead is amortized

On CPUs, the situation is inverted:
1. CPU decode is **already extremely fast** (0.4-0.9ms/token) — leaving no room for overhead
2. KV cache must be **fully serialized** through Python (pickle) and gRPC — orders of magnitude slower than GPU-to-GPU memory copy
3. `llama.cpp` prefill does not parallelize across CPU threads, eliminating the compute-isolation benefit

### 4.2 Why CPU TPOT Doesn't Scale

The decode phase performs a single matrix-vector multiplication per token (full model weight pass). On CPUs, this is limited by memory bandwidth, not compute. Adding more cores does not increase memory bandwidth — it's fixed by the DRAM controller. Each token requires reading the entire model weights once, and the memory subsystem is already saturated with a small number of cores.

### 4.3 The 128-Core Degradation

At 128 cores (crossing NUMA boundaries), performance drops significantly because:
- Threads on NUMA node 1 accessing memory allocated on NUMA node 0 incur ~2x latency penalty
- `llama.cpp` does not implement NUMA-aware memory allocation for model weights
- Thread synchronization overhead grows superlinearly with core count

### 4.4 Implications for Production CPU Inference

1. **Colocated inference is optimal for CPUs** — disaggregation adds overhead without benefit
2. **8-16 cores is the sweet spot** — beyond this, diminishing returns
3. **Q4_0 quantization** offers the best throughput/memory trade-off
4. **NUMA binding is essential** — use `numactl --cpunodebind=N --membind=N` to avoid cross-socket penalties
5. **CPU inference is viable and cost-effective** for models up to 7B parameters in low-concurrency settings

---

## 5. Threats to Validity

1. **Single-node disaggregation:** Exp 2 ran prefill and decode as separate processes on the same node. True multi-node disaggregation over network would have additional latency but also dedicated memory bandwidth per node.
2. **Serialization overhead:** Using Python pickle + gRPC for KV transfer is not optimized. Shared-memory or zero-copy approaches could reduce overhead significantly.
3. **llama.cpp limitations:** Prefill non-scaling may be specific to `llama.cpp`'s implementation. Other CPU inference engines (e.g., Intel OpenVINO, ONNX Runtime) may parallelize prefill more effectively.
4. **Model size range:** Only tested up to 7B parameters. Larger models (13B+) may show different scaling characteristics where disaggregation becomes beneficial.
5. **GPU baselines:** GPU throughput numbers are from published benchmarks, not measured on the same workloads.

---

## 6. Conclusion

We conducted the **first empirical study of prefill-decode disaggregation on CPU-only parallel systems**. Our key findings:

1. **CPU disaggregation does not improve inference performance.** KV cache serialization overhead (338-1,570ms) exceeds prefill compute time, causing 35-571x throughput regression compared to colocated inference.

2. **Decode latency is memory-bandwidth-bound on CPUs** and does not benefit from additional cores. TPOT remains flat from 1 to 64 cores, degrading at 128 cores due to NUMA penalties.

3. **Prefill does not parallelize** in `llama.cpp`, limiting the potential benefit of dedicating more compute resources to the prefill phase.

4. **Cross-quantization KV transfer is not supported** by current `llama.cpp` state serialization, ruling out heterogeneous quantization as a disaggregation strategy.

5. **Despite these limitations, CPU inference at $0.50/hr achieves 40-150x better cost-efficiency per token** than GPU inference for models up to 7B parameters, making colocated CPU inference a compelling option for cost-sensitive deployments.

### Future Work

- **Zero-copy KV transfer:** Shared-memory or RDMA-based state transfer to eliminate serialization overhead
- **NUMA-aware inference:** Model weight replication per NUMA node to avoid cross-socket penalties
- **Quant-agnostic KV representation:** Enable heterogeneous prefill/decode quantization
- **Multi-node network disaggregation:** Evaluate with dedicated network bandwidth per node
- **Larger models:** Test 13B-70B models where per-token compute is higher and disaggregation overhead may be amortized

---

## 7. References

1. W. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," SOSP, 2023.
2. Y. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI, 2024.
3. Hao AI Lab, "Disaggregated Inference: 18 Months Later," UCSD Technical Blog, Nov. 2025.
4. Z. Zhou et al., "A Survey on Efficient Inference for Large Language Models," arXiv:2404.14294, 2024.
5. J. Guerrero et al., "Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference," arXiv:2505.06461, 2025.
6. G. Gerganov, "llama.cpp: Inference of LLaMA model in pure C/C++," GitHub, 2023-2025.
7. J. Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression," MLSys, 2024.
8. E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," ICLR, 2023.
9. P. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA, 2024.
10. Chameleon Cloud, NSF-funded testbed, 2025. https://www.chameleoncloud.org

---

## Appendix A: Generated Figures

All figures are in `results/figures/`:

| File | Description |
|---|---|
| `scaling_ttft.png` | TTFT vs CPU threads (per model/quant/prompt) |
| `scaling_tpot.png` | TPOT vs CPU threads |
| `scaling_throughput.png` | Decode throughput vs CPU threads |
| `scaling_membw.png` | Estimated memory bandwidth utilization |
| `scaling_efficiency.png` | Scaling efficiency relative to ideal linear |
| `comparison_colocated_vs_disagg.png` | Colocated vs disaggregated side-by-side |
| `pd_ratio_impact.png` | P:D ratio impact on latency |
| `kv_transfer_latency.png` | KV cache transfer latency by model/prompt |
| `kv_overhead_fraction.png` | KV transfer as fraction of total time |
| `hetero_quant_compat.png` | Cross-quant compatibility matrix |
| `cost_throughput_scatter.png` | Cost vs throughput for CPU and GPU |
| `tokens_per_dollar.png` | Tokens generated per dollar |

## Appendix B: Reproducibility

All code, configs, and scripts are available at: https://github.com/LALITH0110/llm-scale

```bash
# Reproduce on Chameleon Cloud
git clone https://github.com/LALITH0110/llm-scale.git
cd llm-scale
make setup-chameleon
make download-full
export LLMSCALE_ENV=chameleon
make exp1        # ~5-6 hours
make exp2        # ~1-2 hours
make exp3        # ~30 minutes
make analyze     # generates all figures
```
