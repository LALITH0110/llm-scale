# LLM-SCALE — Next Steps (Round 2)

Prof feedback + planned extensions. All experiments run on Chameleon Cloud.

---

## 1. Open Questions From Professor

1. **Is CPU really better on $/token?** Current claim (40–150× better tokens/$) uses single-stream GPU = underused. Apples-to-oranges. Must re-measure w/ batched GPU inference + matched concurrency.
2. **GPU prefill + CPU decode** — prefill compute-bound (GPU wins), decode memory-BW bound (CPU adequate). Hybrid may beat both extremes.
3. **Full cross-config sweep** — need numbers for:
   - 2× GPU (prefill-GPU + decode-GPU, true DistServe-on-Chameleon)
   - 1× GPU + 1× CPU (hybrid, prefill-GPU + decode-CPU)
   - 2× CPU (current Exp2 — already have)
   - 1× GPU colocated (baseline GPU)
   - 1× CPU colocated (Exp1 — already have)

---

## 2. New Configurations to Benchmark

All on Chameleon. Same models, same prompts, same 128-token output (plus 512/2048 sweep for amortization).

| Config ID | Prefill | Decode | Purpose |
|---|---|---|---|
| **C1** | CPU | CPU (same node) | Colocated baseline — **DONE (Exp1)** |
| **C2** | CPU | CPU (diff node) | Disaggregated CPU — **DONE (Exp2)** |
| **C3** | GPU | GPU (same node) | GPU colocated baseline — **TODO** |
| **C4** | GPU | GPU (diff node) | Full GPU disaggregation (DistServe replica) — **TODO** |
| **C5** | GPU | CPU | **Hybrid** (the hypothesis) — **TODO** |
| **C6** | CPU | GPU | Reverse hybrid (sanity check, expect bad) — **TODO** |

6 configs × 3 models × 3 quants × 3 prompt lens × 3 output lens = ~500 runs w/ reps.

---

## 3. Models to Add

Keep Llama 3.2 (1B, 3B) + DeepSeek 7B. Add:

- **Gemma 3** — 1B, 4B, 12B, 27B (GGUF available)
- **DeepSeek-R1-Distill** — Qwen 7B, Llama 8B variants
- **Qwen 2.5** — 7B, 14B (control group)
- (Optional) **Llama 3.3 70B Q4** — stress test decode BW limit

---

## 4. Chameleon Reservation Plan

### Nodes needed

| Node | Hardware | Purpose |
|---|---|---|
| GPU-A | CHI@UC `gpu_rtx_6000` or `gpu_a100` | C3, C4 prefill, C5 prefill |
| GPU-B | Same as GPU-A | C4 decode |
| CPU-A | Cascade Lake 96c (current) | C1, C2 prefill, C5 decode, C6 prefill |
| CPU-B | Cascade Lake 96c | C2 decode, C6 decode |

- All on **sharednet1** (same VLAN, 10GbE)
- Lease: ~72h continuous for full sweep
- Fallback: RTX 6000 if A100 unavailable

### Build matrix

- GPU nodes: `llama.cpp` built w/ `GGML_CUDA=1`, `llama-cpp-python[cuda]` wheel
- CPU nodes: `llama.cpp` AVX-512 (current build)
- Same model weights (GGUF) portable across backends

---

## 5. New Experiments

### Exp 4 — GPU Colocated Baseline (C3)

- Single GPU node, standard llama.cpp CUDA
- Sweep batch size: 1, 4, 16, 32, 64 (critical for honest GPU $/tok)
- All models × quants × prompts
- Metrics: TTFT, TPOT, throughput, **goodput** (SLO: TTFT<500ms, TPOT<50ms), GPU util, VRAM

### Exp 5 — Full GPU Disaggregation (C4, 2× GPU)

- Replicates DistServe architecture on Chameleon
- Prefill GPU → KV gRPC → Decode GPU
- Same sweep as Exp 4 + P:D ratios
- Validates whether DistServe's 7.4× speedup holds on Chameleon's network (vs NVLink)
- **Key comparison point**: if our KV transfer over 10GbE kills GPU disagg too, that's a major finding

### Exp 6 — GPU Prefill + CPU Decode Hybrid (C5) ⭐ main new contribution

- GPU runs prefill (FP16 weights)
- KV cache → gRPC → CPU decode node (Q4_0 weights)
- **Implementation phases:**
  - **Phase A (quick)**: FP16 both sides, skip quant mismatch, prove concept — 1 day
  - **Phase B (real)**: FP16 prefill + Q4 decode, handle KV repack or use `--cache-type-k f16` on CPU side — 3 days
- Sweep output lengths 128 / 512 / 2048 → find break-even point
- Expected pattern: hybrid wins when `TTFT_saved_vs_CPU > KV_transfer_cost`

### Exp 7 — CPU Prefill + GPU Decode (C6)

- Sanity check — expect this to lose (worst of both worlds)
- Small run, 1 model × 1 quant, just enough to confirm

### Exp 8 — GPU Batched Baseline for Honest Cost Comparison

- vLLM on 1× A100 (or Chameleon GPU)
- Batch sizes 1, 8, 32, 64, 128
- Same models, same prompts
- **Fixes the $/tok claim** — replaces hand-wavy GPU single-stream numbers in current §3.4

---

## 6. Cost Analysis Rework

Current §3.4 is the weakest section. Replace with:

- **Pareto frontier plot**: x = latency-SLO compliance rate, y = $/Mtok
- **Matched-concurrency table**: CPU batch=1 vs GPU batch={1, 32, 64, 128}
- **Break-even curves**:
  - CPU beats GPU when concurrency < X
  - Hybrid beats CPU when output_len > Y
  - Hybrid beats GPU when model < Z params
- **Scenario split**: spot vs on-demand; cloud hourly vs amortized co-lo
- **Chameleon cost model**: use published rates, note academic allocation is "free" but equivalent commercial price used

Figs to add:
- `pareto_cost_latency.png`
- `hybrid_breakeven_output_length.png`
- `goodput_vs_concurrency.png`
- `config_comparison_all_six.png` (C1–C6 bars)

---

## 7. Implementation Order (priority)

1. **Gemma + DeepSeek-R1 on existing Exp1** — low risk, ~1 day, 200 runs subset (8/16/32c × Q4/Q8)
2. **Exp 8 GPU batched baseline** — fixes cost claim, required for honest paper
3. **Exp 4 GPU colocated (C3)** — foundation for all GPU configs
4. **Exp 6 Hybrid Phase A** (FP16 both sides) — prove hybrid end-to-end works
5. **Exp 5 Full GPU disagg (C4)** — can reuse Exp 4 code + gRPC from Exp 2
6. **Exp 6 Hybrid Phase B** (FP16→Q4 real version)
7. **Exp 7 reverse hybrid (C6)** — quick sanity
8. **Rework §3.4 cost + new figs**
9. **Rewrite §4 discussion** — possibly pivot narrative: "CPU disagg fails; **GPU-prefill + CPU-decode hybrid** is the right architecture for cost-sensitive LLM serving"

---

## 8. Code Changes Needed

| File | Change |
|---|---|
| `config/models.yaml` | Add Gemma 3, DeepSeek-R1-Distill, Qwen 2.5 entries |
| `config/experiments.yaml` | Add exp4/5/6/7/8 sections; batch size sweep |
| `setup/install_chameleon.sh` | Add CUDA llama.cpp build path for GPU nodes |
| `src/disaggregated/prefill_server.py` | Backend flag (cpu / cuda); FP16 KV extract |
| `src/disaggregated/decode_server.py` | Handle incoming FP16 KV; `--cache-type-k f16` option |
| `src/experiments/exp4_gpu_colocated.py` | **NEW** |
| `src/experiments/exp5_gpu_disagg.py` | **NEW** (or param of exp2) |
| `src/experiments/exp6_hybrid.py` | **NEW** |
| `src/experiments/exp7_reverse_hybrid.py` | **NEW** |
| `src/experiments/exp8_gpu_batched.py` | **NEW** (vLLM-based) |
| `src/analysis/cost_analysis.py` | Pareto + break-even + goodput plots |
| `src/analysis/plot_hybrid.py` | **NEW** — hybrid-specific figs |

---

## 9. Expected Outcomes (predictions to test)

- **C3 GPU colocated**: wins on absolute throughput at batch>=32, loses on $/tok at batch=1
- **C4 GPU disagg**: likely **also loses** on 10GbE (not NVLink) → supports narrative that transport, not compute, is the real bottleneck
- **C5 Hybrid**: wins on TTFT vs CPU-only; wins on $/tok vs GPU-only for long outputs + small models; expected Pareto sweet spot
- **C6 Reverse**: loses on everything (confirmation)

---

## 10. Unresolved Questions

- GPU node avail on Chameleon — A100 vs RTX 6000 (affects perf + cost math)?
- Lease window — 72h continuous possible or split?
- Include 70B in this round or defer?
- Use vLLM for GPU batched or stick w/ llama.cpp CUDA for apples-to-apples?
- Hybrid Phase B — implement KV quant repack ourselves or patch llama.cpp upstream?
- Prof OK w/ narrative pivot if hybrid wins?
- Deadline — full sweep fits remaining weeks?
- Keep original Exp3 cross-quant failure in paper or demote to appendix?
- Report length budget — IEEE 8 pages already tight, adding 3 configs may need supplementary material
