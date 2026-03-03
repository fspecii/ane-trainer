# ANE Optimization Plan

## Current State (post-Phase 7 / Plan C)
- Forward+backward: ~54ms/step (10-step batch = ~540ms)
- Adam: ~37ms/batch (3.7ms/step amortized)
- Total effective: ~57.7ms/step
- ANE utilization: ~15.8% of 15.8 TFLOPS peak
- FLOPs on ANE: ~78% (136B/174B per step)
- ANE kernels per layer: fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd, dwW2, dwW13, dwWoQKV = 9
- Total ANE evals per step: 9×12 = 108

## Plan A: Backward Attention Kernel Fusion (FAILED)
Goal: Fuse sdpaBwd1 + sdpaBwd2 + qkvBwd into single bwdAttn kernel

Failed: ANE compiler "cycle path" errors — fused kernel had data dependency cycles
that the compiler refused to compile. Restored to Phase 6 state.

bwdAttn spec (for future retry):
- Input:  [1, 4*DIM, 1, SEQ] = qf|kf|vf|dx2f
- Weights: Wo, Wq, Wk, Wv each [1,DIM,DIM]
- Output: [1, 4*DIM, 1, SEQ] = dx_qkv|dvf|dqf|dkf
- 4*DIM = 3072 channels OK

Expected: 108->84 ANE evals/step, ~45ms/step, ~18-20% util
Status: Deferred — cycle path errors need investigation

## Plan B: ANE Adam (INFEASIBLE)
Moving Adam optimizer to ANE. Three hard blockers:
1. fp16 moment precision: b1^t ≈ 2.6e-46 after ~17 steps (underflows fp16)
2. Per-step bias correction factors 1/(1-b1^t) change every batch → can't be MIL constants
3. ANE dispatch overhead ~0.02ms >> element-wise compute ~0.37µs per tensor

## Plan C: Parallel + Fused Adam (COMPLETE)
Implemented in phases — total improvement: Adam 80ms → 37ms per 10-step batch.

### C1: Thread-safe adam_update
- Removed `static` from mh/vh/g2 stack buffers — enables parallel dispatch

### C2: Parallel layer Adam
- Dispatched 12-layer Adam+write_layer_weights to existing per-layer dw_q[L] queues
- All 12 layers run concurrently; serial queue order ensures Adam before weight write
- Saved ~6ms vs serial layer loop

### C3: adam_update_fused with inline gsc
- New NEON-explicit Adam: vrsqrteq_f32 + one Newton-Raphson step
- Fuses gradient scaling (gsc=1/steps_batch) into a single DRAM pass
- Eliminates vDSP chunking overhead (was 120K dispatches for embed)
- All Adam calls switched from chunked vDSP to fused NEON

### C4: Sparse embed Adam
- Track unique tokens per batch (embed_seen[VOCAB] uint8_t, set during embed_backward)
- Expected ~2464 unique tokens / 32000 vocab (~7.7% of rows) per 10-step batch
- Only process seen rows: ~37MB vs 491MB (~13x less DRAM traffic)
- Sparse zero of gembed inside the loop; eliminates 98MB memset per batch

### C5: Fused embed Adam + IOSurface write
- Lock classifier IOSurface once per Adam step
- Within sparse loop: Adam update then cvt_f32_f16 to IOSurface in one pass
- Wrote rows are cache-hot from Adam → near-zero extra DRAM traffic for fp16 write
- Eliminated 13ms full-surface io_write_fp16 call
- Result: embed 80ms → 3ms (27x speedup)

### Measured Results
| Stage            | Adam/batch | Embed | Layer+wait | Notes                |
|------------------|-----------|-------|------------|----------------------|
| Phase 6 baseline | 80ms      | 80ms  | 0ms        | embed was bottleneck |
| After C1-C3      | 45ms      | 11ms  | 34ms       | layers now bottleneck|
| After C4-C5      | 37ms      |  3ms  | 34ms       | bandwidth-bound      |

Layer Adam is bandwidth-bound: 12 layers × ~200MB × 7 passes / 68GB/s ≈ 37ms min.

## Plan D: sdpaBwd1 + sdpaBwd2 Fusion (COMPLETE — net neutral)
Fused sdpaBwd1+sdpaBwd2 into single gen_sdpa_bwd12 kernel.
- Cycle-path blocker resolved by saving forward-pass softmax aw in fwdAttn ioOut (ch 6*DIM)
  - gen_sdpa_fwd_taps: output expanded to [oo|qf|kf|vf|af|xn|aw_flat] = (6*DIM+SCORE_CH)*SEQ
  - gen_sdpa_bwd12: input [probs_flat|qf|kf|vf|dx2f] = (SCORE_CH+4*DIM)*SEQ — no softmax recompute
- Result: 108 → 96 ANE evals/step (12 fewer: sdpaBwd2 + io_copy per layer eliminated)
- Measured performance: ~63-65ms/step vs ~57.7ms Plan C baseline
  - Net neutral: 12 fewer ANE evals (~5ms savings) offset by +18MB/step probs write to fwdAttn ioOut
  - The saved probs (SCORE_CH=3072 ch × 12 layers = 18MB) costs bandwidth ≈ what we saved in XPC dispatch
- Status: Compiles, runs correctly, 96 evals/step confirmed

## Option E: Pipelined Adam (Future)
Overlap Adam with next batch's first steps by streaming layer-by-layer:
- Layer L's Adam dispatched → immediately start forward pass for layer L+1
- Requires fine-grained dependency tracking (each layer's write must precede its read)
- Potential: hide most of 37ms Adam behind forward pass
- High implementation complexity

## Option F: Reduce Weight Write Bandwidth
write_layer_weights accounts for ~12ms of the 37ms layer Adam (12 layers × ~1MB fp16)
- Cache weights in fp16 on CPU; Adam updates fp32 copy then converts
- Already doing this. No further optimization without fp16 Adam precision issues.

## Plan G: Fused fwdFwd Kernel (COMPLETE)
Fused gen_sdpa_fwd_taps + gen_ffn_fwd_taps into single gen_fwd_fused kernel.
- 9 weights: rms1, Wq, Wk, Wv, Wo, rms2, W1, W3, W2
- 13-value concat output: x_out|oo|qf|kf|vf|af|xn|aw_flat|x2|h1|h3|gate|x2n
- Output size: (9*DIM+SCORE_CH+3*HIDDEN)*SEQ = 16128*256 channels
- Eliminates 12 fwdFFN ane_eval + 12 fwdFFN io_write per step
- Also moves 2 residual vDSP_vadd per layer into kernel (24 fewer CPU ops)
- Evals/step: 96 → 84 (7 kernels × 12 layers)
- Measured improvement: ~2-4ms/step (~61ms vs ~63-65ms Plan D)
  - Theoretical savings: 12 × 420µs = ~5ms
  - Partial offset: larger fwdFwd ioOut (8MB vs 4MB) flush + x2 io_read back
- 86 kernels compiled at startup (vs 74 old count, now correctly printed)

## Plan H: ANE Classifier Backward (COMPLETE)
Added gen_cls_bwd(): embed^T @ dlogits → dy on ANE, replacing CPU cblas_sgemm in backward critical path.
- Kernel: [1,VOCAB,1,SEQ] dlogits + [1,VOCAB,DIM] We → [1,DIM,1,SEQ] dy
  - matmul(transpose_x=true, x=We, y=dl3) — same IOSurface shape as classifier forward
  - We IOSurface written once per Adam batch (io_write_fp16, 49MB, ~0.7ms/batch)
- Per-step cost: io_write dlogits (16MB, ~0.24ms) + ane_eval (~0.5ms) + io_read dy (0.4MB)
- Eliminated: cblas_sgemm(CblasTrans, DIM, SEQ, VOCAB) = 6.3 GFLOPS/step on CPU (~10ms)
- Measured: cls(fwd+bwd)=4.5ms vs old ~12ms (cls_fwd ~2ms + untracked cblas ~10ms)
- Step time: ~61ms (Plan G) → ~52ms (Plan H) ≈ -9ms/step
- 87 kernels compiled at startup
