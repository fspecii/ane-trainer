# ANE Training — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results (M4, Stories110M — 12 layers, dim=768, seq=256, vocab=32000):**
- **~52 ms/step**, **~16% ANE utilization** (2.5 TFLOPS sustained)
- 87 kernels compiled once at startup — no per-batch recompile
- All forward + backward dx + dW gradient accumulation on ANE
- Classifier forward + backward on ANE (32K-vocab matmul)
- Adam optimizer with sparse embed updates, gradient accumulation (10 steps/batch)

## Architecture

Weights are passed as **dynamic IOSurface inputs** — compiled once at startup, updated in-place after each Adam step with no recompilation. This is the key architectural advance over the BLOBFILE approach (which requires recompiling all kernels after every weight update).

### ANE Kernels (7 per layer × 12 layers = 84, plus 3 classifier kernels)

| Kernel | Function | Inputs |
|--------|----------|--------|
| `fwdFwd` | RMSNorm1 + QKV + SDPA + Wo + RMSNorm2 + SwiGLU FFN (fused) | x, rms1, Wq, Wk, Wv, Wo, rms2, W1, W3, W2 |
| `ffnBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | dx, h1, h3 |
| `sdpaBwd12` | SDPA backward (fused: dV, softmax grad, dQ, dK) | probs, qf, kf, vf, dx2 |
| `qkvBwd` | QKV backward (Wq^T + Wk^T + Wv^T → dx) | dq, dk, dv |
| `dwW2` | dW2 gradient accumulation | dffn, gate |
| `dwW13` | dW1 + dW3 gradient accumulation | dh1, dh3, x2n |
| `dwWoQKV` | dWo + dWq + dWk + dWv gradient accumulation | dx2, af, xn |
| `classifierFwd` | embed @ x_final → logits (32K-ch matmul) | x_final, embed |
| `softmax` | Softmax over vocab axis | logits |
| `clsBwd` | embed^T @ dlogits → dy (classifier backward) | dlogits, embed |

CPU handles: RMSNorm backward, residual additions, cross-entropy loss + gradient, sparse embed Adam, layer Adam.

### Key Optimizations

- **Dynamic IOSurface weights** — weights written to IOSurface once per Adam step, no recompile. Eliminates the per-batch compile cost (was ~660ms/batch) entirely.
- **Fused fwdFwd kernel** — attention + FFN forward in one ANE eval. Saves 12 kernel dispatches/step, keeps all forward taps (qf, kf, vf, probs, h1, h3, gate, x2n) in ANE memory for backward.
- **Fused sdpaBwd12** — sdpaBwd1 + sdpaBwd2 merged; saves probs in fwdFwd output to avoid recompute and eliminate cycle-path errors.
- **ANE dW kernels** — weight gradient accumulation on ANE via matmul, no CPU cblas or activation round-trips. Eliminates cblas_wait from the critical path.
- **IOSurface-native backward taps** — backward kernels read forward activations directly from ANE IOSurfaces via `io_copy`, no CPU readback.
- **ANE classifier backward** — `embed^T @ dlogits` (6.3 GFLOPS) on ANE, replaces CPU cblas_sgemm in backward critical path. Saves ~10ms/step.
- **beginRealTimeTask** — `_ANEClient` real-time scheduling reduces p99 dispatch jitter from 35ms → 3ms (90% reduction).
- **Sparse embed Adam** — tracks unique tokens per batch, updates only seen rows (~7.7% of 32K vocab). Embed Adam 80ms → 3ms.
- **Parallel layer Adam** — 12-layer Adam dispatched to per-layer serial GCD queues, all layers update concurrently.
- **Channel-first layout** — activations stored `[C, S]` throughout, matching ANE IOSurface `[1,C,1,S]` format. No transpose overhead.

## Performance History

All measurements on M4, 12-layer Stories110M (dim=768, seq=256), random init, `ACCUM_STEPS=10`.

| Optimization | ms/step | ANE util | Notes |
|---|---|---|---|
| M6: Dynamic weights baseline | 138 | 4% | Compile once; dynamic IOSurface weights |
| M7: ANE classifier + parallel dW queues | 96 | 7% | cblas dW overlap |
| Phase 6: ANE dW kernels | 55 | 15.5% | cblas_wait → 0ms |
| Plan D: fused sdpaBwd12 | 63 | — | Net neutral (probs save cost = dispatch save) |
| Plan G: fused fwdFwd | 61 | — | 12 fewer ANE evals/step |
| Plan H: ANE classifier backward | **52** | **~16%** | Eliminated ~10ms cblas_sgemm |

## Building

Requires macOS 15+ on Apple Silicon.

```bash
cd training && make train_large

# Or manually:
xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
  -o train_large train_large.m \
  -framework Foundation -framework CoreML \
  -framework IOSurface -ldl -framework Accelerate

# Run (random init)
./train_large --steps 1000

# Run with pretrained weights
./train_large stories110M.bin --steps 1000

# Resume from checkpoint
./train_large --resume
```

No external dependencies. All ANE APIs resolved at runtime via `objc_msgSend` and `dlopen`.

## File Structure

```
└── training/
    ├── train_large.m         # Main 12-layer training loop
    ├── stories_mil.h         # MIL kernel generators (fwdFwd, ffnBwd, sdpaBwd12, qkvBwd, dW kernels)
    ├── stories_config.h      # Model constants, structs, alloc helpers
    ├── stories_io.h          # IOSurface helpers, compile_kern_dyn, ane_eval, beginRealTimeTask
    ├── stories_cpu_ops.h     # RMSNorm fwd/bwd, Adam (NEON fused), cross-entropy, embed ops
    ├── ane_classifier.h      # Classifier forward, softmax, cls_bwd MIL generators
    ├── ane_mil_gen.h         # Reference matmul/conv MIL generators
    ├── ane_runtime.h         # Low-level ANE wrapper (standalone)
    ├── PLAN.md               # Optimization history and notes
    └── Makefile
```

## How It Works

1. **MIL generation** — Objective-C constructs MIL program text at runtime: conv for linear layers (weights as runtime tensor inputs), matmul for attention and dW accumulation, softmax, element-wise ops.
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL + optional static weight blobs to ANE programs in memory; no `.mlmodelc` on disk.
3. **IOSurface I/O** — Activations and weights passed via IOSurface shared memory in `[1, C, 1, S]` fp16 format. `io_copy` moves data between surfaces without CPU round-trip.
4. **Dynamic weight updates** — After each Adam step, updated weights are written to weight IOSurfaces (`io_write_fp16`). ANE reads the new values on the next eval with no recompile.
5. **Gradient flow** — `fwdFwd` kernel exposes 13 output channels (activations + forward taps). Backward kernels (`ffnBwd`, `sdpaBwd12`, `qkvBwd`) compute dx on ANE; dW kernels (`dwW2`, `dwW13`, `dwWoQKV`) accumulate weight gradients on ANE via matmul. CPU handles RMSNorm backward and Adam.

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into Q@K^T (ANE) → add(mask) → softmax (ANE) → scores@V (ANE), with mask as a static BLOBFILE constant.
- **fp16 precision** — All ANE compute is fp16; Adam moments and weights are maintained in fp32 on CPU.
- **No RoPE** — Rotary position embeddings not yet implemented on ANE.

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)
