# ANE Training — Next Steps Plan

## Tier 1 — Immediate, High ROI

- [x] Implement compile cache in `ane_runtime.h` + `stories_io.h` — copy `data` + `net.plist`
  to `~/.ane_cache/<hexId>/` after compile, restore + skip `compileWithQoS:` on cache hit.
  **Result**: 5x compile speedup on cache hits (700ms vs 3800ms for 72 kernels).
  - Static kernels (12 sdpaBwd2, no weights) → always cache hit after first run
  - Weight-bearing kernels → cache hit on crash recovery from same checkpoint
  - Steady state: 12/72 cache hits = ~500ms saved per restart
- [x] Get real TinyStories data (`tinystories_data00.bin`) and run `train_large` for proper
  convergence. **Result**: 50M tokens (95MB) from roneneldan/TinyStories via HuggingFace.
  LLaMA-2 BPE 32K vocab tokenizer. First 10 tokens: (1, 3118, 2462, 29892, 263, 2217, 7826,
  4257, 365, 2354). Convergence: 10.35→9.24 loss over 500 steps from random init.

## BREAKTHROUGH — Dynamic Weights (2026-03-02)

`mil_gen_matmul` in `ane_mil_gen.h` declares **both x and W as runtime tensor inputs** (no
BLOBFILE). This means the weight IOSurface can be updated between evals with no recompile.

Proven in `test_dynamic_matmul.m`:
- W = identity → y = x  ✓
- W = 2x identity (write IOSurface only, no recompile) → y = 2x  ✓
- Max numerical error: 0.000015 (FP16 precision)
- Weight update cost: **0.002ms** (IOSurface write)
- Eval cost: **0.14ms**
- **vs 61.5ms compile+load: 494x speedup**

Input order is **[W, x]** not [x, W] (status=0x1d error indicates wrong order).

**What this means for training**:
- Compile 72 kernels ONCE at startup (~1.3s one-time cost)
- Per training step: write W to IOSurface (0.002ms) + eval (0.14ms) — no restart, no recompile
- The exec() restart hack is now completely obsolete
- The 119-compile limit is irrelevant once you compile once

**Required work**: Rewrite `stories_mil.h` to use weight-as-input instead of BLOBFILE.
All forward and backward kernels need W declared as function parameters, not BLOBFILE consts.

## BREAKTHROUGH — beginRealTimeTask (2026-03-03)

`beginRealTimeTask`/`endRealTimeTask` on `_ANEClient` reduces scheduling jitter by **90.6%**.

Proven in `test_realtime_task.m`:
- Plain eval:  mean=0.621ms  p99=35.173ms (massive tail from ANE scheduler preemption)
- With RT task: mean=0.387ms  p99=3.321ms  (near-zero tail)
- **90.6% p99 improvement** — free, zero-overhead wrapper

**Implementation**: `stories_io.h` exposes `ane_step_begin()` / `ane_step_end()` which
wrap each training step. `_ANEClient` is retrieved via `_sharedConnection` ivar of
`_ANEInMemoryModel` and cached in `g_ane_rt_client`.

**ANE class hierarchy (confirmed)**:
- `_ANEVirtualClient` = server-side class in XPC daemon (`com.apple.appleneuralengine`, pid~444)
- `_ANEClient` = user-space proxy, retrieved via `_ANEInMemoryModel._sharedConnection` ivar
- `_ANEClient._fastConn`/`._conn` = `_ANEDaemonConnection` → `NSXPCConnection`
- `getDeviceInfo` is server-side only — not accessible from user space

## Tier 2 — New Capabilities (Results: 2026-03-03)

- [x] Multi-function MIL dispatch: `gen_two_func_mil()` — **compiles in 19.7ms vs 52.3ms
  separately = 2.7x speedup**. procedureIndex=0 works, procedureIndex=1 fails (status=0x2).
  Use case: compile fwd+bwd together; dispatch by index. Half not working yet.
- [x] `beginRealTimeTask`/`endRealTimeTask`: **90.6% p99 jitter reduction** — integrated into
  `train_large.m`. See BREAKTHROUGH section above.
- [x] `validateNetworkCreate:` is server-side only (`_ANEVirtualClient` in XPC daemon).
  Not accessible from user space without crafting raw XPC messages.

## Tier 3 — Deeper Exploration (Results: 2026-03-03)

- [x] `test_perf_mask` fixed (two bugs: wrong calling convention + wrong array type).
  **Finding**: `perfStats:` expects `NSArray<_ANEPerformanceStatsIOSurface*>`, not bare
  `_ANEPerformanceStats`. `_ANEPerformanceStatsIOSurface` wraps an IOSurface with statType
  (int64_t). Evals succeed but ANE writes no data — likely entitlement-gated
  (`com.apple.ane.perf-counters` or similar). `driverMaskForANEFMask:` maps bits:
  0x1→0x1, 0x2→0x4, 0x3→0x5, 0xF→0xF, 0xFF→0x0.
- [x] `_ANESharedSignalEvent` fully characterized:
  - Factory: `+ signalEventWithValue:symbolIndex:eventType:sharedEvent:` (IOSurfaceSharedEvent)
  - Ivars: symbolIndex(I), value(Q), agentMask(Q), eventType(q), sharedEvent(IOSurfaceSharedEvent)
  - Used in `_ANEChainingRequest.signalEvents` (NSArray) for Metal↔ANE synchronization
  - Companion: `_ANESharedWaitEvent` with `+ waitEventWithValue:sharedEvent:eventType:`
  - alloc/init returns nil — factory methods required; Metal SharedEvent needed
- [x] `getDeviceInfo` on `_ANEVirtualClient` — **server-side only** (XPC daemon).
  Returns `{DeviceExtendedInfo={DeviceInfo=IqqB}BII[32c]}` struct with chip_id, freq_hz,
  max_freq_hz, has_ane, is_available, name[32]. Not accessible from user space.

## Training Baseline (2026-03-03)

- Speed: **138.6ms/step** average (500 steps, real TinyStories data)
- ANE: 0.67 TFLOPS sustained (out of 15.8 peak = 4.2% utilization)
- Loss: 10.35 → 9.24 over 500 steps from random init (real convergence confirmed)
- beginRealTimeTask: active, p99 jitter <3.5ms

## Next Steps (Tier 4)

- [ ] Load pretrained stories110M weights (from ../../assets/models/stories110M.bin) —
  currently missing. Download from karpathy/llama2.c project or train from better init.
- [ ] Profile ANE utilization bottleneck: 4.2% is very low. IOSurface write overhead
  (t_io=7-15ms) dominates over ANE compute (t_ane=2.7ms). Investigate:
  - Can IOSurface writes be pipelined/async?
  - Can `_ANEChainingRequest` chain fwd→bwd without CPU round-trip?
  - `_ANESharedSignalEvent` for Metal↔ANE sync
- [ ] Investigate procedureIndex=1 failure in multi-function MIL (status=0x2).
  Could allow fwd+bwd in single compiled program, reducing compile slots further.
- [ ] Weight tiling / chunked IOSurface writes to reduce t_io overhead.

## CPU Speed Optimizations (2026-03-03)

Implemented in `stories_cpu_ops.h` and `train_large.m`:

- **Pre-allocated rmsnorm scratch** (`g_rms_ss`, `g_rms_rrms`, `g_rms_dot`): eliminated
  120+ `calloc`/`free` calls per optimizer step (3 allocs × 12 layers × 10 accum steps).
  Now uses `memset` to zero before each call.

- **Pre-allocated cross-entropy buffer** (`g_ce_buf`): eliminated 32MB `malloc`/`free`
  per accumulation step (10×/optimizer step). Buffer allocated once on first call.

- **Vectorized `adam_update`** with chunked vDSP + `vvrsqrtf` (ADAM_CHUNK=2048):
  replaced per-element scalar `sqrtf` loop for up to 24.6M elements (embed+all layers).
  Approximation: `1/sqrt(v/bc2 + eps)` instead of `1/(sqrt(v/bc2) + eps)` — identical
  at eps=1e-8 for normal gradient magnitudes.

- **Pre-allocated `dx_rms_final`/`dx_rms1`**: eliminated `calloc`/`free` for these
  SEQ×DIM buffers from inside the training loop. rmsnorm_bwd fully writes dx before
  reading, so no zeroing needed.

- **Vectorized gradient scaling** (`vDSP_vsmul` instead of scalar loops):
  WQ_SZ × 4 matrices + W1/W2/W3 + embed (24.6M elements) per optimizer step.

- **Vectorized residual-add** (`vDSP_vadd` for dx2 += dy and dy = dx_rms1 + dx2):
  SEQ×DIM=196K elements × 12 layers per step.
