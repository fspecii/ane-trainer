// train_large.m — Train stories110M (12 layers, 768dim, 3072hidden) on ANE
// Uses pretokenized TinyStories data with cross-entropy loss
// 72 kernels compiled once at startup (dynamic weights — no per-batch recompile)
// + 2 new: ANE classifier (32000-ch conv) + ANE softmax = 74 total
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"
#include "ane_classifier.h"

#define CKPT_PATH "ane_stories110M_ckpt.bin"
#define MODEL_PATH "../../assets/models/stories110M.bin"
#define DATA_PATH "tinystories_data00.bin"

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch! Expected dim=%d hidden=%d layers=%d\n", DIM, HIDDEN, NLAYERS);
        fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    bool shared = cfg.vocab_size > 0;

    // Read in llama2.c order: embed, rms_att[all], wq[all], wk[all], wv[all], wo[all],
    //                         rms_ffn[all], w1[all], w2[all], w3[all], rms_final, [wcls]
    fread(embed, 4, V * DIM, f);

    // rms_att weights for all layers (contiguous)
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    // wq for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    // wk for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wk, 4, WQ_SZ, f);
    // wv for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wv, 4, WQ_SZ, f);
    // wo for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    // rms_ffn weights for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    // w1 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    // w2 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    // w3 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    // rms_final
    fread(rms_final, 4, DIM, f);
    // wcls = embed if shared (we just use embed pointer)

    fclose(f);
    printf("  Loaded pretrained weights (%s)\n", shared ? "shared embed/cls" : "separate cls");
    return true;
}

// ===== Compile one layer's kernels (dynamic weights — compile once at startup) =====
// Weights are NOT baked in; they're written to wIns IOSurfaces before each eval.
static bool compile_layer_kernels(LayerKernels *lk) {
    NSDictionary *mask_dict = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}
    };

    // fwdAttn: wIns[0]=rms1[DIM], [1]=Wq[DIM²], [2]=Wk[DIM²], [3]=Wv[DIM²], [4]=Wo[DIM²]
    int fwd_attn_w[] = {DIM*2, DIM*DIM*2, DIM*DIM*2, DIM*DIM*2, DIM*DIM*2};
    lk->fwdAttn = compile_kern_dyn(gen_sdpa_fwd_taps(), mask_dict,
        DIM*SEQ*2, fwd_attn_w, 5, (6*DIM+SCORE_CH)*SEQ*2);

    // fwdFFN: wIns[0]=rms2[DIM], [1]=W1[HIDDEN×DIM], [2]=W3[HIDDEN×DIM], [3]=W2[DIM×HIDDEN]
    int fwd_ffn_w[] = {DIM*2, HIDDEN*DIM*2, HIDDEN*DIM*2, DIM*HIDDEN*2};
    lk->fwdFFN = compile_kern_dyn(gen_ffn_fwd_taps(), @{},
        DIM*SEQ*2, fwd_ffn_w, 4, (2*DIM+3*HIDDEN)*SEQ*2);

    // ffnBwd: wIns[0]=W2[DIM×HIDDEN], [1]=W1[HIDDEN×DIM], [2]=W3[HIDDEN×DIM]
    // transpose_x=true used in MIL — same weight shape, no separate transposed copy
    int ffn_bwd_w[] = {DIM*HIDDEN*2, HIDDEN*DIM*2, HIDDEN*DIM*2};
    lk->ffnBwd = compile_kern_dyn(gen_ffn_bwd(), @{},
        (DIM+2*HIDDEN)*SEQ*2, ffn_bwd_w, 3, (DIM+2*HIDDEN)*SEQ*2);

    // sdpaBwd12: fused Wo+SDPA bwd1+2; in=[probs_flat|qf|kf|vf|dx2f]; out=[dqf|dkf|dvf]
    // probs_flat from fwdAttn ioOut ch 6*DIM — no softmax recompute, no cycle-path errors
    int bwd12_w[] = {DIM*DIM*2};
    lk->sdpaBwd12 = compile_kern_dyn(gen_sdpa_bwd12(), mask_dict,
        (SCORE_CH+4*DIM)*SEQ*2, bwd12_w, 1, 3*DIM*SEQ*2);
    // qkvBwd: dq+dk+dv → dx; wIns[0]=Wq, [1]=Wk, [2]=Wv
    int qkv_bwd_w[] = {DIM*DIM*2, DIM*DIM*2, DIM*DIM*2};
    lk->qkvBwd = compile_kern_dyn(gen_qkvb(), @{},
        3*DIM*SEQ*2, qkv_bwd_w, 3, DIM*SEQ*2);

    // dwW2: dffn @ silu^T = dW2[DIM,HIDDEN]; input [DIM+HIDDEN,SEQ], output [DIM,HIDDEN]
    lk->dwW2 = compile_kern_dyn(gen_dw_w2(), @{},
        (DIM+HIDDEN)*SEQ*2, NULL, 0, DIM*HIDDEN*2);

    // dwW13: dh13 @ x2n^T = concat(dW1,dW3)[2*HIDDEN,DIM]; input [2*HIDDEN+DIM,SEQ], output [2*HIDDEN,DIM]
    lk->dwW13 = compile_kern_dyn(gen_dw_w13(), @{},
        (2*HIDDEN+DIM)*SEQ*2, NULL, 0, 2*HIDDEN*DIM*2);

    // dwWoQKV: outer products for Wo,Wq,Wk,Wv; input [6*DIM,SEQ], output [4*DIM,DIM]
    lk->dwWoQKV = compile_kern_dyn(gen_dw_woqkv(), @{},
        6*DIM*SEQ*2, NULL, 0, 4*DIM*DIM*2);

    return lk->fwdAttn && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd12 && lk->qkvBwd
        && lk->dwW2 && lk->dwW13 && lk->dwWoQKV;
}

// Write current layer weights to all weight IOSurfaces.
// Call once after Adam update (or at startup) to sync weights with ANE kernels.
static void write_layer_weights(LayerKernels *lk, LayerWeights *w) {
    // fwdAttn: [0]=rms_att, [1]=Wq, [2]=Wk, [3]=Wv, [4]=Wo
    io_write_wf16(lk->fwdAttn->wIns[0], w->rms_att, DIM);
    io_write_wf16(lk->fwdAttn->wIns[1], w->Wq, DIM*DIM);
    io_write_wf16(lk->fwdAttn->wIns[2], w->Wk, DIM*DIM);
    io_write_wf16(lk->fwdAttn->wIns[3], w->Wv, DIM*DIM);
    io_write_wf16(lk->fwdAttn->wIns[4], w->Wo, DIM*DIM);
    // fwdFFN: [0]=rms_ffn, [1]=W1, [2]=W3, [3]=W2
    io_write_wf16(lk->fwdFFN->wIns[0], w->rms_ffn, DIM);
    io_write_wf16(lk->fwdFFN->wIns[1], w->W1, HIDDEN*DIM);
    io_write_wf16(lk->fwdFFN->wIns[2], w->W3, HIDDEN*DIM);
    io_write_wf16(lk->fwdFFN->wIns[3], w->W2, DIM*HIDDEN);
    // ffnBwd: [0]=W2, [1]=W1, [2]=W3
    io_write_wf16(lk->ffnBwd->wIns[0], w->W2, DIM*HIDDEN);
    io_write_wf16(lk->ffnBwd->wIns[1], w->W1, HIDDEN*DIM);
    io_write_wf16(lk->ffnBwd->wIns[2], w->W3, HIDDEN*DIM);
    // sdpaBwd12: [0]=Wo
    io_write_wf16(lk->sdpaBwd12->wIns[0], w->Wo, DIM*DIM);
    // qkvBwd: [0]=Wq, [1]=Wk, [2]=Wv
    io_write_wf16(lk->qkvBwd->wIns[0], w->Wq, DIM*DIM);
    io_write_wf16(lk->qkvBwd->wIns[1], w->Wk, DIM*DIM);
    io_write_wf16(lk->qkvBwd->wIns[2], w->Wv, DIM*DIM);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd12); free_kern(lk->qkvBwd);
    free_kern(lk->dwW2); free_kern(lk->dwW13); free_kern(lk->dwWoQKV);
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd12 = lk->qkvBwd = NULL;
    lk->dwW2 = lk->dwW13 = lk->dwWoQKV = NULL;
}

// ===== Checkpoint save/load =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double cc, double ct, double cw, int cs, int cb, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb; h.adam_t = adam_t;
    fwrite(&h, sizeof(h), 1, f);
    // Per-layer weights + adam
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WQ_SZ,f);
        fwrite(lw[L].Wv,4,WQ_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        // Adam state
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WQ_SZ,f); fwrite(la[L].Wk.v,4,WQ_SZ,f);
        fwrite(la[L].Wv.m,4,WQ_SZ,f); fwrite(la[L].Wv.v,4,WQ_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *cc, double *ct, double *cw, int *cs, int *cb, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WQ_SZ,f);
        fread(lw[L].Wv,4,WQ_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WQ_SZ,f); fread(la[L].Wk.v,4,WQ_SZ,f);
        fread(la[L].Wv.m,4,WQ_SZ,f); fread(la[L].Wv.v,4,WQ_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;

        // Parse args
        bool do_resume = false;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) lr = atof(argv[++i]);
        }

        // Allocate per-layer state
        LayerWeights lw[NLAYERS];
        LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS];
        LayerGrads grads[NLAYERS];
        LayerKernels kern[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc();
            la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc();
            grads[L] = layer_grads_alloc();
            memset(&kern[L], 0, sizeof(LayerKernels));
        }

        // Final RMSNorm + embedding + classifier
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);  // [VOCAB, DIM] row-major
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);
        uint8_t *embed_seen = (uint8_t*)calloc(VOCAB, 1);

        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;

        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Training: Stories110M (12 layers) ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
            if (!load_pretrained(lw, rms_final, embed, MODEL_PATH)) {
                printf("Pretrained load failed, using random init\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wq[i]=scale_d*(2*drand48()-1);lw[L].Wk[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wv[i]=scale_d*(2*drand48()-1);lw[L].Wo[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            }
            size_t tp = (size_t)NLAYERS*LAYER_PARAMS + DIM + (size_t)VOCAB*DIM;
            double xfmr_params = (double)NLAYERS*LAYER_PARAMS;
            double embed_params = (double)VOCAB*DIM;
            printf("Params: %.2fM (transformer %.2fM + embed %.2fM)\n", tp/1e6, xfmr_params/1e6, embed_params/1e6);
            printf("Kernels: %d (%d weight-bearing + %d dW)\n",
                   TOTAL_WEIGHT_KERNELS+3*NLAYERS, TOTAL_WEIGHT_KERNELS, 3*NLAYERS);
            printf("Accum %d steps/batch | Adam LR=%.1e b1=%.1f b2=%.3f | dynamic weights\n", ACCUM_STEPS, lr, adam_b1, adam_b2);
            double fwd_f = NLAYERS*(4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
            double bwd_dx_f = fwd_f, bwd_dw_f = fwd_f;
            double sdpa_f = NLAYERS*2.0*HEADS*5*SEQ*SEQ*HD;
            double cls_f = 2.0*VOCAB*DIM*SEQ;
            double total_f = fwd_f + bwd_dx_f + bwd_dw_f + sdpa_f + cls_f*3;
            double ane_f = fwd_f + bwd_dx_f + bwd_dw_f + sdpa_f;
            printf("FLOPs/step: fwd=%.0fM bwd_dx=%.0fM bwd_dW=%.0fM sdpa_bwd=%.0fM total=%.0fM\n",
                   fwd_f/1e6, bwd_dx_f/1e6, bwd_dw_f/1e6, sdpa_f/1e6, total_f/1e6);
            printf("ANE FLOPs/step: %.0fM (fwd+bwd_dx+bwd_dW+sdpa_bwd) | CPU: cls only\n\n", ane_f/1e6);
        }

        // mmap token data
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Gradient buffers shared across layers (reused each step)
        float *dy = (float*)malloc(SEQ*DIM*4);            // gradient flowing backward
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);

        // x buffer for input to each layer (channel-first [DIM, SEQ])
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);     // after final rmsnorm
        float *probs   = (float*)malloc((size_t)SEQ*VOCAB*4);  // ANE softmax output [VOCAB,SEQ]
        float *dlogits = (float*)malloc((size_t)SEQ*VOCAB*4);
        // Pre-allocated to avoid calloc inside the training loop.
        // rmsnorm_bwd writes dx fully before reading, so no zeroing needed.
        float *dx_rms_final = (float*)malloc(SEQ*DIM*4);
        float *dx_rms1      = (float*)malloc(SEQ*DIM*4);

        // Compile all kernels once at startup (dynamic weights — no recompile per batch)
        uint64_t tc0 = mach_absolute_time();
        for (int L=0; L<NLAYERS; L++) {
            printf("  Compiling layer %d/%d...\r", L+1, NLAYERS); fflush(stdout);
            if (!compile_layer_kernels(&kern[L])) {
                printf("\nCompile failed at layer %d\n", L); return 1;
            }
        }
        // Classifier: embed [VOCAB,DIM] as dynamic weight, updated per Adam step
        int cls_w[1] = {(int)((size_t)VOCAB*DIM*2)};
        Kern *classifierKern = compile_kern_dyn(gen_classifier_fwd_dyn(), @{},
            DIM*SEQ*2, cls_w, 1, (int)((size_t)VOCAB*SEQ*2));
        if (!classifierKern) { printf("classifier compile failed\n"); return 1; }
        // Softmax: no weights, static kernel
        Kern *softmaxKern = compile_kern_mil_w(gen_softmax_vocab(), @{},
            (int)((size_t)VOCAB*SEQ*2), (int)((size_t)VOCAB*SEQ*2));
        if (!softmaxKern) { printf("softmax compile failed\n"); return 1; }

        double startup_compile_ms = tb_ms(mach_absolute_time() - tc0);
        printf("  Compiled %d kernels in %.0fms (one-time cost)\n",
               TOTAL_WEIGHT_KERNELS + NLAYERS + 2, startup_compile_ms);

        // Write initial weights to IOSurfaces
        for (int L=0; L<NLAYERS; L++) write_layer_weights(&kern[L], &lw[L]);
        io_write_fp16(classifierKern->wIns[0], embed, VOCAB, DIM);

        // Cache _ANEClient for beginRealTimeTask/endRealTimeTask (90%+ jitter reduction)
        ane_step_client_init(kern[0].fwdAttn);

        // Per-layer serial queues: layer L's dW runs concurrently with all other layers.
        // Single dispatch_group tracks all pending work for the global Adam-update barrier.
        dispatch_queue_t dw_q[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            char name[32]; snprintf(name, sizeof(name), "dw.%d", L);
            dw_q[L] = dispatch_queue_create(name, DISPATCH_QUEUE_SERIAL);
        }
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_compile_ms = startup_compile_ms, total_train_ms=0;
        int total_steps_done=0, total_batches=0;
        uint64_t t_wall_start = mach_absolute_time();

        srand48(42 + start_step);

        int step = start_step;
        while (step < total_steps) {
            // Zero gradient accumulators
            for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
            memset(grms_final, 0, DIM*4);
            // gembed rows are zeroed sparsely in the Adam loop; invariant maintained across batches.

            int steps_batch = 0;
            uint64_t tt = mach_absolute_time();
            double t_ane=0,t_io=0,t_elem=0,t_rms=0,t_cblas_wait=0,t_cls=0;

            for (int a=0; a<ACCUM_STEPS && step<total_steps; a++, step++) {
                ane_step_begin();
                uint64_t t0,t1;
                // Sample random position in token data
                size_t max_pos = n_tokens - SEQ - 1;
                size_t pos = (size_t)(drand48() * max_pos);
                uint16_t *input_tokens = token_data + pos;
                uint16_t *target_tokens = token_data + pos + 1;

                // Embedding lookup → x_cur [DIM, SEQ] channel-first
                t0=mach_absolute_time();
                embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0);

                // ===== FORWARD (12 layers) =====
                for (int L=0; L<NLAYERS; L++) {
                    LayerActs *ac = &acts[L];

                    // Save layer input for rmsnorm1 backward
                    memcpy(ac->layer_in, x_cur, SEQ*DIM*4);
                    // Attention forward: x_cur → o_out,Q,K,V,attn_out,xnorm
                    t0=mach_absolute_time();
                    dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                    t1=mach_absolute_time(); t_cblas_wait+=tb_ms(t1-t0); t0=t1;
                    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdAttn);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                    // FFN forward
                    io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdFFN);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0);
                }

                // Final RMSNorm (CPU — single call, cost ~0.1ms)
                t0=mach_absolute_time();
                rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0); t0=t1;

                // Classifier on ANE: x_final [DIM,SEQ] × embed [VOCAB,DIM] → logits [VOCAB,SEQ]
                // embed IOSurface is refreshed once per Adam step after weight update.
                io_write_fp16(classifierKern->ioIn, x_final, DIM, SEQ);
                ane_eval(classifierKern);
                // Softmax on ANE: io_copy avoids CPU round-trip between classifier and softmax
                io_copy(softmaxKern->ioIn, 0, classifierKern->ioOut, 0, VOCAB, SEQ);
                ane_eval(softmaxKern);
                t1=mach_absolute_time(); t_cls+=tb_ms(t1-t0); t0=t1;

                // NLL loss + gradient on CPU (needs target indexing — can't avoid readback)
                io_read_fp16(softmaxKern->ioOut, probs, 0, VOCAB, SEQ);
                float total_loss = 0.0f;
                float invS = 1.0f / SEQ;
                memcpy(dlogits, probs, (size_t)VOCAB*SEQ*4);
                for (int t = 0; t < SEQ; t++) {
                    int tgt = target_tokens[t];
                    total_loss -= logf(probs[tgt*SEQ + t] + 1e-10f);
                    dlogits[tgt*SEQ + t] -= 1.0f;
                }
                vDSP_vsmul(dlogits, 1, &invS, dlogits, 1, (vDSP_Length)((size_t)VOCAB*SEQ));
                float loss = total_loss / SEQ;
                last_loss = loss;
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                // ===== BACKWARD =====
                // dlogits already computed by cross_entropy_loss

                // Classifier backward: dx_final = embed^T @ dlogits, dembed += dlogits @ x_final^T
                // dx_final[DIM,SEQ] = embed^T[DIM,VOCAB] @ dlogits[VOCAB,SEQ]
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            DIM, SEQ, VOCAB, 1.0f,
                            embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);

                // dembed[VOCAB,DIM] += dlogits[VOCAB,SEQ] @ x_final^T[SEQ,DIM]
                dispatch_group_async(dw_grp, dw_q[0], ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                VOCAB, DIM, SEQ, 1.0f,
                                dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
                });

                // Final RMSNorm backward
                rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
                memcpy(dy, dx_rms_final, SEQ*DIM*4);

                // ===== BACKWARD (12 layers, reverse) =====
                for (int L=NLAYERS-1; L>=0; L--) {
                    LayerActs *ac = &acts[L];
                    LayerGrads *gr = &grads[L];

                    // dy is the gradient at the output of this layer
                    // dffn = dy (residual connection: d(x2 + ffn) = dy for both)
                    memcpy(dffn, dy, SEQ*DIM*4);

                    // FFN backward (ANE)
                    io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);
                    io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                    ane_eval(kern[L].ffnBwd);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0, DIM, SEQ);

                    // dW2 = dffn[DIM,SEQ] @ silu[HIDDEN,SEQ]^T
                    io_copy(kern[L].dwW2->ioIn, 0,   kern[L].ffnBwd->ioIn,  0,           DIM,    SEQ);
                    io_copy(kern[L].dwW2->ioIn, DIM,  kern[L].fwdFFN->ioOut, DIM+2*HIDDEN, HIDDEN, SEQ);
                    ane_eval(kern[L].dwW2);
                    {
                        Kern *dw2 = kern[L].dwW2;
                        float *gW2 = gr->W2;
                        dispatch_group_async(dw_grp, dw_q[L], ^{
                            IOSurfaceLock(dw2->ioOut, kIOSurfaceLockReadOnly, NULL);
                            acc_f16_f32(gW2, (const _Float16*)IOSurfaceGetBaseAddress(dw2->ioOut), DIM*HIDDEN);
                            IOSurfaceUnlock(dw2->ioOut, kIOSurfaceLockReadOnly, NULL);
                        });
                    }

                    // dW13 = concat(dh1,dh3)[2*HIDDEN,SEQ] @ x2n[DIM,SEQ]^T
                    io_copy(kern[L].dwW13->ioIn, 0,       kern[L].ffnBwd->ioOut, DIM,          2*HIDDEN, SEQ);
                    io_copy(kern[L].dwW13->ioIn, 2*HIDDEN, kern[L].fwdFFN->ioOut, DIM+3*HIDDEN, DIM,     SEQ);
                    ane_eval(kern[L].dwW13);
                    {
                        Kern *dw13 = kern[L].dwW13;
                        float *gW1 = gr->W1, *gW3 = gr->W3;
                        dispatch_group_async(dw_grp, dw_q[L], ^{
                            IOSurfaceLock(dw13->ioOut, kIOSurfaceLockReadOnly, NULL);
                            const _Float16 *base = (const _Float16*)IOSurfaceGetBaseAddress(dw13->ioOut);
                            acc_f16_f32(gW1, base,              HIDDEN*DIM);
                            acc_f16_f32(gW3, base + HIDDEN*DIM, HIDDEN*DIM);
                            IOSurfaceUnlock(dw13->ioOut, kIOSurfaceLockReadOnly, NULL);
                        });
                    }

                    // RMSNorm2 backward
                    memset(dx2, 0, SEQ*DIM*4);
                    rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                    // Add residual: dx2 += dy (from skip connection)
                    vDSP_vadd(dx2, 1, dy, 1, dx2, 1, (vDSP_Length)(SEQ*DIM));

                    // Fused SDPA backward (ANE): [probs_flat|qf|kf|vf|dx2f] → [dqf|dkf|dvf]
                    // probs_flat from fwdAttn ioOut ch 6*DIM (saved by gen_sdpa_fwd_taps)
                    io_copy(kern[L].sdpaBwd12->ioIn, 0,          kern[L].fwdAttn->ioOut, 6*DIM, SCORE_CH, SEQ);
                    io_copy(kern[L].sdpaBwd12->ioIn, SCORE_CH,   kern[L].fwdAttn->ioOut, DIM,   3*DIM,    SEQ);
                    io_write_fp16_at(kern[L].sdpaBwd12->ioIn, SCORE_CH+3*DIM, dx2, DIM, SEQ);
                    ane_eval(kern[L].sdpaBwd12);

                    // dWoQKV: outer products for Wo,Wq,Wk,Wv each [DIM,DIM]
                    // sdpaBwd12 output: [dqf|dkf|dvf] at [0|DIM|2*DIM]
                    io_copy(kern[L].dwWoQKV->ioIn, 0,     kern[L].sdpaBwd12->ioIn, SCORE_CH+3*DIM, DIM, SEQ);  // dx2f
                    io_copy(kern[L].dwWoQKV->ioIn, DIM,   kern[L].sdpaBwd12->ioOut, 0,     2*DIM, SEQ);  // dqf+dkf
                    io_copy(kern[L].dwWoQKV->ioIn, 3*DIM, kern[L].sdpaBwd12->ioOut, 2*DIM, DIM,   SEQ);  // dvf
                    io_copy(kern[L].dwWoQKV->ioIn, 4*DIM, kern[L].fwdAttn->ioOut,   4*DIM, 2*DIM, SEQ);  // attn+xnorm
                    ane_eval(kern[L].dwWoQKV);
                    {
                        Kern *dwaqkv = kern[L].dwWoQKV;
                        float *gWo = gr->Wo, *gWq = gr->Wq, *gWk = gr->Wk, *gWv = gr->Wv;
                        dispatch_group_async(dw_grp, dw_q[L], ^{
                            IOSurfaceLock(dwaqkv->ioOut, kIOSurfaceLockReadOnly, NULL);
                            const _Float16 *base = (const _Float16*)IOSurfaceGetBaseAddress(dwaqkv->ioOut);
                            acc_f16_f32(gWo, base,             DIM*DIM);
                            acc_f16_f32(gWq, base +   DIM*DIM, DIM*DIM);
                            acc_f16_f32(gWk, base + 2*DIM*DIM, DIM*DIM);
                            acc_f16_f32(gWv, base + 3*DIM*DIM, DIM*DIM);
                            IOSurfaceUnlock(dwaqkv->ioOut, kIOSurfaceLockReadOnly, NULL);
                        });
                    }

                    // QKV backward (ANE): concat(dqf,dkf,dvf) → dx
                    io_copy(kern[L].qkvBwd->ioIn, 0, kern[L].sdpaBwd12->ioOut, 0, 3*DIM, SEQ);
                    ane_eval(kern[L].qkvBwd);
                    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);

                    // RMSNorm1 backward (using saved layer input)
                    rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);

                    // dy for next layer (going backward) = dx_rms1 + dx2 residual
                    // Actually: layer output = layer_input + o_out, and x2 = layer_input + o_out
                    // So dx(layer_input) = dx_attn_rmsnorm + dx2 (residual from attn skip)
                    // Wait, dx2 already includes the attn skip residual gradient.
                    // dy = dx_rms1 (through rmsnorm1) is the gradient to the layer input
                    // But there's also the skip connection: layer_input → x2 directly
                    // So total gradient to layer_input = dx_rms1 + dx2_skip
                    // dx2 was computed as rmsnorm2_bwd + dy(ffn_skip), which already flows to x2
                    // x2 = layer_input + o_out, so d(layer_input) from x2 path = dx2
                    // And d(layer_input) from attn path through rmsnorm1 = dx_rms1
                    // Total: dy_prev = dx_rms1 (attn rmsnorm path)
                    // Wait no - dx2 = d(loss)/d(x2), not d(loss)/d(layer_input)
                    // d(layer_input) = d(loss)/d(x2) * d(x2)/d(layer_input) = dx2 (since x2 = input + o_out, d(x2)/d(input) = 1)
                    // Plus the path through rmsnorm1: dx_rms1
                    // Hmm but dx2 was already used as input to SDPA backward... let me reconsider.
                    //
                    // Actually the gradient flow is:
                    //   dy → split to (dffn, dy_skip)  [dy_skip = dy due to residual]
                    //   dffn → ffnBwd → dx_ffn
                    //   dx_ffn → rmsnorm2_bwd → dx_rms2
                    //   dx2 = dx_rms2 + dy  (skip connection from residual x2 → output)
                    //   dx2 → sdpaBwd → dx_attn through Wo^T
                    //   dx_attn → qkvBwd → dx_qkv
                    //   dx_qkv → rmsnorm1_bwd → dx_rms1
                    //   dy_prev_layer = dx_rms1 + dx2  (skip connection input → x2)
                    //
                    // So: dy for previous layer = dx_rms1 + dx2
                    vDSP_vadd(dx_rms1, 1, dx2, 1, dy, 1, (vDSP_Length)(SEQ*DIM));
                }

                // Embedding backward
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                embed_backward(gembed, dy, input_tokens, DIM, SEQ);
                for (int j = 0; j < SEQ; j++) embed_seen[input_tokens[j]] = 1;

                ane_step_end();
                steps_batch++;
                if (step % 10 == 0 || step == start_step)
                    printf("step %-4d loss=%.4f\n", step, loss);

                // JSON telemetry to stderr
                double step_ane = t_ane/steps_batch, step_io = t_io/steps_batch;
                double step_cls = t_cls/steps_batch, step_elem = t_elem/steps_batch;
                double step_rms = t_rms/steps_batch, step_cbw = t_cblas_wait/steps_batch;
                fprintf(stderr, "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,"
                    "\"t_ane\":%.3f,\"t_io\":%.3f,\"t_cls\":%.3f,"
                    "\"t_elem\":%.3f,\"t_rms\":%.3f,\"t_cblas_wait\":%.3f,"
                    "\"compiles\":%d}\n",
                    step, loss, step_ane, step_io, step_cls, step_elem, step_rms, step_cbw, g_compile_count);
            }
            double tms = tb_ms(mach_absolute_time() - tt);
            total_train_ms += tms;
            total_steps_done += steps_batch;
            total_batches++;

            // Ensure all async dW finished
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            // Adam + weight write — parallelized across layers using existing per-layer serial queues.
            // dw_q[L] is serial: Adam and write_layer_weights are in one block, so write always
            // follows Adam for that layer. All 12 layers run concurrently across queues.
            uint64_t ta0 = mach_absolute_time();
            float gsc = 1.0f / steps_batch;
            adam_t++;
            int t = adam_t;
            for (int L = 0; L < NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                LayerWeights *w = &lw[L];
                LayerAdam *a = &la[L];
                LayerKernels *kk = &kern[L];
                dispatch_group_async(dw_grp, dw_q[L], ^{
                    adam_update_fused(w->Wq, g->Wq, &a->Wq, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->Wk, g->Wk, &a->Wk, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->Wv, g->Wv, &a->Wv, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->Wo, g->Wo, &a->Wo, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->W1, g->W1, &a->W1, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->W2, g->W2, &a->W2, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->W3, g->W3, &a->W3, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->rms_att, g->rms_att, &a->rms_att, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    adam_update_fused(w->rms_ffn, g->rms_ffn, &a->rms_ffn, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                    write_layer_weights(kk, w);
                });
            }
            // rms_final + embed on main thread concurrently with layer Adams
            uint64_t te0 = mach_absolute_time();
            adam_update_fused(rms_final, grms_final, &arms_final, t, lr, adam_b1, adam_b2, adam_eps, gsc);
            uint64_t te1 = mach_absolute_time();
            IOSurfaceLock(classifierKern->wIns[0], 0, NULL);
            _Float16 *cls_base = (_Float16*)IOSurfaceGetBaseAddress(classifierKern->wIns[0]);
            for (int v = 0; v < VOCAB; v++) {
                if (!embed_seen[v]) continue;
                embed_seen[v] = 0;
                float *gv = gembed + (size_t)v * DIM;
                float *wv = embed + (size_t)v * DIM;
                AdamState rs = {aembed.m + (size_t)v * DIM, aembed.v + (size_t)v * DIM, DIM};
                adam_update_fused(wv, gv, &rs, t, lr, adam_b1, adam_b2, adam_eps, gsc);
                cvt_f32_f16(cls_base + (size_t)v * DIM, wv, DIM);
                memset(gv, 0, DIM * 4);
            }
            IOSurfaceUnlock(classifierKern->wIns[0], 0, NULL);
            double t_embed_ms = tb_ms(mach_absolute_time() - te0);
            double t_sparse_ms = tb_ms(mach_absolute_time() - te1), t_iow_ms = 0.0;
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            double t_adam_ms = tb_ms(mach_absolute_time() - ta0);

            printf("  [batch %d: train=%.1fms (%.1fms/step) adam=%.1fms (embed=%.1f sparse=%.1f iow=%.1f layers_wait=%.1f)]\n",
                   steps_batch, tms, tms/steps_batch, t_adam_ms, t_embed_ms, t_sparse_ms, t_iow_ms, t_adam_ms - t_embed_ms);
            printf("    ane=%.1f io=%.1f cls=%.1f elem=%.1f rms=%.1f cblas_wait=%.1f ms/step\n",
                   t_ane/steps_batch, t_io/steps_batch, t_cls/steps_batch, t_elem/steps_batch,
                   t_rms/steps_batch, t_cblas_wait/steps_batch);

            // JSON batch telemetry to stderr
            {
                double bf = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
                double bs = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
                double ane_f_batch = (bf*3 + bs) * steps_batch;
                double ane_tflops = ane_f_batch / (tms * 1e9);
                fprintf(stderr, "{\"type\":\"batch\",\"batch\":%d,\"train_ms\":%.1f,\"ms_per_step\":%.1f,\"adam_ms\":%.1f,\"embed_ms\":%.1f,\"layers_ms\":%.1f}\n",
                    steps_batch, tms, tms/steps_batch, t_adam_ms, t_embed_ms, t_adam_ms - t_embed_ms);
                fprintf(stderr, "{\"type\":\"perf\",\"ane_tflops\":%.3f,\"ane_util_pct\":%.2f}\n",
                    ane_tflops, 100.0*ane_tflops/15.8);
            }
        }

        // Save final checkpoint
        {
            double wall0 = tb_ms(mach_absolute_time() - t_wall_start);
            save_checkpoint(CKPT_PATH, step, total_steps, lr, last_loss,
                total_compile_ms+cum_compile, total_train_ms+cum_train, wall0+cum_wall,
                total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            printf("[checkpoint saved: step %d, loss=%.4f]\n", step, last_loss);
        }

        // Efficiency report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        total_compile_ms += cum_compile; total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;
        double fwd_flops = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
        double sdpa_flops = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
        double cls_flops = 2.0*VOCAB*DIM*SEQ;
        double total_flops = (fwd_flops*3 + sdpa_flops + cls_flops*3) * total_steps_done;
        double ane_flops = (fwd_flops*3 + sdpa_flops) * total_steps_done;
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (%.1f%%)\n", total_compile_ms, 100*total_compile_ms/wall);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg train:       %.1f ms/step\n", total_train_ms/total_steps_done);
        printf("ANE TFLOPS:      %.2f sustained\n", ane_flops / (total_train_ms * 1e9));
        printf("Total TFLOPS:    %.2f (ANE+CPU)\n", total_flops / (total_train_ms * 1e9));
        printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", 100*ane_flops/(total_train_ms*1e9)/15.8);

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            free_layer_kernels(&kern[L]);
            layer_weights_free(&lw[L]);
            layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]);
            layer_grads_free(&grads[L]);
        }
        free_kern(classifierKern); free_kern(softmaxKern);
        munmap(token_data, data_len);
        close(data_fd);
        free(rms_final); free(embed); free(grms_final); free(gembed); free(embed_seen);
        adam_free(&arms_final); adam_free(&aembed);
        free(dy); free(dffn); free(dx_ffn); free(dx2); free(dx_attn);
        free(dx_rms_final); free(dx_rms1);
        free(x_cur); free(x_final); free(probs); free(dlogits);
    }
    return 0;
}
