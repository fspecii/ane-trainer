// stories_cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, softmax
#pragma once
#include "stories_config.h"

// Pre-allocated scratch buffers — size = SEQ, allocated once on first use.
// All rmsnorm / rmsnorm_bwd calls are sequential (never concurrent).
static float *g_rms_tmp  = NULL;  // vDSP temp (elementwise products)
static float *g_rms_ss   = NULL;  // zero-init accumulator (ss / rrms² reuse)
static float *g_rms_rrms = NULL;  // reciprocal-sqrt output
static float *g_rms_dot  = NULL;  // zero-init dot accumulator
static float *g_ce_buf   = NULL;  // transposed logit scratch [SEQ * VOCAB]

#define ADAM_CHUNK 2048

static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    if (!g_rms_ss)  g_rms_ss  = (float*)malloc(S*4);
    memset(g_rms_ss, 0, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, g_rms_ss, 1, g_rms_ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(g_rms_ss, 1, &invd, &eps, g_rms_ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(g_rms_ss, g_rms_ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, g_rms_ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp)  g_rms_tmp  = (float*)malloc(S*4);
    if (!g_rms_ss)   g_rms_ss   = (float*)malloc(S*4);
    if (!g_rms_rrms) g_rms_rrms = (float*)malloc(S*4);
    if (!g_rms_dot)  g_rms_dot  = (float*)malloc(S*4);
    memset(g_rms_ss,  0, S*4);
    memset(g_rms_dot, 0, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, g_rms_ss, 1, g_rms_ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(g_rms_ss, 1, &invd, &eps, g_rms_ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(g_rms_rrms, g_rms_ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], g_rms_dot, 1, g_rms_dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(g_rms_rrms, 1, g_rms_rrms, 1, g_rms_ss, 1, (vDSP_Length)S);
    vDSP_vsmul(g_rms_ss, 1, &invd, g_rms_ss, 1, (vDSP_Length)S);
    vDSP_vmul(g_rms_dot, 1, g_rms_ss, 1, g_rms_dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, g_rms_dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, g_rms_rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, g_rms_rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
}

// Vectorized Adam using vDSP + vvrsqrtf (chunked to keep temps on stack).
// Approximates 1/(sqrt(v/bc2) + eps) as 1/sqrt(v/bc2 + eps) — negligible
// difference at eps=1e-8 for typical gradient magnitudes.
static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    float inv_bc1 = 1.0f/bc1, inv_bc2 = 1.0f/bc2, neg_lr = -lr;
    float b1c = 1.0f - b1, b2c = 1.0f - b2;
    static float mh[ADAM_CHUNK], vh[ADAM_CHUNK], g2[ADAM_CHUNK];
    for (size_t i = 0; i < s->n; i += ADAM_CHUNK) {
        int c = (s->n - i < ADAM_CHUNK) ? (int)(s->n - i) : ADAM_CHUNK;
        // m = b1*m + (1-b1)*g
        vDSP_vsmul(s->m+i, 1, &b1, mh, 1, c);
        vDSP_vsma(g+i, 1, &b1c, mh, 1, s->m+i, 1, c);
        // v = b2*v + (1-b2)*g^2
        vDSP_vmul(g+i, 1, g+i, 1, g2, 1, c);
        vDSP_vsmul(s->v+i, 1, &b2, vh, 1, c);
        vDSP_vsma(g2, 1, &b2c, vh, 1, s->v+i, 1, c);
        // mh = m/bc1
        vDSP_vsmul(s->m+i, 1, &inv_bc1, mh, 1, c);
        // vh = v/bc2 + eps
        vDSP_vsmsa(s->v+i, 1, &inv_bc2, &eps, vh, 1, c);
        // step = mh / sqrt(vh) via rsqrt
        int cc = c; vvrsqrtf(g2, vh, &cc);
        vDSP_vmul(mh, 1, g2, 1, mh, 1, c);
        // w -= lr * step
        vDSP_vsma(mh, 1, &neg_lr, w+i, 1, w+i, 1, c);
    }
}

// Cross-entropy loss + gradient. g_ce_buf is allocated once (SEQ*VOCAB floats = 32MB).
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    if (!g_ce_buf) g_ce_buf = (float*)malloc((size_t)S * V * 4);
    float *buf = g_ce_buf;
    // Transpose [V,S] → [S,V]: buf[t*V+v] = logits[v*S+t]
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;
        float maxv; vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);
        int n = V; vvexpf(row, row, &n);
        float sum; vDSP_sve(row, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);
        int tgt = targets[t];
        total_loss -= logf(row[tgt] + 1e-10f);
        row[tgt] -= 1.0f;
        vDSP_vsmul(row, 1, &invS, row, 1, (vDSP_Length)V);
    }
    // Transpose back [S,V] → [V,S]
    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    return total_loss / S;
}

// Embedding lookup: token_ids → x [DIM, SEQ] (channel-first)
// embed is [VOCAB, DIM] row-major (vocab_size rows, dim cols)
static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++) {
            x[d*seq + t] = embed[tok*dim + d];
        }
    }
}

// Embedding backward: accumulate dE[tok] += dx[:,t] for each position
static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++) {
            d_embed[tok*dim + d] += dx[d*seq + t];
        }
    }
}
