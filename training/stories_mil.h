// stories_mil.h — MIL program generators for ANE kernels
// Dynamic-weight version: all trainable weights are function inputs (no BLOBFILE for weights).
// Compile once at startup; update weights by writing to IOSurfaces before each eval.
// Input conventions (matching compile_kern_dyn in stories_io.h):
//   param 0 = activation x (ioIn)
//   param 1+ = weights in declaration order (wIns[0], wIns[1], ...)
//
// Forward kernels use conv([OUT,IN,1,1], x[1,IN,1,S]) — native ANE op, no reshapes.
// Backward kernels use matmul with 3D weight params [1,OUT,IN] — transpose_x=true for W^T.
#pragma once
#include "stories_io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// SDPA forward + taps: x_in → rmsnorm → QKV+SDPA+Wo → concat(o_out, Q, K, V, attn_out, xnorm)
// Params: x [1,DIM,1,SEQ], rms1 [1,DIM,1,1], Wq/Wk/Wv/Wo [DIM,DIM,1,1] (conv kernel format)
// wIns[0]=rms1, [1]=Wq, [2]=Wk, [3]=Wv, [4]=Wo
static NSString *gen_sdpa_fwd_taps(void) {
    float sc = 1.0f/sqrtf((float)HD);
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>("
                    @"tensor<fp16, [1, %d, 1, %d]> x, "
                    @"tensor<fp16, [1, %d, 1, 1]> rms1, "
                    @"tensor<fp16, [%d, %d, 1, 1]> Wq, "
                    @"tensor<fp16, [%d, %d, 1, 1]> Wk, "
                    @"tensor<fp16, [%d, %d, 1, 1]> Wv, "
                    @"tensor<fp16, [%d, %d, 1, 1]> Wo) {\n",
                    DIM, SEQ, DIM, DIM, DIM, DIM, DIM, DIM, DIM, DIM, DIM];

    // RMSNorm: x [1,DIM,1,SEQ] → xn [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendString:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendString:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rms1)[name=string(\"xn\")];\n", DIM, SEQ];

    // Conv constants (1x1 conv = linear projection, native ANE op)
    [m appendString:@CONV_CONST];

    // Q, K, V projections: conv([DIM,DIM,1,1], xn[1,DIM,1,SEQ]) → [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", DIM, SEQ];

    // Reshape Q,K,V to multi-head format and transpose
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS, SEQ, HD];

    // SDPA: Q @ K^T scaled + causal mask → softmax → @ V
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];
    // Mask stays BLOBFILE — static, never updated
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS, HD, SEQ];

    // Reshape at [1,HEADS,HD,SEQ] → [1,DIM,1,SEQ] for Wo projection
    [m appendFormat:@"        tensor<int32, [4]> af4s = const()[name=string(\"af4s\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=af4s,x=at)[name=string(\"af\")];\n", DIM, SEQ];

    // Wo projection: conv([DIM,DIM,1,1], af[1,DIM,1,SEQ]) → oo [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", DIM, SEQ];

    // concat(oo, qf, kf, vf, af, xn) — taps for backward pass
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))[name=string(\"cat\")];\n", 6*DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN forward + taps: x2 → rmsnorm → FFN → concat(ffn_out, h1, h3, silu_out, x2norm)
// Params: x [1,DIM,1,SEQ], rms2 [1,DIM,1,1], W1/W3 [HIDDEN,DIM,1,1], W2 [DIM,HIDDEN,1,1]
// wIns[0]=rms2, [1]=W1, [2]=W3, [3]=W2
static NSString *gen_ffn_fwd_taps(void) {
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>("
                    @"tensor<fp16, [1, %d, 1, %d]> x, "
                    @"tensor<fp16, [1, %d, 1, 1]> rms2, "
                    @"tensor<fp16, [%d, %d, 1, 1]> W1, "
                    @"tensor<fp16, [%d, %d, 1, 1]> W3, "
                    @"tensor<fp16, [%d, %d, 1, 1]> W2) {\n",
                    DIM, SEQ, DIM, HIDDEN, DIM, HIDDEN, DIM, DIM, HIDDEN];

    // RMSNorm: x [1,DIM,1,SEQ] → xn [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendString:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendString:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rms2)[name=string(\"xn\")];\n", DIM, SEQ];

    // Conv constants
    [m appendString:@CONV_CONST];

    // W1, W3 projections: conv([HIDDEN,DIM,1,1], xn[1,DIM,1,SEQ]) → [1,HIDDEN,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", HIDDEN, SEQ];

    // SiLU(h1) * h3 (gate)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN, SEQ];

    // W2 projection: conv([DIM,HIDDEN,1,1], gate[1,HIDDEN,1,SEQ]) → y [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];\n", DIM, SEQ];

    // concat(y, h1, h3, gate, xn) — taps for backward pass
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))[name=string(\"cat\")];\n", 2*DIM+3*HIDDEN, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN backward: concat(dffn,h1,h3) → concat(dx,dh1,dh3)
// Params: x [1,DIM+2*HIDDEN,1,SEQ], W2 [1,DIM,HIDDEN], W1 [1,HIDDEN,DIM], W3 [1,HIDDEN,DIM]
// Backward uses transpose_x=true to compute W^T @ d without storing transposed weights.
// wIns[0]=W2, [1]=W1, [2]=W3
static NSString *gen_ffn_bwd(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>("
                    @"tensor<fp16, [1, %d, 1, %d]> x, "
                    @"tensor<fp16, [1, %d, %d]> W2, "
                    @"tensor<fp16, [1, %d, %d]> W1, "
                    @"tensor<fp16, [1, %d, %d]> W3) {\n",
                    DIM+2*HIDDEN, SEQ, DIM, HIDDEN, HIDDEN, DIM, HIDDEN, DIM];

    // Slice x into (dffn, h1, h3)
    [m appendString:@"        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];\n", HIDDEN, SEQ];

    // dsilu = W2^T @ dffn
    [m appendString:@"        bool wF = const()[name=string(\"wF\"), val=bool(false)];\n"];
    [m appendString:@"        bool wT = const()[name=string(\"wT\"), val=bool(true)];\n"];
    // Reshape activation slice to 3D for matmul (weights are already 3D params)
    [m appendFormat:@"        tensor<int32, [3]> ds = const()[name=string(\"ds\"), val=tensor<int32, [3]>([1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dffn3 = reshape(shape=ds,x=dffn)[name=string(\"dffn3\")];\n", DIM, SEQ];
    // W2^T @ dffn: W2 [1,DIM,HIDDEN], transpose_x=true → [1,HIDDEN,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dsilu3 = matmul(transpose_x=wT,transpose_y=wF,x=W2,y=dffn3)[name=string(\"dsilu3\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> hs = const()[name=string(\"hs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = reshape(shape=hs,x=dsilu3)[name=string(\"dsilu\")];\n", HIDDEN, SEQ];

    // SiLU backward: dh1, dh3
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendString:@"        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=string(\"oms\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];

    // dx1 = W1^T @ dh1: W1 [1,HIDDEN,DIM], transpose_x=true → W1^T [1,DIM,HIDDEN] @ dh1 [1,HIDDEN,SEQ]
    [m appendFormat:@"        tensor<int32, [3]> h3s = const()[name=string(\"h3s\"), val=tensor<int32, [3]>([1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dh13 = reshape(shape=h3s,x=dh1)[name=string(\"dh13\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dh33 = reshape(shape=h3s,x=dh3)[name=string(\"dh33\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [3]> dx3s = const()[name=string(\"dx3s\"), val=tensor<int32, [3]>([1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dx1_3 = matmul(transpose_x=wT,transpose_y=wF,x=W1,y=dh13)[name=string(\"dx1_3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dx3_3 = matmul(transpose_x=wT,transpose_y=wF,x=W3,y=dh33)[name=string(\"dx3_3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dxs = const()[name=string(\"dxs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = reshape(shape=dxs,x=dx1_3)[name=string(\"dx1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = reshape(shape=dxs,x=dx3_3)[name=string(\"dx3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];\n", DIM, SEQ];

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"cat\")];\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 1 + Wo backward: concat(qf,kf,vf,dx2f) → concat(dv, probs, dp)
// Params: x [1,4*DIM,1,SEQ], Wo [1,DIM,DIM]
// Wo^T used via transpose_x=true — no separate Wot needed.
// wIns[0]=Wo
static NSString *gen_sdpa_bwd1(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>("
                    @"tensor<fp16, [1, %d, 1, %d]> x, "
                    @"tensor<fp16, [1, %d, %d]> Wo) {\n",
                    4*DIM, SEQ, DIM, DIM];

    // Slice x into (qf, kf, vf, dx2f)
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx2f = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];\n", DIM, SEQ];

    // df = Wo^T @ dx2f: Wo [1,DIM,DIM], transpose_x=true → [1,DIM,SEQ]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<int32, [3]> ds = const()[name=string(\"ds\"), val=tensor<int32, [3]>([1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dx2f3 = reshape(shape=ds,x=dx2f)[name=string(\"dx2f3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> df3 = matmul(transpose_x=bT,transpose_y=bF,x=Wo,y=dx2f3)[name=string(\"df3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dfs = const()[name=string(\"dfs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> df = reshape(shape=dfs,x=df3)[name=string(\"df\")];\n", DIM, SEQ];

    // Reshape Q,K,V,df to multi-head format
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=df)[name=string(\"rd\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> da = transpose(perm=pm,x=dr)[name=string(\"td\")];\n", HEADS, SEQ, HD];

    // Re-compute SDPA scores (needed for backward)
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];
    // Mask stays BLOBFILE
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // dV and dP
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da)[name=string(\"dv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n", SCORE_CH, SEQ];

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];\n", DIM+2*SCORE_CH, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2: concat(probs,dp,Q,K) → concat(dQ,dK)
// No weights — purely computational. Unchanged from original.
static NSString *gen_sdpa_bwd2(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n", HEADS, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];\n", DIM, SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];\n", 2*DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// QKV backward: concat(dq,dk,dv) → dx
// Params: x [1,3*DIM,1,SEQ], Wq [1,DIM,DIM], Wk [1,DIM,DIM], Wv [1,DIM,DIM]
// Wq^T/Wk^T/Wv^T via transpose_x=true — no separate transposed weight storage.
// wIns[0]=Wq, [1]=Wk, [2]=Wv
static NSString *gen_qkvb(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>("
                    @"tensor<fp16, [1, %d, 1, %d]> x, "
                    @"tensor<fp16, [1, %d, %d]> Wq, "
                    @"tensor<fp16, [1, %d, %d]> Wk, "
                    @"tensor<fp16, [1, %d, %d]> Wv) {\n",
                    3*DIM, SEQ, DIM, DIM, DIM, DIM, DIM, DIM];

    // Slice x into (dq, dk, dv)
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM, SEQ];

    [m appendString:@"        bool wF = const()[name=string(\"wF\"), val=bool(false)];\n"];
    [m appendString:@"        bool wT = const()[name=string(\"wT\"), val=bool(true)];\n"];
    // Reshape activation slices to 3D for matmul (weights are already 3D params)
    [m appendFormat:@"        tensor<int32, [3]> ds = const()[name=string(\"ds\"), val=tensor<int32, [3]>([1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dq3 = reshape(shape=ds,x=dq)[name=string(\"dq3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dk3 = reshape(shape=ds,x=dk)[name=string(\"dk3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dv3 = reshape(shape=ds,x=dv)[name=string(\"dv3\")];\n", DIM, SEQ];
    // Wq^T @ dq, Wk^T @ dk, Wv^T @ dv (transpose_x=true, weights 3D [1,DIM,DIM])
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dxq3 = matmul(transpose_x=wT,transpose_y=wF,x=Wq,y=dq3)[name=string(\"dxq3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dxk3 = matmul(transpose_x=wT,transpose_y=wF,x=Wk,y=dk3)[name=string(\"dxk3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dxv3 = matmul(transpose_x=wT,transpose_y=wF,x=Wv,y=dv3)[name=string(\"dxv3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> dxqk3 = add(x=dxq3,y=dxk3)[name=string(\"aqk\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> out3 = add(x=dxqk3,y=dxv3)[name=string(\"out3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> outs = const()[name=string(\"outs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = reshape(shape=outs,x=out3)[name=string(\"out\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Mask blob (causal mask [SEQ,SEQ]) — static, used as BLOBFILE in sdpa kernels
static NSData *g_mask_blob = nil;
static NSData *get_mask_blob(void) {
    if (!g_mask_blob) {
        _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
            mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, SEQ*SEQ);
        free(mask);
    }
    return g_mask_blob;
}
