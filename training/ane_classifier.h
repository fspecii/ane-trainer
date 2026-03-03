// ane_classifier.h — Classifier forward (embed@x_final) and softmax on ANE.
//
// gen_classifier_fwd_dyn(): Dynamic embed weight input — compile ONCE, write embed
//   IOSurface once per Adam step. Avoids per-batch recompile.
//   Proven: ANE accepts VOCAB=32000 output channels in conv.
//
// gen_softmax_vocab(): Softmax over VOCAB axis — no weights, compile once.
//   Combined with io_copy from classifier output, skips CPU logit round-trip.
#pragma once
#include "stories_mil.h"

// Classifier forward: x_final [1,DIM,1,SEQ] × embed^T → logits [1,VOCAB,1,SEQ]
// Uses matmul (3D) — proven with dynamic inputs in our backward kernels.
// Upstream uses 32000-channel conv with BLOBFILE (compiled-in), but ANE rejects
// conv with VOCAB=32000 when the weight is a dynamic function parameter.
//
// Param 0 = x    (ioIn,    DIM*SEQ*2 bytes fp16) — declared as [1,DIM,1,SEQ], reshaped internally
// Param 1 = We   (wIns[0], VOCAB*DIM*2 bytes fp16, [1,VOCAB,DIM]) — updated per Adam step
// Output  = logits [1,VOCAB,1,SEQ] fp16 (VOCAB*SEQ*2 bytes)
static NSString *gen_classifier_fwd_dyn(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>("
                    @"tensor<fp16, [1, %d, 1, %d]> x, "
                    @"tensor<fp16, [1, %d, %d]> We) {\n",
                    DIM, SEQ, VOCAB, DIM];
    // Reshape x: [1, DIM, 1, SEQ] → [1, DIM, SEQ] for 3D matmul
    [m appendFormat:@"        tensor<int32, [3]> sh3 = const()[name=string(\"sh3\"), "
        "val=tensor<int32, [3]>([1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> x3 = reshape(shape=sh3,x=x)"
        "[name=string(\"rx\")];\n", DIM, SEQ];
    // matmul: We[1,VOCAB,DIM] @ x3[1,DIM,SEQ] → mm[1,VOCAB,SEQ]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d]> mm = matmul(transpose_x=bF,"
        "transpose_y=bF,x=We,y=x3)[name=string(\"mm\")];\n", VOCAB, SEQ];
    // Reshape mm: [1, VOCAB, SEQ] → [1, VOCAB, 1, SEQ] for softmax compatibility
    [m appendFormat:@"        tensor<int32, [4]> sh4 = const()[name=string(\"sh4\"), "
        "val=tensor<int32, [4]>([1,%d,1,%d])];\n", VOCAB, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = reshape(shape=sh4,x=mm)"
        "[name=string(\"out\")];\n", VOCAB, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Softmax over VOCAB (axis=1): logits [1,VOCAB,1,SEQ] → probs [1,VOCAB,1,SEQ]
// No weights — compile once at startup and reuse.
// Use io_copy from classifierKern->ioOut to avoid CPU round-trip between classifier and softmax.
static NSString *gen_softmax_vocab(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", VOCAB, SEQ];
    [m appendString:@"        int32 ax = const()[name=string(\"ax\"), val=int32(1)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> out = softmax(axis=ax,x=x)[name=string(\"sm\")];\n",
        VOCAB, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}
