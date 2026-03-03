// stories_io.h — IOSurface helpers, blob builders, NEON conversion
#pragma once
#include "stories_config.h"
#include <arm_neon.h>

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_t(const float *w, int rows, int cols) {
    int ws=cols*rows*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) fp16[j*rows+i]=(_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// IOSurface I/O (channel-first [C,S] layout)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}

// Write weight data (f32 → fp16) directly to a weight IOSurface
static void io_write_wf16(IOSurfaceRef s, const float *data, int n) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, n);
    IOSurfaceUnlock(s, 0, NULL);
}

// Kernel compile/eval — static weights (BLOBFILE), single activation input
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    NSString *cacheBase  = [[@"~/.ane_cache" stringByExpandingTildeInPath] stringByAppendingPathComponent:hx];
    NSString *cachedData = [cacheBase stringByAppendingPathComponent:@"data"];
    NSString *cachedPlist= [cacheBase stringByAppendingPathComponent:@"net.plist"];
    BOOL loaded = NO;
    if ([fm fileExistsAtPath:cachedData]) {
        [fm copyItemAtPath:cachedData  toPath:[td stringByAppendingPathComponent:@"data"]     error:nil];
        [fm copyItemAtPath:cachedPlist toPath:[td stringByAppendingPathComponent:@"net.plist"] error:nil];
        loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    }
    if (!loaded) {
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
        }
        [fm createDirectoryAtPath:cacheBase withIntermediateDirectories:YES attributes:nil error:nil];
        [fm copyItemAtPath:[td stringByAppendingPathComponent:@"data"]     toPath:cachedData  error:nil];
        [fm copyItemAtPath:[td stringByAppendingPathComponent:@"net.plist"] toPath:cachedPlist error:nil];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("  [compile] load FAIL\n"); return NULL;
        }
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    k->wIns = NULL; k->nWIns = 0;
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
    }
}
static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    for (int i = 0; i < k->nWIns; i++) CFRelease(k->wIns[i]);
    free(k->wIns);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}
// Compile a kernel with dynamic weight inputs (no BLOBFILE for weights).
// staticWeights: optional dict for genuinely static data (e.g. causal mask), same format as compile_kern_mil_w.
// w_bytes: byte size of each weight IOSurface (fp16 elements × 2).
// nW: number of weight inputs.
// Request input order: [ioIn(0), wIns[0](1), wIns[1](2), ...]
static Kern *compile_kern_dyn(NSString *mil, NSDictionary *staticWeights,
                               int act_bytes, int *w_bytes, int nW, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
                                                          md, staticWeights, nil);
    if (!desc) { printf("  [compile_dyn] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (staticWeights) {
        for (NSString *path in staticWeights) {
            NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
            [staticWeights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
        }
    }
    NSError *e = nil;
    NSString *cacheBase  = [[@"~/.ane_cache" stringByExpandingTildeInPath] stringByAppendingPathComponent:hx];
    NSString *cachedData = [cacheBase stringByAppendingPathComponent:@"data"];
    NSString *cachedPlist= [cacheBase stringByAppendingPathComponent:@"net.plist"];
    BOOL loaded = NO;
    if ([fm fileExistsAtPath:cachedData]) {
        [fm copyItemAtPath:cachedData  toPath:[td stringByAppendingPathComponent:@"data"]     error:nil];
        [fm copyItemAtPath:cachedPlist toPath:[td stringByAppendingPathComponent:@"net.plist"] error:nil];
        loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    }
    if (!loaded) {
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("  [compile_dyn] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
        }
        [fm createDirectoryAtPath:cacheBase withIntermediateDirectories:YES attributes:nil error:nil];
        [fm copyItemAtPath:[td stringByAppendingPathComponent:@"data"]     toPath:cachedData  error:nil];
        [fm copyItemAtPath:[td stringByAppendingPathComponent:@"net.plist"] toPath:cachedPlist error:nil];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("  [compile_dyn] load FAIL\n"); return NULL;
        }
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn  = make_surface(act_bytes);
    k->ioOut = make_surface(oc_bytes);
    k->nWIns = nW;
    k->wIns  = nW > 0 ? (IOSurfaceRef*)malloc(nW * sizeof(IOSurfaceRef)) : NULL;
    for (int i = 0; i < nW; i++) k->wIns[i] = make_surface(w_bytes[i]);

    // Build input arrays: [ioIn, wIns[0], wIns[1], ...]
    NSMutableArray *ins  = [NSMutableArray arrayWithCapacity:1 + nW];
    NSMutableArray *idxs = [NSMutableArray arrayWithCapacity:1 + nW];
    [ins  addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn)];
    [idxs addObject:@0];
    for (int i = 0; i < nW; i++) {
        [ins  addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->wIns[i])];
        [idxs addObject:@(i + 1)];
    }
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        ins, idxs, @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
    }
}

static void ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}

// beginRealTimeTask / endRealTimeTask — wrap a full training step to reduce ANE scheduling jitter.
// Tested: p99 latency drops 90%+ (35ms → 3ms for a single kernel).
// Call ane_step_client_init() once after first kernel is compiled; then ane_step_begin/end per step.
static id g_ane_rt_client = nil;

static void ane_step_client_init(Kern *k) {
    if (g_ane_rt_client) return;
    id mdl = (__bridge id)k->model;
    // _ANEInMemoryModel holds _sharedConnection which is the _ANEClient
    Ivar iv = class_getInstanceVariable([mdl class], "_sharedConnection");
    if (!iv) return;
    id conn = object_getIvar(mdl, iv);
    if (!conn) return;
    // Verify it actually has the real-time selectors before caching
    if ([conn respondsToSelector:@selector(beginRealTimeTask)] &&
        [conn respondsToSelector:@selector(endRealTimeTask)]) {
        g_ane_rt_client = conn;
        printf("  [rt] beginRealTimeTask available on %s\n",
               [NSStringFromClass([conn class]) UTF8String]);
    }
}

static void ane_step_begin(void) {
    if (!g_ane_rt_client) return;
    ((void(*)(id,SEL))objc_msgSend)(g_ane_rt_client, @selector(beginRealTimeTask));
}

static void ane_step_end(void) {
    if (!g_ane_rt_client) return;
    ((void(*)(id,SEL))objc_msgSend)(g_ane_rt_client, @selector(endRealTimeTask));
}
