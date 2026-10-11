// Reasoning stop mask (slot 850): rows inside a reply's open reasoning block
// may not draw the model's own stop tokens (the engine's
// sampler/reasoning.rs - SGLang's merged rule, a stop does not finish a
// reply whose reasoning phase is open). Launched on the logits rows right
// before the sampling kernels, so the draw that follows - greedy,
// categorical, truncated, any mode - is the masked row's: the masking
// happens before truncation, exactly as a host logits processor would.
//
// Per row, `par` = 8 u32 words:
//   {active, inv_t f32 bits (0 = greedy), n_stops (<= 4), stop0..stop3, pad}
// An inactive row is untouched and gets flag 0. An active row is read once
// for its max, `flags[row]` records stop PRESSURE - a stop within ln 2 of the
// max after temperature (p_stop >= p_top / 2), or AT the max for a greedy
// row - and the stop logits are then set to -inf. The engine counts pressure
// rows to force the block's exit (a model that keeps asking to finish gets
// to, instead of thinking on to max_tokens); a lone spike never forces.
//
// One block per row; the max is the argmax kernel's float4 sweep. The stop
// ids are read after the sweep, from the unmasked row, by thread 0.

#define PD_SMASK_THREADS 256

__global__ void __launch_bounds__(PD_SMASK_THREADS)
pd_stop_mask_rows_kernel(float* __restrict__ logits, const uint32_t* __restrict__ par,
                         uint32_t* __restrict__ flags, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint32_t* p = par + (size_t)row * 8;
    if (p[0] == 0) {
        if (tid == 0) flags[row] = 0;
        return;
    }
    float* x = logits + (size_t)row * n;
    float bv = -INFINITY;
    const uint32_t head =
        min(n, (uint32_t)(((16u - (reinterpret_cast<uintptr_t>(x) & 15u)) & 15u) >> 2));
    if (tid < head) bv = fmaxf(bv, x[tid]);
    const float4* x4 = reinterpret_cast<const float4*>(x + head);
    const uint32_t n4 = (n - head) >> 2;
    for (uint32_t b = tid; b < n4; b += PD_SMASK_THREADS) {
        const float4 r = __ldg(x4 + b);
        bv = fmaxf(fmaxf(bv, r.x), fmaxf(r.y, fmaxf(r.z, r.w)));
    }
    for (uint32_t i = head + (n4 << 2) + tid; i < n; i += PD_SMASK_THREADS) bv = fmaxf(bv, x[i]);
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) bv = fmaxf(bv, __shfl_down_sync(0xffffffffu, bv, off));
    __shared__ float sv[PD_SMASK_THREADS / 32];
    if ((tid & 31u) == 0) sv[tid >> 5] = bv;
    __syncthreads();
    if (tid != 0) return;
    float m = sv[0];
#pragma unroll
    for (uint32_t w = 1; w < PD_SMASK_THREADS / 32; ++w) m = fmaxf(m, sv[w]);
    const float inv_t = __uint_as_float(p[1]);
    const uint32_t ns = min(p[2], 4u);
    uint32_t pressure = 0;
    for (uint32_t i = 0; i < ns; ++i) {
        const uint32_t id = p[3 + i];
        if (id >= n) continue;
        const float l = x[id];
        // ln 2: a stop at least half as likely as the top token
        if (inv_t > 0.0f ? (l - m) * inv_t >= -0.69314718f : l >= m) pressure = 1;
    }
    for (uint32_t i = 0; i < ns; ++i) {
        const uint32_t id = p[3 + i];
        if (id < n) x[id] = -INFINITY;
    }
    flags[row] = pressure;
}

PD_EXPORT
int pd_stop_mask_rows(void* logits, const void* par, void* flags, uint32_t rows, uint32_t n,
                      void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_stop_mask_rows_kernel<<<rows, PD_SMASK_THREADS, 0, (cudaStream_t)stream>>>(
        (float*)logits, (const uint32_t*)par, (uint32_t*)flags, n);
    return pd_launch_status();
}
