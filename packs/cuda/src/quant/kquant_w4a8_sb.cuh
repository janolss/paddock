// quant/kquant_w4a8_sb.cuh - the k-quant W4A8 SMALL-BATCH decode GEMM
// (slot 852, pd_kquant_gemm_w4a8_sb): b <= 8 activation columns against a
// weight plane streamed once, at the b=1 GEMV's bandwidth class.
//
// The decode band of a k-quant model at b = 2..8 (one spec verify is
// slots x (k+1) rows) ran the dp4a z-tile or the 64-row K-split mma. On
// GB202 (bench/kq_dec_gb202_bench.cu, Qwen3.8-27B's Q5_K planes, b=8) they
// hold 33-57% and 56-68% of the 1.63 TB/s read roof while the b=1 GEMV holds
// 71-97%: dp4a parks every 512-element chunk on a synchronous staging load
// (ncu long-scoreboard 10.6 an issue at 3 CTAs/SM), and the K-split's
// 2-stage ring at 2 CTAs/SM spends ~65 instructions a mma on per-word index
// math, per-block scale re-extraction and divide-per-copy staging between
// two barriers a 256-K step. The pure stream of the same bytes in the same
// strided order runs at 84% (the launch's own ceiling at these plane sizes),
// so the loss is structure, not access pattern.
//
// This rung is the GEMV's shape on the tensor core: a CTA owns 16 output
// rows (one m16 fragment); its 8 warps split K round-robin by super-block
// (warp w walks supers w, w+8, ...), each warp loading EVERYTHING a super
// needs - its raw strip words straight into the mma A-fragment layout, the
// 24 B scale records, the activation B fragments and the per-column scales
// and sums - one super ahead in registers (189-250 of them, one CTA an SM:
// the 2-CTA cut at 128 registers spilled and ran at 40% of roof, and 8
// warps with two supers each in flight already cover the DRAM latency), so
// nothing waits inside the fold: no shared staging, no barrier until the 8
// partial tiles meet once in shared memory and sum in fixed warp order.
// For Q4_K / Q5_K / Q4_0 the k order inside each 32-block is permuted so a
// lane's A words are 8 CONTIGUOUS raw bytes (one 64-bit load = two blocks'
// nibbles) and B is loaded in the same order (one 64-bit load a block);
// an exact integer dot does not care which lane holds which k. nz > 1
// writes the K-split's partial planes and reuses its combine; b > 8 and
// the i-quant family go to the K-split.
// GB202, cold planes, us a launch, ks -> sb: qkv [5120 -> 10240] b=4
// 35.5 -> 28.2, b=8 35.6 -> 30.4; up [5120 -> 17408] b=4 56.8 -> 42.7, b=8
// 61.2 -> 44.8; down [17408 -> 5120] b=4 62.0 -> 44.7, b=8 69.8 -> 49.1;
// IQ4_XS gate b=8 47.8 -> 40.7 - 76-97% of the read roof.
// Numerics: the decode class (exact int8 dots, f32 per-block scale
// application, the ks rung's fold expressions); the K order differs from
// ks (8 interleaved slices, then a fixed-order sum), so results are
// reorder-equal to ks / dp4a (bit-identical on the harness data) and, per
// row, fixed for a given (plane, nz) and independent of the batch width.

template <uint32_t DT, uint32_t CS, uint32_t MINB>
__global__ void __launch_bounds__(256, MINB) pd_kquant_sb_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool K16 = (DT == PD_KQ_Q6K);
    constexpr bool Q5 = (DT == PD_KQ_Q5K);
    constexpr uint32_t DATAB = DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    // raw words a lane holds per row per super: Q4K 8 (4 groups x 2 k-halves),
    // Q5K 8 + 2 qh, Q6K 8 ql + 4 qh, IQ4XS 8
    constexpr uint32_t NQ = 8u;
    constexpr uint32_t NH = Q5 ? 2u : (K16 ? 4u : 0u);
    constexpr uint32_t C4 = Q5 ? 0x10101010u : 0x08080808u;
    // PERM: Q4_K / Q5_K / Q4_0 lanes take 8 CONTIGUOUS raw bytes a group (one
    // 64-bit load for two blocks' nibbles) and the k order inside each
    // 32-block is permuted to match - lane t holds k 8t..8t+7 instead of
    // 4t..4t+3 + 16+4t..; the activation fragment is loaded in the same
    // order (one 64-bit load a block), and an exact integer dot does not
    // care which lane holds which k. Q6_K's per-16 scales and IQ4_XS's
    // lo/hi-nibble halves keep the standard map.
    constexpr bool PERM = !K16 && DT != PD_KQ_IQ4XS;
    // 8 warps = CS column fragments x (8 / CS) round-robin K slices
    constexpr uint32_t NSL = 8u / CS;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t wsub = warp % CS, wsl = warp / CS;
    const uint32_t row_base = blockIdx.x * 16u;
    const uint32_t n_super = in_dim >> 8u, nb32 = in_dim >> 5u, nb16 = in_dim >> 4u;
    uint32_t s_lo = 0u, s_hi = n_super;
    if (gridDim.z > 1u) {
        const uint32_t per = (n_super + gridDim.z - 1u) / gridDim.z;
        s_lo = blockIdx.z * per;
        s_hi = s_lo + per < n_super ? s_lo + per : n_super;
        y += (size_t)blockIdx.z * out_dim * batch;
    }
    const uint32_t r0 = row_base + g, r8 = r0 + 8u;
    const bool ok0 = r0 < out_dim, ok8 = r8 < out_dim;
    const uint8_t* const w0 = data + (size_t)(ok0 ? r0 : 0u) * n_super * DATAB + 4u * t;
    const uint8_t* const w8 = data + (size_t)(ok8 ? r8 : 0u) * n_super * DATAB + 4u * t;
    const uint8_t* const e0 = scales + (size_t)(ok0 ? r0 : 0u) * n_super * PD_KQ_SCB;
    const uint8_t* const e8 = scales + (size_t)(ok8 ? r8 : 0u) * n_super * PD_KQ_SCB;
    // activation columns this warp's B fragment and C columns touch
    const uint32_t bcol = wsub * 8u + g, ccol0 = wsub * 8u + 2u * t;

    struct Raw {
        uint32_t q0[NQ], q8[NQ], h0[NH > 0 ? NH : 1], h8[NH > 0 ? NH : 1];
        uint32_t rec0[6], rec8[6];
        int b[8][2];          // B fragment words, 8 blocks
        float xc[2][8];       // per-32 activation scales, the lane's 2 C columns
        float su[2][8];       // per-32 sums (two per-16 added), MU only
    };
    auto load = [&](uint32_t kt, Raw& r) {
        {
            const bool okc = bcol < batch;
            const int8_t* pb = xq + (size_t)(okc ? bcol : 0u) * in_dim + kt * 256u + 4u * t;
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j) {
                if (PERM) {
                    const uint2 bv = okc ? __ldg((const uint2*)(pb + j * 32u + 4u * t)) : make_uint2(0u, 0u);
                    r.b[j][0] = (int)bv.x; r.b[j][1] = (int)bv.y;
                } else {
                    r.b[j][0] = okc ? __ldg((const int*)(pb + j * 32u)) : 0;
                    r.b[j][1] = okc ? __ldg((const int*)(pb + j * 32u + 16u)) : 0;
                }
            }
        }
        #pragma unroll
        for (uint32_t c = 0; c < 2u; ++c) {
            const uint32_t col = ccol0 + c;
            const bool okc = col < batch;
            const float4* px = (const float4*)(xs + (size_t)(okc ? col : 0u) * nb32 + kt * 8u);
            const float4 x0 = okc ? __ldg(px) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 x1 = okc ? __ldg(px + 1) : make_float4(0.f, 0.f, 0.f, 0.f);
            r.xc[c][0] = x0.x; r.xc[c][1] = x0.y; r.xc[c][2] = x0.z; r.xc[c][3] = x0.w;
            r.xc[c][4] = x1.x; r.xc[c][5] = x1.y; r.xc[c][6] = x1.z; r.xc[c][7] = x1.w;
            if (MU) {
                const float4* ps = (const float4*)(xsums + (size_t)(okc ? col : 0u) * nb16 + kt * 16u);
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    const float4 sv = okc ? __ldg(ps + q) : make_float4(0.f, 0.f, 0.f, 0.f);
                    r.su[c][2u * q] = sv.x + sv.y;
                    r.su[c][2u * q + 1u] = sv.z + sv.w;
                }
            }
        }
        const uint8_t* p0 = w0 + (size_t)kt * DATAB;
        const uint8_t* p8 = w8 + (size_t)kt * DATAB;
        if (PERM) {
            // group gq: 8 B at gq*32 + 8t -> q[2gq] = k 8t..8t+3, q[2gq+1] = k 8t+4..8t+7
            // (lo nibbles block 2gq, hi block 2gq+1); the base pointers carry +4t,
            // so +4t more lands on 8t
            #pragma unroll
            for (uint32_t gq = 0; gq < 4u; ++gq) {
                const uint2 a = ok0 ? __ldcs((const uint2*)(p0 + gq * 32u + 4u * t)) : make_uint2(0u, 0u);
                const uint2 b = ok8 ? __ldcs((const uint2*)(p8 + gq * 32u + 4u * t)) : make_uint2(0u, 0u);
                r.q0[2u * gq] = a.x; r.q0[2u * gq + 1u] = a.y;
                r.q8[2u * gq] = b.x; r.q8[2u * gq + 1u] = b.y;
            }
            if (Q5) {
                const uint2 a = ok0 ? __ldcs((const uint2*)(p0 + 128u + 4u * t)) : make_uint2(0u, 0u);
                const uint2 b = ok8 ? __ldcs((const uint2*)(p8 + 128u + 4u * t)) : make_uint2(0u, 0u);
                r.h0[0] = a.x; r.h0[1] = a.y;
                r.h8[0] = b.x; r.h8[1] = b.y;
            }
        } else {
            #pragma unroll
            for (uint32_t i = 0; i < NQ; ++i) {
                // Q6K: ql chunk (n6, a, kh) = i at n6*64 + a*32 + kh*16; IQ4XS: block i at i*16
                const uint32_t off = (DT == PD_KQ_IQ4XS) ? i * 16u : (i >> 1u) * 32u + (i & 1u) * 16u;
                r.q0[i] = ok0 ? __ldcs((const uint32_t*)(p0 + off)) : 0u;
                r.q8[i] = ok8 ? __ldcs((const uint32_t*)(p8 + off)) : 0u;
            }
            #pragma unroll
            for (uint32_t i = 0; i < NH; ++i) {
                // Q6K: qh (n6, kh) = i at 128 + n6*32 + kh*16
                const uint32_t off = 128u + (i >> 1u) * 32u + (i & 1u) * 16u;
                r.h0[i] = ok0 ? __ldcs((const uint32_t*)(p0 + off)) : 0u;
                r.h8[i] = ok8 ? __ldcs((const uint32_t*)(p8 + off)) : 0u;
            }
        }
        const uint2* q0 = (const uint2*)(e0 + (size_t)kt * PD_KQ_SCB);
        const uint2* q8 = (const uint2*)(e8 + (size_t)kt * PD_KQ_SCB);
        #pragma unroll
        for (uint32_t i = 0; i < 3u; ++i) {
            const uint2 a = ok0 ? __ldg(q0 + i) : make_uint2(0u, 0u);
            const uint2 b = ok8 ? __ldg(q8 + i) : make_uint2(0u, 0u);
            r.rec0[2u * i] = a.x; r.rec0[2u * i + 1u] = a.y;
            r.rec8[2u * i] = b.x; r.rec8[2u * i + 1u] = b.y;
        }
    };
    // A fragment word for block j (0..7 of the super), k-half kh, from a row's raw words
    auto frag = [&](const uint32_t* q, const uint32_t* h, uint32_t j, uint32_t kh) -> int {
        if (K16) {
            const uint32_t n6 = j >> 2u, r6 = j & 3u;
            const uint32_t qw = q[n6 * 4u + (r6 & 1u) * 2u + kh];
            const uint32_t hw = h[n6 * 2u + kh];
            const uint32_t nib = (r6 < 2u) ? (qw & 0x0F0F0F0Fu) : ((qw >> 4u) & 0x0F0F0F0Fu);
            return (int)__vsub4(nib | (((hw >> (2u * r6)) & 0x03030303u) << 4u), 0x20202020u);
        } else if (DT == PD_KQ_IQ4XS) {
            const uint32_t qw = q[j];
            return pd_kq_iq4_prmt((kh ? (qw >> 4u) : qw) & 0x0F0F0F0Fu);
        } else {
            const uint32_t qw = q[(j >> 1u) * 2u + kh];
            uint32_t nib = (j & 1u) ? ((qw >> 4u) & 0x0F0F0F0Fu) : (qw & 0x0F0F0F0Fu);
            if (Q5) nib |= (((h[kh] >> ((j >> 2u) * 4u)) << 4u) >> (j & 3u)) & 0x10101010u;
            return (int)__vsub4(nib, C4);
        }
    };

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    Raw ra, rb;
    uint32_t kt = s_lo + wsl;
    if (kt < s_hi) load(kt, ra);
    for (; kt < s_hi; kt += NSL) {
        const uint32_t kn = kt + NSL;
        if (kn < s_hi) load(kn, rb);  // next super's bytes in flight while this one folds
        // per-row d / dmin for this super
        const float d0 = __half2float(__ushort_as_half((unsigned short)(ra.rec0[0] & 0xFFFFu)));
        const float d8 = __half2float(__ushort_as_half((unsigned short)(ra.rec8[0] & 0xFFFFu)));
        const float m0 = MU ? __half2float(__ushort_as_half((unsigned short)(ra.rec0[0] >> 16u))) : 0.0f;
        const float m8 = MU ? __half2float(__ushort_as_half((unsigned short)(ra.rec8[0] >> 16u))) : 0.0f;
        // B-side per-column material for the super (prefetched with the strip):
        // the ks rung's xc*(s16a+s16b) fold for the mu term
        float xc[2][8], sx[2][8];
        #pragma unroll
        for (uint32_t c = 0; c < 2u; ++c) {
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j) {
                xc[c][j] = ra.xc[c][j];
                sx[c][j] = MU ? ra.xc[c][j] * ra.su[c][j] : 0.0f;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const int a0 = frag(ra.q0, ra.h0, j, 0u), a1 = frag(ra.q8, ra.h8, j, 0u);
            const int a2 = frag(ra.q0, ra.h0, j, 1u), a3 = frag(ra.q8, ra.h8, j, 1u);
            // row scales for block j (the ks rung's expressions)
            float d0s = 0.f, d8s = 0.f, m0s = 0.f, m8s = 0.f, s0lo = 0.f, s0hi = 0.f, s8lo = 0.f, s8hi = 0.f;
            if (K16) {
                s0lo = d0 * (float)(int8_t)(ra.rec0[1u + (j >> 1u)] >> (8u * ((2u * j) & 3u)));
                s0hi = d0 * (float)(int8_t)(ra.rec0[1u + (j >> 1u)] >> (8u * ((2u * j + 1u) & 3u)));
                s8lo = d8 * (float)(int8_t)(ra.rec8[1u + (j >> 1u)] >> (8u * ((2u * j) & 3u)));
                s8hi = d8 * (float)(int8_t)(ra.rec8[1u + (j >> 1u)] >> (8u * ((2u * j + 1u) & 3u)));
            } else if (DT == PD_KQ_IQ4XS) {
                d0s = d0 * (float)(int8_t)(ra.rec0[1u + (j >> 2u)] >> (8u * (j & 3u)));
                d8s = d8 * (float)(int8_t)(ra.rec8[1u + (j >> 2u)] >> (8u * (j & 3u)));
            } else if (DT == PD_KQ_Q40) {
                d0s = __half2float(__ushort_as_half((unsigned short)(ra.rec0[j >> 1u] >> (16u * (j & 1u)))));
                d8s = __half2float(__ushort_as_half((unsigned short)(ra.rec8[j >> 1u] >> (16u * (j & 1u)))));
            } else {
                const uint32_t sh = 8u * (j & 3u), i2 = 1u + (j >> 2u);
                const float Cf = Q5 ? 16.0f : 8.0f;
                d0s = d0 * (float)((ra.rec0[i2] >> sh) & 0xFFu);
                d8s = d8 * (float)((ra.rec8[i2] >> sh) & 0xFFu);
                m0s = Cf * d0s - m0 * (float)((ra.rec0[2u + i2] >> sh) & 0xFFu);
                m8s = Cf * d8s - m8 * (float)((ra.rec8[2u + i2] >> sh) & 0xFFu);
            }
            {
                const int b0 = ra.b[j][0], b1 = ra.b[j][1];
                const float xc0 = xc[0][j], xc1 = xc[1][j];
                if (K16) {
                    int d0_ = 0, d1_ = 0, d2_ = 0, d3_ = 0, e0_ = 0, e1_ = 0, e2_ = 0, e3_ = 0;
                    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+r"(d0_), "+r"(d1_), "+r"(d2_), "+r"(d3_) : "r"(a0), "r"(a1), "r"(b0));
                    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+r"(e0_), "+r"(e1_), "+r"(e2_), "+r"(e3_) : "r"(a2), "r"(a3), "r"(b1));
                    acc[0] += xc0 * (s0lo * (float)d0_ + s0hi * (float)e0_);
                    acc[1] += xc1 * (s0lo * (float)d1_ + s0hi * (float)e1_);
                    acc[2] += xc0 * (s8lo * (float)d2_ + s8hi * (float)e2_);
                    acc[3] += xc1 * (s8lo * (float)d3_ + s8hi * (float)e3_);
                } else {
                    int d0_ = 0, d1_ = 0, d2_ = 0, d3_ = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0_), "+r"(d1_), "+r"(d2_), "+r"(d3_)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    acc[0] += d0s * xc0 * (float)d0_;
                    acc[1] += d0s * xc1 * (float)d1_;
                    acc[2] += d8s * xc0 * (float)d2_;
                    acc[3] += d8s * xc1 * (float)d3_;
                    if (MU) {
                        acc[0] += m0s * sx[0][j];
                        acc[1] += m0s * sx[1][j];
                        acc[2] += m8s * sx[0][j];
                        acc[3] += m8s * sx[1][j];
                    }
                }
            }
        }
        ra = rb;
    }

    // the K slices' partial tiles meet once: [slice][row 16][col CS*8], summed in slice order
    __shared__ float red[8][16][16];
    red[wsl][g][ccol0] = acc[0];
    red[wsl][g][ccol0 + 1u] = acc[1];
    red[wsl][g + 8u][ccol0] = acc[2];
    red[wsl][g + 8u][ccol0 + 1u] = acc[3];
    __syncthreads();
    if (tid < 16u * CS * 8u) {
        const uint32_t row = tid / (CS * 8u), col = tid % (CS * 8u);
        float v = 0.0f;
        #pragma unroll
        for (uint32_t w = 0; w < NSL; ++w) v += red[w][row][col];
        if (row_base + row < out_dim && col < batch) y[(size_t)col * out_dim + row_base + row] = v;
    }
#else
    (void)data; (void)scales; (void)xq; (void)xs; (void)xsums; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_kquant_gemm_w4a8_sb(const void* data, const void* scales, const void* xq,
                           const void* xs, const void* xsums, void* part, void* y,
                           uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                           uint32_t dtype, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    const bool mine = dtype == PD_KQ_Q4K || dtype == PD_KQ_Q5K || dtype == PD_KQ_Q6K
                   || dtype == PD_KQ_IQ4XS || dtype == PD_KQ_Q40;
    // b = 9..16 measured slower than the K-split here (the two-fragment split
    // across warps doubles the strip reads an SM; GB202 up b=16 69 vs 61 us)
    if (!mine || batch > 8u || (in_dim & 255u) != 0u)
        return pd_kquant_gemm_mma_ks(data, scales, xq, xs, xsums, part, y, in_dim, out_dim, batch,
                                     dtype, stream);
    if (pd_kq_has_mu(dtype) && xsums == nullptr) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 15u) / 16u;
    const uint32_t n_super = in_dim >> 8u;
    // fill two CTAs an SM; a z-slice keeps at least 8 supers (one a warp)
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    if (nz > 8u) nz = 8u;
    if (nz > n_super / 8u) nz = n_super / 8u;
    if (nz < 1u) nz = 1u;
    const uint32_t per = (n_super + nz - 1u) / nz;
    nz = (n_super + per - 1u) / per;
    float* dst = nz > 1u ? (float*)part : (float*)y;
    #define PD_SB_LAUNCH(DTV, NS)                                                      \
        pd_kquant_sb_kernel<DTV, NS, 1u><<<dim3(tiles, 1u, nz), 256, 0, st>>>(              \
            (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,           \
            (const float*)xs, (const float*)xsums, dst, in_dim, out_dim, batch)
    #define PD_SB_DT(NS)                                                               \
        switch (dtype) {                                                               \
            case PD_KQ_Q40: PD_SB_LAUNCH(PD_KQ_Q40, NS); break;                        \
            case PD_KQ_Q4K: PD_SB_LAUNCH(PD_KQ_Q4K, NS); break;                        \
            case PD_KQ_Q5K: PD_SB_LAUNCH(PD_KQ_Q5K, NS); break;                        \
            case PD_KQ_Q6K: PD_SB_LAUNCH(PD_KQ_Q6K, NS); break;                        \
            default: PD_SB_LAUNCH(PD_KQ_IQ4XS, NS); break;                             \
        }
    PD_SB_DT(1u);
    #undef PD_SB_DT
    #undef PD_SB_LAUNCH
    if (nz > 1u) {
        const uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}
