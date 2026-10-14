// quant/kquant_w4a8_pipe4.cuh - the k-quant W4A8 prefill tile with NO unpacked
// weight tile (slot 851, pd_kquant_gemm_w4a8_pipe4).
//
// pipe2 / pipe3 (quant/kquant_w4a8.cuh, kquant_w4a8_ring.cuh) spend every
// half-super on a three-phase block: unpack the raw nibbles into a 32-bit
// tile_x in shared, barrier, then feed the mma from tile_x with scalar 4-byte
// loads (32 fragment words + 32 scale floats a thread a half) and the
// activation tile with another 64 fragment and 128 scale/sum loads. On GB202
// that sat the 27B's prefill at ~207 TOPS (Q5_K) against ~660 sustained for
// int8 mma.sync (the SM clock settles at ~1.7 GHz under this load; 825 in a
// 2.3 GHz burst): ncu put the shared feed, the barrier phases and the
// in-order LSU queue on the clock, not the tensor pipe. Marlin's design
// point (compressed weights live in shared, dequant at consumption) applied
// to the int8 ladder, plus what the 2026-10-14 kernel ladder measured:
//   - the raw 4-6.6 bpw strips stay on the 2-deep cp.async ring, padded to
//     a 16 B-odd row stride (DATAB + 16) so ldmatrix reads 8 rows
//     conflict-free without a swizzle;
//   - A fragments come straight off the raw strips through ldmatrix.x4 (one
//     x4 = 2 sub-blocks' worth of nibbles for a 16-row fragment) and unpack
//     in registers with the identical nibble math the build phase used;
//   - the per-row (dj, mu) scales expand in registers from the 24 B records
//     (the same f32 expressions), so tile_x and its barrier are gone;
//   - the activation tile rides the ring one half-super ahead for every
//     type (pipe3's trick, Q4_K-only there because tile_x ate the budget);
//   - B fragments come through ldmatrix.x4 (two n8 fragments an instruction)
//     off the mmq tile; a fragment's four column scales (dB0, dB1, S0, S1)
//     come as ONE LDS.128 off quads repacked once a step (3.4x cheaper than
//     the four scalar loads - the C-fragment lanes share 4 addresses);
//   - the next step's copies are issued one slot at a time from inside the
//     walk, not as a burst in front of it (the LSU queue is in order; a
//     burst of ~3,000 copy requests held the walk's shared loads behind it).
// The accumulation order (super, half, kk ascending; d term then mu term
// per block) and every fold expression are pipe2's, so y is byte-identical
// to v1 / pipe2 / pipe3 - the harness gates it. Types: Q4_K / Q5_K / Q6_K /
// IQ4_XS / Q4_0; the launcher hands the i-quant family to pipe2.
// Shared: Q4_K 92,160 B | Q5_K 100,352 | Q6_K 100,352 | IQ4_XS 83,968 - all
// under sm_120's 101,376 B single-block cap, 1 CTA/SM like every rung here.
// GB202 [5120 -> 17408] x 4096 rows, pipe2 -> pipe4: Q5_K 208 -> 237 TOPS,
// Q4_K 223 -> 245, Q6_K 181 -> 200, IQ4_XS 224 -> 263 (+10-16% at 1024 rows
// and on the qkv / out / down planes). What remains is the fold itself -
// 16 of the 23 instructions a mma, fixed by the per-32 f32 scales on both
// sides (bench/kq_w4a8_fold_gb202_bench.cu).

template <uint32_t DT>
__host__ __device__ __forceinline__ uint32_t pd_kwp4_rs() {
    return pd_kq_datab(DT) + 16u;  // padded raw row stride (16 B-odd: conflict-free ldmatrix)
}
template <uint32_t DT>
__host__ __device__ __forceinline__ uint32_t pd_kwp4_smem() {
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    // raw ring + records + activation ring (+ sums) + the (dB0, dB1[, S0, S1])
    // quads of the NEXT step, repacked at the end of this one
    return 2u * 128u * pd_kwp4_rs<DT>() + 2u * 128u * PD_KQ_SCB + 2u * 128u * PD_MMQ_YK * 4u
         + (MU ? 2u * 128u * 4u * 4u : 0u) + 2u * 4u * 64u * (MU ? 16u : 8u);
}

// x4 ldmatrix of four 8x8 b16 tiles; lane l supplies the row address of tile
// l>>3, row l&7. Returns the lane's 4 B of each tile: for an s8 fragment that
// is bytes 4t..4t+3 of row g (g = l>>2, t = l&3) - exactly the scalar words
// the tile_x path read.
__device__ __forceinline__ void pd_kwp4_ldsm(const void* p, int& r0, int& r1, int& r2, int& r3) {
    pd_mma_ldm_x4(p, r0, r1, r2, r3);
}

template <uint32_t DT>
__global__ void __launch_bounds__(256, 1) pd_kquant_w4a8_pipe4_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const uint8_t* __restrict__ yq, const float* __restrict__ xsums,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool K16 = (DT == PD_KQ_Q6K);
    constexpr uint32_t DATAB = DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    constexpr uint32_t RS = DATAB + 16u;
    constexpr uint32_t RAW_W = 128u * RS, RAW_R = 128u * PD_KQ_SCB;
    constexpr uint32_t TY = 128u * PD_MMQ_YK * 4u, TS = MU ? 128u * 4u * 4u : 0u;
    constexpr uint32_t C4 = DT == PD_KQ_Q5K ? 0x10101010u : 0x08080808u;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t i0 = (warp >> 1u) * 32u;    // the warp's 32-row strip
    const uint32_t jw = (warp & 1u) * 64u;     // the warp's 64-column half
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nct = batch_pad >> 7u;
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * 128u;
    // ldmatrix lane roles: tiles 0/1 = rows 0-7 / 8-15, tiles 2/3 the same
    // rows at the second 16 B chunk (k-half, or the next sub-block for IQ4_XS)
    const uint32_t lrow = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;
    const uint32_t lchk = (lane >> 4u) & 1u;

    extern __shared__ __align__(16) unsigned char pd_kwp4_sh[];
    unsigned char* const raw_w0 = pd_kwp4_sh;
    unsigned char* const raw_w1 = raw_w0 + RAW_W;
    unsigned char* const raw_r0 = raw_w1 + RAW_W;
    unsigned char* const raw_r1 = raw_r0 + RAW_R;
    unsigned char* const ty0 = raw_r1 + RAW_R;
    unsigned char* const ty1 = ty0 + TY;
    unsigned char* const ts0 = ty1 + TY;
    unsigned char* const ts1 = ts0 + TS;
    constexpr uint32_t QB = MU ? 16u : 8u;           // quad bytes: (dB0, dB1[, S0, S1])
    constexpr uint32_t QD = 4u * 64u * QB;            // one step's quads: [kk][col pair]
    unsigned char* const qd0 = ts1 + TS;
    unsigned char* const qd1 = qd0 + QD;
    float acc[4][2][2][4] = {};  // [16-col group][n8 frag][m16 frag n][c0..c3]

    // Copy roles are fixed per thread (no per-copy division): thread = (row
    // tid>>1, parity tid&1); it walks chunks parity, parity+2, ... of its row.
    // The row's global strip base is loop-invariant; a step adds kt * DATAB.
    const uint32_t srow = tid >> 1u, spar = tid & 1u;
    const bool srow_ok = (row_base + srow) < out_dim;
    const uint8_t* const wrow_g = data + (size_t)(row_base + srow) * n_super * DATAB;
    const uint8_t* const rrow_g = scales + (size_t)(row_base + srow) * n_super * PD_KQ_SCB;
    // The next step's copies are issued one slot at a time from inside the
    // walk (16 slots, one a (sub-block, column group) iteration) instead of as
    // one burst in front of it: the LSU queue is in order, and a burst of
    // ~3,000 copy requests held every shared load of the walk behind it
    // (measured -30% on the isolated walk; spreading recovered a third).
    // Slots 0..NA-1 the activation tile, NA the sums, then the weight strip
    // and its records (a super's first step only).
    constexpr uint32_t NA = (TY / 16u + 255u) / 256u;   // 5 (4.5 copies a thread)
    constexpr uint32_t WCH = DATAB / 16u;
    constexpr uint32_t NW = WCH / 2u;                  // weight chunks a thread
    auto copy_slot = [&](uint32_t slot, uint32_t s_next, bool with_w) {
        if (slot < NA) {
            const uint32_t i = slot * 256u + tid;
            if (i < TY / 16u) {
                unsigned char* const ty = (s_next & 1u) ? ty1 : ty0;
                const unsigned char* by = yq + ((size_t)s_next * batch_pad + col_base) * 144u;
                pd_mma_cpa16p(ty + i * 16u, by + i * 16u, true);
            }
        } else if (slot == NA) {
            if (MU && tid < TS / 16u) {
                unsigned char* const ts = (s_next & 1u) ? ts1 : ts0;
                const unsigned char* bs = (const unsigned char*)(xsums + ((size_t)s_next * batch_pad + col_base) * 4u);
                pd_mma_cpa16p(ts + tid * 16u, bs + tid * 16u, true);
            }
        } else if (with_w) {
            const uint32_t kt_n = s_next >> 1u, wb = kt_n & 1u;
            const uint32_t k = slot - NA - 1u;
            if (k < NW) {
                const uint32_t c = spar + 2u * k;
                pd_mma_cpa16p((wb ? raw_w1 : raw_w0) + srow * RS + c * 16u,
                              wrow_g + (size_t)kt_n * DATAB + c * 16u, srow_ok);
            } else if (k == NW) {
                unsigned char* const rr = (wb ? raw_r1 : raw_r0) + srow * PD_KQ_SCB;
                const uint8_t* const rg = rrow_g + (size_t)kt_n * PD_KQ_SCB;
                pd_kq_cpa8p(rr + spar * 8u, rg + spar * 8u, srow_ok);
                if (spar == 0u) pd_kq_cpa8p(rr + 16u, rg + 16u, srow_ok);
            }
        }
    };
    // (dB0, dB1[, S0, S1]) for column pair p, sub-block kk of step `step`, one
    // quad a thread, off the landed activation tile. The walk then reads a
    // fragment's four column scales with one LDS.128 instead of four LDS.32
    // (measured 3.4x cheaper on GB202 for this 4-distinct-address pattern).
    auto build_quads = [&](uint32_t step) {
        const uint32_t kk = tid >> 6u;
        const uint32_t l = tid & 63u;
        const uint32_t p = ((l & 3u) << 4u) | (l >> 2u);  // spread the 72-word stride over the banks
        const float* ty = (const float*)((step & 1u) ? ty1 : ty0);
        unsigned char* const qd = ((step & 1u) ? qd1 : qd0) + (kk * 64u + p) * QB;
        const float b0 = ty[(2u * p) * PD_MMQ_YK + kk], b1 = ty[(2u * p + 1u) * PD_MMQ_YK + kk];
        if (MU) {
            const float* ts = (const float*)((step & 1u) ? ts1 : ts0);
            *(float4*)qd = make_float4(b0, b1, ts[(2u * p) * 4u + kk], ts[(2u * p + 1u) * 4u + kk]);
        } else {
            *(float2*)qd = make_float2(b0, b1);
        }
    };

    auto step_mma = [&](uint32_t half, uint32_t wbuf, uint32_t abuf, bool next, uint32_t s_next,
                        bool next_w) {
        const unsigned char* const rw = wbuf ? raw_w1 : raw_w0;
        const unsigned char* const rr = wbuf ? raw_r1 : raw_r0;
        const int* const ty = (const int*)(abuf ? ty1 : ty0);
        const unsigned char* const qd = abuf ? qd1 : qd0;

        // ---- raw fragment words for the whole half, straight off the strips:
        // Q4_K/Q5_K/Q4_0: qs x4 per sub-block PAIR (+ qh x4, Q5_K); Q6_K: ql x4
        // per a (+ qh x4); IQ4_XS: qs x4 per pair. Index [n][.][kh*2 + rh]
        // (the mma's a0..a3 order).
        int rq[2][2][4], rh4[2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const unsigned char* arow = rw + (i0 + n * 16u + lrow) * RS;
            if (K16) {
                pd_kwp4_ldsm(arow + half * 64u + lchk * 16u, rq[n][0][0], rq[n][0][1], rq[n][0][2], rq[n][0][3]);
                pd_kwp4_ldsm(arow + half * 64u + 32u + lchk * 16u, rq[n][1][0], rq[n][1][1], rq[n][1][2], rq[n][1][3]);
                pd_kwp4_ldsm(arow + 128u + half * 32u + lchk * 16u, rh4[n][0], rh4[n][1], rh4[n][2], rh4[n][3]);
            } else if (DT == PD_KQ_IQ4XS) {
                #pragma unroll
                for (uint32_t kp = 0; kp < 2u; ++kp)
                    pd_kwp4_ldsm(arow + (half * 4u + kp * 2u + lchk) * 16u, rq[n][kp][0], rq[n][kp][1], rq[n][kp][2], rq[n][kp][3]);
            } else {
                if (DT == PD_KQ_Q5K) pd_kwp4_ldsm(arow + 128u + lchk * 16u, rh4[n][0], rh4[n][1], rh4[n][2], rh4[n][3]);
                #pragma unroll
                for (uint32_t kp = 0; kp < 2u; ++kp)
                    pd_kwp4_ldsm(arow + (half * 2u + kp) * 32u + lchk * 16u, rq[n][kp][0], rq[n][kp][1], rq[n][kp][2], rq[n][kp][3]);
            }
        }
        // per-row f16 d (and dmin) for rows i0+n*16+rh*8+g, off the 24 B records
        float dr[2][2], dmr[2][2];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            #pragma unroll
            for (uint32_t rh = 0; rh < 2u; ++rh) {
                const uint32_t w0 = *(const uint32_t*)(rr + (i0 + n * 16u + rh * 8u + g) * PD_KQ_SCB);
                __half hd, hm;
                const uint16_t hdb = (uint16_t)(w0 & 0xFFFFu), hmb = (uint16_t)(w0 >> 16u);
                memcpy(&hd, &hdb, 2u);
                memcpy(&hm, &hmb, 2u);
                dr[n][rh] = __half2float(hd);
                dmr[n][rh] = __half2float(hm);
            }
        }

        #pragma unroll
        for (uint32_t kk = 0; kk < 4u; ++kk) {
            // ---- this sub-block's A fragments (identical nibble math to pipe2's build)
            int A[2][4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t i = 0; i < 4u; ++i) {
                    if (K16) {
                        const uint32_t qw = (uint32_t)rq[n][kk & 1u][i], hw = (uint32_t)rh4[n][i];
                        const uint32_t nib = kk < 2u ? (qw & 0x0F0F0F0Fu) : ((qw >> 4u) & 0x0F0F0F0Fu);
                        A[n][i] = (int)__vsub4(nib | (((hw >> (2u * kk)) & 0x03030303u) << 4u), 0x20202020u);
                    } else if (DT == PD_KQ_IQ4XS) {
                        // i = kh*2 + rh: kh 0 = lo nibbles, 1 = hi, of word rh of chunk kk
                        const uint32_t qw = (uint32_t)rq[n][kk >> 1u][(kk & 1u) * 2u + (i & 1u)];
                        A[n][i] = pd_kq_iq4_prmt(((i >> 1u) ? (qw >> 4u) : qw) & 0x0F0F0F0Fu);
                    } else {
                        // Q5_K's 5th bit for sub-block j is bit j of each qh byte;
                        // hb holds the half's four bits at 0..3, so one shift lands
                        // bit kk at bit 4 of every byte (same value as pipe2's
                        // ((hw >> j) & 0x01010101) << 4)
                        const uint32_t qw = (uint32_t)rq[n][kk >> 1u][i];
                        const uint32_t hb = (uint32_t)rh4[n][i] >> (half * 4u);
                        uint32_t nib = (kk & 1u) ? ((qw >> 4u) & 0x0F0F0F0Fu) : (qw & 0x0F0F0F0Fu);
                        if (DT == PD_KQ_Q5K) nib |= ((hb << 4u) >> kk) & 0x10101010u;
                        A[n][i] = (int)__vsub4(nib, C4);
                    }
                }
            }
            // ---- this sub-block's row scales (pipe2's expressions; zero-filled
            // records on dead rows give d = 0 -> every scale 0)
            const uint32_t j = half * 4u + kk;
            float dA[2][2], muA[2][2], sA[2][2][2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t rh = 0; rh < 2u; ++rh) {
                    const uint8_t* rec = rr + (i0 + n * 16u + rh * 8u + g) * PD_KQ_SCB;
                    const float d = dr[n][rh];
                    if (K16) {
                        sA[n][rh][0] = d * (float)((const int8_t*)rec)[4u + 2u * j];
                        sA[n][rh][1] = d * (float)((const int8_t*)rec)[5u + 2u * j];
                    } else if (DT == PD_KQ_IQ4XS) {
                        dA[n][rh] = d * (float)((const int8_t*)rec)[4u + j];
                        muA[n][rh] = 0.0f;
                    } else if (DT == PD_KQ_Q40) {
                        dA[n][rh] = pd_kq40_dj(rec, j);
                        muA[n][rh] = 0.0f;
                    } else {
                        const float C = DT == PD_KQ_Q5K ? 16.0f : 8.0f;
                        const float dj = d * (float)rec[4u + j];
                        dA[n][rh] = dj;
                        muA[n][rh] = C * dj - dmr[n][rh] * (float)rec[12u + j];
                    }
                }
            }
            // ---- the walk over the warp's 64 columns: 4 groups of two n8 frags
            #pragma unroll
            for (uint32_t jg = 0; jg < 4u; ++jg) {
                if (next) copy_slot(kk * 4u + jg, s_next, next_w);
                const uint32_t jc0 = jw + jg * 16u;
                const unsigned char* brow = (const unsigned char*)ty
                    + (jc0 + (lane & 7u) + ((lane >> 4u) & 1u) * 8u) * (PD_MMQ_YK * 4u)
                    + 16u + ((lane >> 3u) & 1u) * 16u + kk * 32u;
                int b[4];  // b0, b1 of frag jc0; b0, b1 of frag jc0 + 8
                pd_kwp4_ldsm(brow, b[0], b[1], b[2], b[3]);
                #pragma unroll
                for (uint32_t f = 0; f < 2u; ++f) {
                    const uint32_t jc = jc0 + f * 8u;
                    // this fragment's columns jc+2t, jc+2t+1: one quad load
                    float dB0, dB1, S0 = 0.0f, S1 = 0.0f;
                    if (MU) {
                        const float4 q = *(const float4*)(qd + (kk * 64u + (jc >> 1u) + t) * 16u);
                        dB0 = q.x; dB1 = q.y; S0 = q.z; S1 = q.w;
                    } else {
                        const float2 q = *(const float2*)(qd + (kk * 64u + (jc >> 1u) + t) * 8u);
                        dB0 = q.x; dB1 = q.y;
                    }
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        float* a = acc[jg][f][n];
                        if (K16) {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][0]), "r"(A[n][1]), "r"(b[f * 2u]));
                            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)
                                : "r"(A[n][2]), "r"(A[n][3]), "r"(b[f * 2u + 1u]));
                            const float sa0 = sA[n][0][0], sa1 = sA[n][0][1];
                            const float sb0_ = sA[n][1][0], sb1_ = sA[n][1][1];
                            a[0] += dB0 * (sa0 * (float)d0 + sa1 * (float)e0);
                            a[1] += dB1 * (sa0 * (float)d1 + sa1 * (float)e1);
                            a[2] += dB0 * (sb0_ * (float)d2 + sb1_ * (float)e2);
                            a[3] += dB1 * (sb0_ * (float)d3 + sb1_ * (float)e3);
                        } else {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][0]), "r"(A[n][1]), "r"(A[n][2]), "r"(A[n][3]),
                                  "r"(b[f * 2u]), "r"(b[f * 2u + 1u]));
                            a[0] += dA[n][0] * dB0 * (float)d0;
                            a[1] += dA[n][0] * dB1 * (float)d1;
                            a[2] += dA[n][1] * dB0 * (float)d2;
                            a[3] += dA[n][1] * dB1 * (float)d3;
                            if (MU) {
                                a[0] += muA[n][0] * S0;
                                a[1] += muA[n][0] * S1;
                                a[2] += muA[n][1] * S0;
                                a[3] += muA[n][1] * S1;
                            }
                        }
                    }
                }
            }
        }
    };

    const uint32_t nsteps = 2u * n_super;
    // prologue: step 0's tiles, then its quads
    for (uint32_t slot = 0; slot <= NA + NW + 1u; ++slot) copy_slot(slot, 0u, true);
    pd_attn_cpa_commit();
    pd_mma_cpa_waitN<0>();
    __syncthreads();
    build_quads(0u);
    __syncthreads();
    for (uint32_t s = 0; s < nsteps; ++s) {
        const uint32_t kt = s >> 1u, half = s & 1u, cur = kt & 1u;
        const bool next = s + 1u < nsteps;
        // step s computes while step s+1's copies trickle in from the walk;
        // the super's weights ride the first step's slots (kt+1 lands in the
        // other raw buffer, which super kt-1 released at the barrier above)
        step_mma(half, cur, s & 1u, next, s + 1u, half == 1u);
        if (next) {
            pd_attn_cpa_commit();
            pd_mma_cpa_waitN<0>();
        }
        __syncthreads();  // step s+1's tiles visible, step s's buffers released
        if (next) build_quads(s + 1u);
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t jg = 0; jg < 4u; ++jg) {
        #pragma unroll
        for (uint32_t f = 0; f < 2u; ++f) {
            const uint32_t c0 = col_base + jw + jg * 16u + f * 8u + 2u * t;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = row_base + i0 + n * 16u + g;
                const uint32_t r8 = r0 + 8u;
                const float* a = acc[jg][f][n];
                if (r0 < out_dim) {
                    if (c0 < batch) y[(size_t)c0 * out_dim + r0] = a[0];
                    if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = a[1];
                }
                if (r8 < out_dim) {
                    if (c0 < batch) y[(size_t)c0 * out_dim + r8] = a[2];
                    if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = a[3];
                }
            }
        }
    }
#else
    (void)data; (void)scales; (void)yq; (void)xsums; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_kquant_gemm_w4a8_pipe4(const void* data, const void* scales, const void* yq,
                              const void* xsums, void* y, uint32_t in_dim,
                              uint32_t out_dim, uint32_t batch, uint32_t dtype,
                              void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    const bool mine = dtype == PD_KQ_Q4K || dtype == PD_KQ_Q5K || dtype == PD_KQ_Q6K
                   || dtype == PD_KQ_IQ4XS || dtype == PD_KQ_Q40;
    if (!mine || (in_dim & 255u) != 0u || (pd_kq_has_mu(dtype) && xsums == nullptr))
        return pd_kquant_gemm_w4a8_pipe2(data, scales, yq, xsums, y, in_dim, out_dim, batch,
                                         dtype, stream);
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (((batch + 127u) & ~127u) >> 7u);
    #define PD_KWP4_LAUNCH(DTV)                                                   \
        do {                                                                      \
            static const cudaError_t attr = cudaFuncSetAttribute(                 \
                (const void*)pd_kquant_w4a8_pipe4_kernel<DTV>,                    \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pd_kwp4_smem<DTV>()); \
            if (attr != cudaSuccess) return attr;                                 \
            pd_kquant_w4a8_pipe4_kernel<DTV><<<ntiles, 256, pd_kwp4_smem<DTV>(),  \
                                              (cudaStream_t)stream>>>(            \
                (const uint8_t*)data, (const uint8_t*)scales, (const uint8_t*)yq, \
                (const float*)xsums, (float*)y, in_dim, out_dim, batch);          \
        } while (0)
    switch (dtype) {
        case PD_KQ_Q40: PD_KWP4_LAUNCH(PD_KQ_Q40); break;
        case PD_KQ_Q4K: PD_KWP4_LAUNCH(PD_KQ_Q4K); break;
        case PD_KQ_Q5K: PD_KWP4_LAUNCH(PD_KQ_Q5K); break;
        case PD_KQ_Q6K: PD_KWP4_LAUNCH(PD_KQ_Q6K); break;
        default: PD_KWP4_LAUNCH(PD_KQ_IQ4XS); break;
    }
    #undef PD_KWP4_LAUNCH
    return pd_launch_status();
}
