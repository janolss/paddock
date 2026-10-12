// Original Nemotron Mamba-2/NoPE/ReLU² graph. SSD equations: Dao & Gu,
// arXiv:2405.21060; checkpoint semantics: NVIDIA and our CUDA executor.
// All contractions retain F32 operands/accumulators. No CPU model math.
// Sequence descriptor: slot, first packed row, row count, first SSD tile.
// Tile descriptor: slot, first packed row, row count, sequence index.
kernel void nemo_state_copy(device float* state [[buffer(0)]], constant uint* p [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if(i<p[0])state[ulong(p[1])*p[0]+i]=p[2]==UINT_MAX ? 0 : state[ulong(p[2])*p[0]+i];
}
kernel void nemo_conv(device const float* proj [[buffer(0)]],device const float* win [[buffer(1)]],
 device const float* w [[buffer(2)]],device const float* b [[buffer(3)]],device const uint* seq [[buffer(4)]],
 device float* out [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint slot=seq[4*g.y],first=seq[4*g.y+1],count=seq[4*g.y+2],i=g.x*256+tid;
    if(i>=count*6144)return;uint row=i/6144,c=i%6144;float v=b[c];
    for(uint j=0;j<4;++j){int t=int(row)+int(j)-3;
        float x=t<0 ? win[(ulong(slot)*3+uint(t+3))*6144+c] : proj[ulong(first+uint(t))*10304+4096+c];
        v=fma(x,w[c*4+j],v);}
    out[ulong(first+row)*6144+c]=v/(1+exp(-v));
}
// Separate commit after every conv reader completes. Each thread loads all
// three old values before replacing its channel, including a one-row decode.
kernel void nemo_conv_commit(device const float* proj [[buffer(0)]],device float* win [[buffer(1)]],
 device const uint* seq [[buffer(2)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint c=g.x*256+tid;if(c>=6144)return;uint slot=seq[4*g.y],first=seq[4*g.y+1],count=seq[4*g.y+2];
    float v[3];for(uint j=0;j<3;++j){int t=int(count)+int(j)-3;
        v[j]=t<0 ? win[(ulong(slot)*3+uint(t+3))*6144+c] : proj[ulong(first+uint(t))*10304+4096+c];}
    for(uint j=0;j<3;++j)win[(ulong(slot)*3+j)*6144+c]=v[j];
}
inline float nemo_dt(float raw){
    if(raw>20)return raw;
    float v=exp(raw);
    // Metal's log(1+v) loses small positive steps when 1+v rounds to one.
    // This four-term log1p series has <2.3e-8 relative truncation error
    // below -4; preserve tiny dt instead of freezing recurrent updates.
    return raw < -4 ? v*(1+v*(-0.5f+v*(1.0f/3-v*0.25f))) : log(1+v);
}
kernel void nemo_dt_check(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])out[i]=nemo_dt(x[i]);}
// Narrow requests keep one state component in each of four lane registers.
// One SIMD per (head, value column); sequence length, not neighbour batch
// shape, determines the route. Also the GPU witness for SSD tests.
kernel void nemo_scan(device const float* conv [[buffer(0)]],device const float* proj [[buffer(1)]],
 device const float* a [[buffer(2)]],device const float* d [[buffer(3)]],device const float* bias [[buffer(4)]],
 device float* state [[buffer(5)]],device const uint* seq [[buffer(6)]],device float* out [[buffer(7)]],
 constant uint* p [[buffer(8)]],uint3 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]]) {
    uint slot=seq[4*g.z],first=seq[4*g.z+1],count=seq[4*g.z+2];if(count>=16 && !p[0])return;
    uint h=g.y,col=g.x*4+sg,group=h/8;ulong si=(ulong(slot)*4096+h*64+col)*128;
    float s[4];for(uint j=0;j<4;++j)s[j]=state[si+lane+j*32];
    for(uint t=first;t<first+count;++t){
        float dt=nemo_dt(proj[ulong(t)*10304+10240+h]+bias[h]),decay=exp(a[h]*dt);
        float x=conv[ulong(t)*6144+h*64+col],sum=0;
        for(uint j=0;j<4;++j){uint n=lane+j*32;
            float b=conv[ulong(t)*6144+4096+group*128+n],c=conv[ulong(t)*6144+5120+group*128+n];
            s[j]=fma(decay,s[j],(dt*b)*x);sum+=s[j]*c;}
        sum=simd_sum(sum);if(lane==0)out[ulong(t)*4096+h*64+col]=sum+d[h]*x;
    }
    for(uint j=0;j<4;++j)state[si+lane+j*32]=s[j];
}
// Prefix decay resets every 32 rows, avoiding exponent under/overflow over
// a long prompt. Zero padded rows have identity decay and zero contraction.
kernel void nemo_ssd_prepare(device const float* proj [[buffer(0)]],device const float* a [[buffer(1)]],
 device const float* bias [[buffer(2)]],device const uint* tiles [[buffer(3)]],device float* decay [[buffer(4)]],
 device float* dt [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint first=tiles[4*g.y+1],count=tiles[4*g.y+2],h=g.x;float step=lane<count ? nemo_dt(proj[ulong(first+lane)*10304+10240+h]+bias[h]) : 0;
    float prefix=simd_prefix_inclusive_sum(step*a[h]);
    decay[(ulong(g.y)*64+h)*32+lane]=prefix;
    if(lane<count)dt[ulong(first+lane)*64+h]=step;
}
// C B^T (32x128x32), then the causal semiseparable decay mask. BK32
// panels are bounded independently of context and remain under Apple10's
// compiled threadgroup limit with API and shader instrumentation.
template<bool Mlx>
inline void nemo_ssd_matrix_impl(device const float* conv,device const uint* tiles,
 device const float* decay,device const float* dt,device float* matrix,
 uint2 g,uint tid,threadgroup float* ca,threadgroup float* ba) {
    uint h=g.x,first=tiles[4*g.y+1],count=tiles[4*g.y+2],group=h/8;
    auto c=tensor(ca,extents<int,32,32>(),array<int,2>{1,32});
    auto b=tensor(ba,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(c),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<128;base+=32){
        for(uint i=tid;i<1024;i+=128){uint r=i/32,n=base+i%32;
            ca[i]=r<count ? conv[ulong(first+r)*6144+5120+group*128+n] : 0;
            ba[i]=r<count ? conv[ulong(first+r)*6144+4096+group*128+n] : 0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(c,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong off=(ulong(g.y)*64+h)*32;
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        if(it.is_valid_element()){uint t=ij[1],s=ij[0];
            matrix[off*32+t*32+s]=t<count && s<=t ? (Mlx?mlx_bf(*it):*it)*exp(decay[off+t]-decay[off+s])*dt[ulong(first+s)*64+h] : 0;}}
}
kernel void nemo_ssd_matrix(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* dt [[buffer(3)]],device float* matrix [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup float ca[32*32],ba[32*32];nemo_ssd_matrix_impl<false>(conv,tiles,decay,dt,matrix,g,tid,ca,ba);}
kernel void nemo_mlx_ssd_matrix(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* dt [[buffer(3)]],device float* matrix [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup float ca[32*32],ba[32*32];nemo_ssd_matrix_impl<true>(conv,tiles,decay,dt,matrix,g,tid,ca,ba);}
// A tile's final state increment B^T (dt*X*decay). Results use the same
// [head,value,state] layout as decode; no state-sized threadgroup buffer.
kernel void nemo_ssd_delta(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* dt [[buffer(3)]],device float* delta [[buffer(4)]],
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint first=tiles[4*g.z+1],count=tiles[4*g.z+2],h=g.y,col=g.x/4*16,nbase=g.x%4*32;
    ulong off=(ulong(g.z)*64+h)*32;threadgroup float xx[16*32],bb[32*32];
    for(uint i=tid;i<512;i+=128){uint c=col+i/32,t=i%32;
        xx[i]=t<count ? conv[ulong(first+t)*6144+h*64+c]*dt[ulong(first+t)*64+h]*exp(decay[off+count-1]-decay[off+t]) : 0;}
    for(uint i=tid;i<1024;i+=128){uint n=nbase+i/32,t=i%32;
        bb[i]=t<count ? conv[ulong(first+t)*6144+4096+(h/8)*128+n] : 0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x=tensor(xx,extents<int,32,16>(),array<int,2>{1,32});
    auto b=tensor(bb,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.get_destination_cooperative_tensor<decltype(x),decltype(b),float>();op.run(x,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element())
        delta[((ulong(g.z)*64+h)*64+col+ij[1])*128+nbase+ij[0]]=*it;}
}
// Only the short tile-boundary recurrence is sequential. All 524288 state
// components and requests run independently; incoming snapshots belong to
// this dispatch, not a global cache or a host-side model computation.
kernel void nemo_ssd_states(device float* state [[buffer(0)]],device const float* delta [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const uint* seq [[buffer(3)]],device const uint* tiles [[buffer(4)]],
 device float* incoming [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint count=seq[4*g.y+2];if(count<16)return;uint i=g.x*256+tid,slot=seq[4*g.y],first=seq[4*g.y+3],h=i/8192;
    float s=state[ulong(slot)*524288+i];
    for(uint t=first;t<first+(count+31)/32;++t){uint n=tiles[4*t+2];ulong off=ulong(t)*524288+i;
        incoming[off]=s;s=fma(exp(decay[(ulong(t)*64+h)*32+n-1]),s,delta[off]);}
    state[ulong(slot)*524288+i]=s;
}
// Incoming-state and within-tile output contractions are fused in registers.
// Each TG owns 32 tokens x 16 value columns, staging only BK32 panels.
template<bool Mlx>
inline void nemo_ssd_output_impl(device const float* conv,device const uint* tiles,
 device const float* decay,device const float* matrix,device const float* incoming,
 device const float* d,device float* out,uint3 g,
 uint tid,threadgroup float* aa,threadgroup float* bb) {
    uint first=tiles[4*g.z+1],count=tiles[4*g.z+2],h=g.y,col=g.x*16;ulong off=(ulong(g.z)*64+h)*32;
    auto a=tensor(aa,extents<int,32,32>(),array<int,2>{1,32});
    auto b=tensor(bb,extents<int,32,16>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,16,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<128;base+=32){
        for(uint i=tid;i<1024;i+=128){uint t=i/32,n=base+i%32;
            aa[i]=t<count ? conv[ulong(first+t)*6144+5120+(h/8)*128+n] : 0;}
        for(uint i=tid;i<512;i+=128)bb[i]=incoming[((ulong(g.z)*64+h)*64+col+i/32)*128+base+i%32];
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element())*it*=exp(decay[off+ij[1]]);}
    for(uint i=tid;i<1024;i+=128)aa[i]=matrix[off*32+i];
    for(uint i=tid;i<512;i+=128){uint t=i%32;bb[i]=t<count ? conv[ulong(first+t)*6144+h*64+col+i/32] : 0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count){
        ulong at=ulong(first+ij[1])*4096+h*64+col+ij[0];
        out[at]=Mlx ? mlx_bf(mlx_bf(*it)+mlx_bf(d[h]*conv[ulong(first+ij[1])*6144+h*64+col+ij[0]])) : *it+d[h]*conv[ulong(first+ij[1])*6144+h*64+col+ij[0]];}}
}
kernel void nemo_ssd_output(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* matrix [[buffer(3)]],device const float* incoming [[buffer(4)]],
 device const float* d [[buffer(5)]],device float* out [[buffer(6)]],uint3 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {threadgroup float aa[32*32],bb[16*32];nemo_ssd_output_impl<false>(conv,tiles,decay,matrix,incoming,d,out,g,tid,aa,bb);}
kernel void nemo_mlx_ssd_output(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* matrix [[buffer(3)]],device const float* incoming [[buffer(4)]],
 device const float* d [[buffer(5)]],device float* out [[buffer(6)]],uint3 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {threadgroup float aa[32*32],bb[16*32];nemo_ssd_output_impl<true>(conv,tiles,decay,matrix,incoming,d,out,g,tid,aa,bb);}
kernel void nemo_gated_norm(device const float* y [[buffer(0)]],device const float* proj [[buffer(1)]],
 device const float* w [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sum[256];uint c=g.x*512+tid;ulong at=ulong(g.y)*4096+c;
    float z0=proj[ulong(g.y)*10304+c],z1=proj[ulong(g.y)*10304+c+256];
    float v0=y[at]*(z0/(1+exp(-z0))),v1=y[at+256]*(z1/(1+exp(-z1)));
    sum[tid]=v0*v0+v1*v1;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=128;s>0;s>>=1){if(tid<s)sum[tid]+=sum[tid+s];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=rsqrt(sum[0]/512+as_type<float>(p[0]));out[at]=v0*inv*w[c];out[at+256]=v1*inv*w[c+256];
}
kernel void nemo_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
 device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],device const uint* meta [[buffer(4)]],
 device const uint* pages [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*256)return;uint row=i/256,pos=meta[row*2+1],slot=meta[row*2];
    ulong at=(ulong(pages[slot*p[1]+pos/16])*16+pos%16)*256+i%256;kc[at]=half(k[i]);vc[at]=half(v[i]);
}
kernel void nemo_attention_decode(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
 device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[16*32],prob[16*32],highs[16],sums[16];
    gemma_decode<128,16,true>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}

// Selection-only correction, normalized original sigmoid probabilities.
kernel void nemo_route(device const float* logits [[buffer(0)]],device const float* bias [[buffer(1)]],
 device uint* ids [[buffer(2)]],device float* weights [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float prob[4],score[4];for(uint j=0;j<4;++j){uint e=lane+j*32;prob[j]=1/(1+exp(-logits[row*128+e]));score[j]=prob[j]+bias[e];}
    float selected[6],sum=0;uint chosen[6];
    for(uint pick=0;pick<6;++pick){float best=-INFINITY;for(uint j=0;j<4;++j)best=max(best,score[j]);best=simd_max(best);
        uint id=UINT_MAX;for(uint j=0;j<4;++j)if(score[j]==best)id=min(id,lane+j*32);id=simd_min(id);if(id==UINT_MAX)id=0;
        float value=simd_broadcast(prob[id/32],id%32);chosen[pick]=id;selected[pick]=value;sum+=value;
        for(uint j=0;j<4;++j)if(lane+j*32==id)score[j]=-INFINITY;}
    if(lane==0)for(uint j=0;j<6;++j){ids[row*6+j]=chosen[j];weights[row*6+j]=selected[j]/sum*2.5f;}
}
template<bool Down>
inline void nemo_expert_decode(device const uchar* w,device const float* x,device const uint* ids,device float* out,
 constant uint* p,uint2 g,uint sg,uint lane){
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;ulong wb=(ulong(ids[entry])*p[1]+n)*p[0];float4 acc=0;
    for(uint k=lane*32;k<p[0];k+=1024){device const uchar* b=w+(wb+k)/32*34;float scale=float(*reinterpret_cast<device const half*>(b));
        for(uint j=0;j<32;j+=4)acc=fma(float4(*reinterpret_cast<device const packed_char4*>(b+2+j))*scale,
            *reinterpret_cast<device const float4*>(x+ulong(Down?entry:entry/6)*p[0]+k+j),acc);}
    float v=simd_sum(acc.x+acc.y+acc.z+acc.w);if(!Down)v=max(v,0.0f)*max(v,0.0f);if(lane==0)out[ulong(entry)*p[1]+n]=v;
}
template<bool Down,uint Ty=8>
inline void nemo_expert_grouped(device const uchar* w,device const float* x,device const uint* lists,device const uint* counts,
 device const uint* tiles,device float* out,constant uint* p,uint2 g,uint tid,threadgroup float* aa,threadgroup float* bb){
    if(g.y>=tiles[0])return;uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(16u,counts[e]-first),nbase=g.x*32;
    auto a=tensor(aa,extents<int,64,16>(),array<int,2>{1,64});auto b=tensor(bb,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid;i<1024;i+=128){uint r=i/64,k=base+i%64,entry=r<count?lists[e*p[2]*6+first+r]:0;
            aa[i]=r<count && k<p[0]?x[ulong(Down?entry:entry/6)*p[0]+k]:0;}
        for(uint i=tid;i<2048;i+=128){uint n=nbase+i/64,k=base+i%64;ulong at=(ulong(e)*p[1]+n)*p[0]+k;
            if constexpr(Ty==0x108) bb[i]=n<p[1] && k<p[0]?mlx_bf(dg_weight(w,p[0],p[1]*128,Ty,at)):0;
            else bb[i]=n<p[1] && k<p[0]?qmoe_q8(w,at):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count && nbase+ij[0]<p[1]){
        uint entry=lists[e*p[2]*6+first+ij[1]];float v=*it;if constexpr(Ty==0x108)v=mlx_bf(v);
        if(!Down)v=max(v,0.0f)*max(v,0.0f);if constexpr(Ty==0x108)v=mlx_bf(v);out[ulong(entry)*p[1]+nbase+ij[0]]=v;}}
}
#define NEMO_EXPERT(NAME,DOWN) \
kernel void NAME##_decode(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){nemo_expert_decode<DOWN>(w,x,ids,out,p,g,sg,lane);} \
kernel void NAME##_grouped(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){threadgroup float aa[16*64],bb[32*64];nemo_expert_grouped<DOWN>(w,x,lists,counts,tiles,out,p,g,tid,aa,bb);}
NEMO_EXPERT(nemo_up,false)
NEMO_EXPERT(nemo_down,true)
#undef NEMO_EXPERT
kernel void nemo_relu2(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]){if(i<p[0]){float v=max(x[i],0.0f);x[i]=v*v;}}
kernel void nemo_fold(device const float* out [[buffer(0)]],device const float* weights [[buffer(1)]],device float* shared [[buffer(2)]],
 constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*2688)return;uint row=i/2688,n=i%2688;float v=0;
    for(uint j=0;j<6;++j)v+=out[(ulong(row)*6+j)*2688+n]*weights[row*6+j];shared[i]+=v;}
