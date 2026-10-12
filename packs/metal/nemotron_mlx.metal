// Native MLX affine8 graph. Operation-boundary BF16 is distinct from GGUF;
// activations are F32 carriers, cache BF16, recurrent state remains F32.
// Four packed bytes share their scale and bias; no resident expanded weights.
// Tensor prefill follows BF16 projection/SSD boundaries; narrow live rows keep
// their independent SIMD contract. Legacy F32 kernels remain GPU witnesses.
inline void nemo_mlx_stage4(device const uchar* w,uint K,uint N,ulong at,threadgroup float* out) {
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float4 codes=float4(*reinterpret_cast<device const uchar4*>(w+at));
    *reinterpret_cast<threadgroup float4*>(out)=mlx_bf(codes*float(scales[at/64])+float(biases[at/64]));
}
inline void nemo_mlx_stage4(device const uchar* w,uint K,uint N,ulong at,threadgroup bfloat* out) {
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float4 codes=float4(*reinterpret_cast<device const uchar4*>(w+at));
    *reinterpret_cast<threadgroup bfloat4*>(out)=bfloat4(codes*float(scales[at/64])+float(biases[at/64]));
}
// A thread owns one contiguous half/full quantization group. Reuse metadata
// across all packed bytes instead of refetching it for every four elements.
template<uint Count>
inline void nemo_mlx_stage_group(device const uchar* w,uint K,uint N,ulong at,threadgroup bfloat* out) {
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float scale=float(scales[at/64]),bias=float(biases[at/64]);
    #pragma unroll
    for(uint j=0;j<Count;j+=4){
        float4 codes=float4(*reinterpret_cast<device const uchar4*>(w+at+j));
        *reinterpret_cast<threadgroup bfloat4*>(out+j)=bfloat4(codes*scale+bias);
    }
}
// Fixed affine8 decoding, with the original scalar contraction order. Reuse
// packed codes/metadata across live rows without changing each lane's sum.
template<uint R,uint Items>
inline void nemo_mlx_vectors_impl(device const uchar* w,device const float* x,device float* out,
 constant uint* p,uint2 g,uint sg,uint lane) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=g.x*4+sg;if(n>=p[1])return;uint first=g.y*R,valid=min(R,p[2]-first);
    ulong total=ulong(p[0])*p[1],wr=ulong(n)*p[0];
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+total);
    device const bfloat* biases=scales+total/64;
    float sums[R];for(uint r=0;r<R;++r)sums[r]=0;
    for(uint k=lane*Items;k<p[0];k+=32*Items){
        float4 codes[Items/4];for(uint j=0;j<Items/4;++j)codes[j]=float4(*reinterpret_cast<device const uchar4*>(w+wr+k+j*4));
        float scale=float(scales[(wr+k)/64]),bias=float(biases[(wr+k)/64]);
        for(uint r=0;r<R;++r){if(r>=valid)continue;float dot=0,xsum=0;
            for(uint j=0;j<Items/4;++j){float4 v=*reinterpret_cast<device const float4*>(x+ulong(first+r)*p[0]+k+j*4);
                for(uint c=0;c<4;++c){dot+=v[c]*codes[j][c];xsum+=v[c];}}
            sums[r]+=dot*scale+xsum*bias;}
    }
    for(uint r=0;r<R;++r){float sum=simd_sum(sums[r]);if(lane==0 && r<valid)out[ulong(first+r)*p[1]+n]=mlx_bf(sum);}
}
#define NEMO_VECTORS(NAME,R) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    if(p[0]%256==0)nemo_mlx_vectors_impl<R,8>(w,x,out,p,g,sg,lane);else nemo_mlx_vectors_impl<R,4>(w,x,out,p,g,sg,lane);}
NEMO_VECTORS(nemo_mlx_vector1,1)
NEMO_VECTORS(nemo_mlx_vector4,4)
#undef NEMO_VECTORS
// Share each input load across output columns as well as live request rows.
// The per-output lane sums and affine-operation order remain unchanged.
template<uint R,uint Items,uint Cols>
inline void nemo_mlx_vectors_reuse_impl(device const uchar* w,device const float* x,device float* out,
 constant uint* p,uint2 g,uint sg,uint lane) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=(g.x*4+sg)*Cols;if(n>=p[1])return;uint first=g.y*R,valid=min(R,p[2]-first);
    ulong total=ulong(p[0])*p[1];
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+total);
    device const bfloat* biases=scales+total/64;
    float sums[R][Cols];for(uint r=0;r<R;++r)for(uint c=0;c<Cols;++c)sums[r][c]=0;
    for(uint k=lane*Items;k<p[0];k+=32*Items){
        float4 codes[Cols][Items/4];float scales_c[Cols],biases_c[Cols];
        for(uint c=0;c<Cols;++c){ulong at=ulong(n+c)*p[0]+k;
            for(uint j=0;j<Items/4;++j)codes[c][j]=n+c<p[1]?float4(*reinterpret_cast<device const uchar4*>(w+at+j*4)):float4(0);
            scales_c[c]=n+c<p[1]?float(scales[at/64]):0;biases_c[c]=n+c<p[1]?float(biases[at/64]):0;}
        for(uint r=0;r<R;++r){if(r>=valid)continue;float dots[Cols],xsum=0;for(uint c=0;c<Cols;++c)dots[c]=0;
            for(uint j=0;j<Items/4;++j){float4 v=*reinterpret_cast<device const float4*>(x+ulong(first+r)*p[0]+k+j*4);
                for(uint i=0;i<4;++i){for(uint c=0;c<Cols;++c)dots[c]+=v[i]*codes[c][j][i];xsum+=v[i];}}
            for(uint c=0;c<Cols;++c)sums[r][c]+=dots[c]*scales_c[c]+xsum*biases_c[c];}
    }
    for(uint r=0;r<R;++r)for(uint c=0;c<Cols;++c){float sum=simd_sum(sums[r][c]);if(lane==0 && r<valid && n+c<p[1])out[ulong(first+r)*p[1]+n+c]=mlx_bf(sum);}
}
#define NEMO_REUSE_VECTORS(NAME,R) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    if(p[0]%256==0)nemo_mlx_vectors_reuse_impl<R,8,2>(w,x,out,p,g,sg,lane);else nemo_mlx_vectors_reuse_impl<R,4,2>(w,x,out,p,g,sg,lane);}
NEMO_REUSE_VECTORS(nemo_mlx_vector1_reuse,1)
NEMO_REUSE_VECTORS(nemo_mlx_vector4_reuse,4)
#undef NEMO_REUSE_VECTORS
// Router weights are already F32. Reuse them across four input rows and share
// inputs across two output columns, preserving each lane's original k order.
kernel void nemo_mlx_router4(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint n=(g.x*4+sg)*2,first=g.y*4;if(n>=p[1])return;
    float sums[4][2];for(uint r=0;r<4;++r)for(uint c=0;c<2;++c)sums[r][c]=0;
    for(uint k=lane;k<p[0];k+=32){
        float w0=w[ulong(n)*p[0]+k],w1=n+1<p[1]?w[ulong(n+1)*p[0]+k]:0;
        for(uint r=0;r<4;++r)if(first+r<p[2]){float v=x[ulong(first+r)*p[0]+k];sums[r][0]+=w0*v;sums[r][1]+=w1*v;}
    }
    for(uint r=0;r<4;++r)for(uint c=0;c<2;++c){float v=simd_sum(sums[r][c]);if(lane==0 && first+r<p[2] && n+c<p[1])out[ulong(first+r)*p[1]+n+c]=v;}
}
template<typename Tile,uint BM,uint BN,typename WeightTile=Tile>
inline void nemo_mlx_project_impl(device const uchar* w,device const float* x,
 device float* out,constant uint* p,uint2 g,uint tid,threadgroup Tile* aa,threadgroup WeightTile* bb) {
    auto a=tensor(aa,extents<int,64,BM>(),array<int,2>{1,64});
    auto b=tensor(bb,extents<int,64,BN>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(BM,BN,64,false,true,sizeof(Tile)==2,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid*4;i<BM*64;i+=128*4){uint r=g.y*BM+i/64,k=base+i%64;
            float4 v=r<p[2]?*reinterpret_cast<device const float4*>(x+ulong(r)*p[0]+k):float4(0);
            *reinterpret_cast<threadgroup vec<Tile,4>*>(aa+i)=vec<Tile,4>(v);}
        for(uint i=tid*4;i<BN*64;i+=128*4){uint n=g.x*BN+i/64,k=base+i%64;
            if(n<p[1])nemo_mlx_stage4(w,p[0],p[1],ulong(n)*p[0]+k,bb+i);
            else *reinterpret_cast<threadgroup vec<WeightTile,4>*>(bb+i)=vec<WeightTile,4>(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint r=g.y*BM+ij[1],n=g.x*BN+ij[0];
        if(it.is_valid_element() && r<p[2] && n<p[1])out[ulong(r)*p[1]+n]=mlx_bf(*it);}
}
#define NEMO_PROJECT(NAME,TYPE,BM,BN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup TYPE aa[BM*64],bb[BN*64];nemo_mlx_project_impl<TYPE,BM,BN>(w,x,out,p,g,tid,aa,bb);}
NEMO_PROJECT(nemo_mlx_project_exact,float,16,32)
NEMO_PROJECT(nemo_mlx_project_exact_wide,float,32,64)
NEMO_PROJECT(nemo_mlx_project,bfloat,16,32)
NEMO_PROJECT(nemo_mlx_project_wide,bfloat,32,64)
#undef NEMO_PROJECT
kernel void nemo_mlx_project_exact_mixed(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float aa[32*64];threadgroup bfloat bb[64*64];nemo_mlx_project_impl<float,32,64,bfloat>(w,x,out,p,g,tid,aa,bb);
}
kernel void nemo_mlx_project_exact_large(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float aa[64*64];threadgroup bfloat bb[64*64];nemo_mlx_project_impl<float,64,64,bfloat>(w,x,out,p,g,tid,aa,bb);
}
// Register staging, with four independent SIMD tiles per threadgroup. Keep
// F32 operands/reductions while avoiding the threadgroup-memory round trip.
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
kernel void nemo_mlx_project_cooperative(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM=16,BN=32,BK=32;
    g=uint2(g.x*2+sg%2,g.y*2+sg/2);
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto a=op.get_left_input_cooperative_tensor<float,float,float>();
    auto b=op.get_right_input_cooperative_tensor<float,float,float>();
    auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=BK){
        for(auto it=a.begin();it!=a.end();++it){auto ij=it.get_multidimensional_index();uint r=g.y*BM+ij[1],k=base+ij[0];
            if(it.is_valid_element())*it=r<p[2] && k<p[0]?x[ulong(r)*p[0]+k]:0;}
        for(auto it=b.begin();it!=b.end();++it){auto ij=it.get_multidimensional_index();uint n=g.x*BN+ij[1],k=base+ij[0];
            if(it.is_valid_element())*it=n<p[1] && k<p[0]?mlx_bf(dg_weight(w,p[0],p[1],0x108,ulong(n)*p[0]+k)):0;}
        op.run(a,b,acc);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint r=g.y*BM+ij[1],n=g.x*BN+ij[0];
        if(it.is_valid_element() && r<p[2] && n<p[1])out[ulong(r)*p[1]+n]=mlx_bf(*it);}
}
#endif
// Partition results and the running reduction both round to BF16.
kernel void nemo_mlx_splitk_reference(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat aa[16*32],bb[32*32];auto a=tensor(aa,extents<int,32,16>(),array<int,2>{1,32});auto b=tensor(bb,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();auto total=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<total.get_capacity();++i)total[i]=0;
    uint part=p[0]/p[5];
    for(uint split=0;split<p[5];++split){
        for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
        for(uint base=split*part;base<(split+1)*part;base+=32){
            for(uint i=tid*4;i<16*32;i+=128*4){uint r=g.y*16+i/32,k=base+i%32;*reinterpret_cast<threadgroup bfloat4*>(aa+i)=r<p[2]?bfloat4(*reinterpret_cast<device const float4*>(x+ulong(r)*p[0]+k)):bfloat4(0);}
            for(uint i=tid*4;i<32*32;i+=128*4){uint n=g.x*32+i/32,k=base+i%32;if(n<p[1])nemo_mlx_stage4(w,p[0],p[1],ulong(n)*p[0]+k,bb+i);else *reinterpret_cast<threadgroup bfloat4*>(bb+i)=bfloat4(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}
        for(uint i=0;i<acc.get_capacity();++i)total[i]=p[6]?mlx_bf(total[i]+mlx_bf(acc[i])):total[i]+mlx_bf(acc[i]);
    }
    for(auto it=total.begin();it!=total.end();++it){auto ij=it.get_multidimensional_index();uint r=g.y*16+ij[1],n=g.x*32+ij[0];if(it.is_valid_element() && r<p[2] && n<p[1])out[ulong(r)*p[1]+n]=mlx_bf(*it);}
}
// Parallel split-K: bounded BF16 partials, followed by the checkpoint's BF16
// reduction. Each group only traverses one partition, including narrow K/V.
kernel void nemo_mlx_project_split(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device bfloat* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat aa[16*32],bb[32*32];auto a=tensor(aa,extents<int,32,16>(),array<int,2>{1,32});auto b=tensor(bb,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint part=p[0]/p[5];
    for(uint base=g.z*part;base<(g.z+1)*part;base+=32){
        for(uint i=tid*4;i<16*32;i+=128*4){uint r=g.y*16+i/32,k=base+i%32;*reinterpret_cast<threadgroup bfloat4*>(aa+i)=r<p[2]?bfloat4(*reinterpret_cast<device const float4*>(x+ulong(r)*p[0]+k)):bfloat4(0);}
        for(uint i=tid*4;i<32*32;i+=128*4){uint n=g.x*32+i/32,k=base+i%32;if(n<p[1])nemo_mlx_stage4(w,p[0],p[1],ulong(n)*p[0]+k,bb+i);else *reinterpret_cast<threadgroup bfloat4*>(bb+i)=bfloat4(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint r=g.y*16+ij[1],n=g.x*32+ij[0];if(it.is_valid_element() && r<p[2] && n<p[1])out[(ulong(g.z)*p[2]+r)*p[1]+n]=bfloat(*it);}
}
kernel void nemo_mlx_project_split_sum(device const bfloat* parts [[buffer(0)]],device float* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint size=p[0]*p[1];if(i>=size)return;float total=0;
    for(uint j=0;j<p[2];++j)total=mlx_bf(total+float(parts[ulong(j)*size+i]));out[i]=total;
}
kernel void nemo_mlx_ssd_delta_reference(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* dt [[buffer(3)]],device float* delta [[buffer(4)]],
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    uint first=tiles[4*g.z+1],count=tiles[4*g.z+2],h=g.y,col=g.x/4*16,nbase=g.x%4*32;
    ulong off=(ulong(g.z)*64+h)*32;threadgroup float xx[16*32],bb[32*32];
    for(uint i=tid;i<512;i+=128){uint c=col+i/32,t=i%32;
        xx[i]=t<count?(dt[ulong(first+t)*64+h]*conv[ulong(first+t)*6144+h*64+c])*exp(decay[off+count-1]-decay[off+t]):0;}
    for(uint i=tid;i<1024;i+=128){uint n=nbase+i/32,t=i%32;
        bb[i]=t<count?conv[ulong(first+t)*6144+4096+(h/8)*128+n]:0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x=tensor(xx,extents<int,32,16>(),array<int,2>{1,32});auto b=tensor(bb,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,true,matmul2d_descriptor::mode::multiply);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.get_destination_cooperative_tensor<decltype(x),decltype(b),float>();op.run(x,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element())
        delta[((ulong(g.z)*64+h)*64+col+ij[1])*128+nbase+ij[0]]=*it;}
}
kernel void nemo_mlx_ssd_matrix_reference(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* dt [[buffer(3)]],device float* matrix [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint h=g.x,first=tiles[4*g.y+1],count=tiles[4*g.y+2],group=h/8;
    threadgroup bfloat ca[32*32],ba[32*32];
    auto c=tensor(ca,extents<int,32,32>(),array<int,2>{1,32});auto b=tensor(ba,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.get_destination_cooperative_tensor<decltype(c),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<128;base+=32){
        for(uint i=tid;i<1024;i+=128){uint r=i/32,n=base+i%32;
            ca[i]=r<count?bfloat(conv[ulong(first+r)*6144+5120+group*128+n]):bfloat(0);
            ba[i]=r<count?bfloat(conv[ulong(first+r)*6144+4096+group*128+n]):bfloat(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(c,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}
    ulong off=(ulong(g.y)*64+h)*32;
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element()){
        uint t=ij[1],s=ij[0];matrix[off*32+t*32+s]=t<count && s<=t?mlx_bf(*it)*exp(decay[off+t]-decay[off+s]):0;}}
}
kernel void nemo_mlx_ssd_output_reference(device const float* conv [[buffer(0)]],device const uint* tiles [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const float* matrix [[buffer(3)]],device const float* incoming [[buffer(4)]],
 device const float* d [[buffer(5)]],device const float* dt [[buffer(6)]],device float* out [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    uint first=tiles[4*g.z+1],count=tiles[4*g.z+2],h=g.y,col=g.x*16;ulong off=(ulong(g.z)*64+h)*32;
    threadgroup float aa[32*32],bb[16*32];auto a=tensor(aa,extents<int,32,32>(),array<int,2>{1,32});auto b=tensor(bb,extents<int,32,16>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,16,32,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto prior=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    auto current=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<prior.get_capacity();++i){prior[i]=0;current[i]=0;}
    for(uint base=0;base<128;base+=32){
        for(uint i=tid;i<1024;i+=128){uint t=i/32,n=base+i%32;aa[i]=t<count?conv[ulong(first+t)*6144+5120+(h/8)*128+n]:0;}
        for(uint i=tid;i<512;i+=128)bb[i]=incoming[((ulong(g.z)*64+h)*64+col+i/32)*128+base+i%32];
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,prior);threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(uint i=tid;i<1024;i+=128)aa[i]=matrix[off*32+i];
    for(uint i=tid;i<512;i+=128){uint t=i%32;bb[i]=t<count?dt[ulong(first+t)*64+h]*conv[ulong(first+t)*6144+h*64+col+i/32]:0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,current);
    uint index=0;for(auto it=current.begin();it!=current.end();++it,++index){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count){
        ulong at=ulong(first+ij[1])*4096+h*64+col+ij[0];float y=*it+prior[index]*exp(decay[off+ij[1]]);
        out[at]=mlx_bf(mlx_bf(y)+mlx_bf(d[h]*conv[ulong(first+ij[1])*6144+h*64+col+ij[0]]));}}
}
// Large prompts reuse one compact BF16 activation plane across Q/K/V and
// across all output tiles. Padded weight pitch avoids repeated bank conflicts.
template<bool Cached,uint BM=64>
inline void nemo_mlx_project_reuse_impl(device const uchar* w,device bfloat* x,
 device float* out,constant uint* p,uint2 g,uint tid,threadgroup bfloat* weights) {
    constexpr uint BN=64,BK=128,PITCH=136;
    auto a=tensor(x,dextents<int,2>(p[0],p[2]),array<int,2>{1,int(p[0])});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,PITCH});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=BK){
        if constexpr(Cached){uint n=g.x*BN+tid/2,k=base+(tid%2)*64,at=tid/2*PITCH+(tid%2)*64;
            if(n<p[1] && k<p[0])nemo_mlx_stage_group<64>(w,p[0],p[1],ulong(n)*p[0]+k,weights+at);
            else for(uint j=0;j<64;j+=4)*reinterpret_cast<threadgroup bfloat4*>(weights+at+j)=bfloat4(0);
        }else for(uint i=tid*4;i<BN*BK;i+=128*4){uint n=g.x*BN+i/BK,k=base+i%BK,at=i/BK*PITCH+i%BK;
            if(n<p[1] && k<p[0])nemo_mlx_stage4(w,p[0],p[1],ulong(n)*p[0]+k,weights+at);
            else *reinterpret_cast<threadgroup bfloat4*>(weights+at)=bfloat4(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(base+BK<=p[0]){auto input=a.slice<BK,BM>(base,g.y*BM);op.run(input,b,acc);}
        else {
            // A tensor slice does not zero out-of-range K loads. Never multiply
            // padded zero weights by unrelated (possibly NaN) input storage.
            constexpr auto tail_desc=matmul2d_descriptor(BM,BN,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<tail_desc,execution_simdgroups<4>> tail_op;
            auto input=a.slice<64,BM>(base,g.y*BM);auto tail=b.template slice<64,BN>(0,0);tail_op.run(input,tail,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint r=g.y*BM+ij[1],n=g.x*BN+ij[0];
        if(it.is_valid_element() && r<p[2] && n<p[1])out[ulong(r)*p[1]+n]=mlx_bf(*it);}
}
#define NEMO_REUSE_PROJECT(NAME,CACHED) \
kernel void NAME(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup bfloat weights[64*136];nemo_mlx_project_reuse_impl<CACHED>(w,x,out,p,g,tid,weights);}
NEMO_REUSE_PROJECT(nemo_mlx_project_reuse,true)
kernel void nemo_mlx_project_reuse128(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup bfloat weights[64*136];nemo_mlx_project_reuse_impl<true,128>(w,x,out,p,g,tid,weights);}
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
NEMO_REUSE_PROJECT(nemo_mlx_project_reuse_baseline,false)
#endif
#undef NEMO_REUSE_PROJECT
template<bool Down>
inline void nemo_mlx_grouped(device const uchar* w,device const float* x,device const uint* lists,device const uint* counts,
 device const uint* tiles,device float* out,constant uint* p,uint2 g,uint tid,threadgroup bfloat* aa,threadgroup bfloat* bb){
    if(g.y>=tiles[0])return;uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(32u,counts[e]-first),nbase=g.x*64;
    auto a=tensor(aa,extents<int,64,32>(),array<int,2>{1,72});auto b=tensor(bb,extents<int,64,64>(),array<int,2>{1,72});
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid*4;i<2048;i+=128*4){uint r=i/64,k=base+i%64,entry=r<count?lists[e*p[2]*6+first+r]:0;
            float4 v=r<count?*reinterpret_cast<device const float4*>(x+ulong(Down?entry:entry/6)*p[0]+k):float4(0);
            *reinterpret_cast<threadgroup bfloat4*>(aa+r*72+i%64)=bfloat4(v);}
        for(uint i=tid*4;i<4096;i+=128*4){uint n=nbase+i/64,k=base+i%64,at=i/64*72+i%64;
            if(n<p[1])nemo_mlx_stage4(w,p[0],p[1]*128,(ulong(e)*p[1]+n)*p[0]+k,bb+at);
            else *reinterpret_cast<threadgroup bfloat4*>(bb+at)=bfloat4(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count && nbase+ij[0]<p[1]){
        uint entry=lists[e*p[2]*6+first+ij[1]];float v=mlx_bf(*it);
        if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));out[ulong(entry)*p[1]+nbase+ij[0]]=v;}}
}
#define NEMO_BF16_EXPERT(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup bfloat aa[32*72],bb[64*72];nemo_mlx_grouped<DOWN>(w,x,lists,counts,tiles,out,p,g,tid,aa,bb);}
NEMO_BF16_EXPERT(nemo_mlx_up_bf16,false)
NEMO_BF16_EXPERT(nemo_mlx_down_bf16,true)
#undef NEMO_BF16_EXPERT
// Compact sorted expert inputs once; every output tile reuses the same BF16
// rows directly from device memory. No padded expert-sized activation arena.
kernel void nemo_mlx_expert_offsets(device const uint* counts [[buffer(0)]],device uint* offsets [[buffer(1)]],uint tid [[thread_index_in_threadgroup]]) {
    uint total=0;for(uint e=0;e<tid;++e)total+=counts[e];offsets[tid]=total;
}
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
kernel void nemo_mlx_route_order(device const uint* lists [[buffer(0)]],device const uint* counts [[buffer(1)]],device uint* order [[buffer(2)]],constant uint* p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]]) {
    uint first=0;for(uint e=0;e<tid;++e)first+=counts[e];
    for(uint i=0;i<counts[tid];++i)order[first+i]=lists[tid*p[0]+i];
}
#endif
kernel void nemo_mlx_expert_pack(device const float* input [[buffer(0)]],device const uint* lists [[buffer(1)]],device const uint* counts [[buffer(2)]],device const uint* offsets [[buffer(3)]],device const uint* tiles [[buffer(4)]],device bfloat* packed [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(p[3],counts[e]-first),i=g.x*256+tid;
    if(i>=count*p[0]/4)return;uint row=i/(p[0]/4),k=(i%(p[0]/4))*4,entry=lists[e*p[1]*6+first+row];
    float4 v=*reinterpret_cast<device const float4*>(input+ulong(p[2]?entry:entry/6)*p[0]+k);
    *reinterpret_cast<device bfloat4*>(packed+ulong(offsets[e]+first+row)*p[0]+k)=bfloat4(v);
}
template<bool Down,uint BM,uint BK=64>
inline void nemo_mlx_expert_reuse(device const uchar* w,device bfloat* x,device const uint* lists,device const uint* counts,device const uint* tiles,device const uint* offsets,device float* out,constant uint* p,uint2 g,uint tid,threadgroup bfloat* weights) {
    if(g.y>=tiles[0])return;uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[e]-first),nbase=g.x*64;
    auto a=tensor(x+ulong(offsets[e]+first)*p[0],dextents<int,2>(p[0],count),array<int,2>{1,int(p[0])});
    auto b=tensor(weights,extents<int,BK,64>(),array<int,2>{1,BK+8});
    constexpr auto desc=matmul2d_descriptor(BM,64,BK,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=BK){
        uint n=nbase+tid/2,k=base+(tid%2)*(BK/2),at=tid/2*(BK+8)+(tid%2)*(BK/2);
        if(n<p[1] && k<p[0])nemo_mlx_stage_group<BK/2>(w,p[0],p[1]*128,(ulong(e)*p[1]+n)*p[0]+k,weights+at);
        else for(uint j=0;j<BK/2;j+=4)*reinterpret_cast<threadgroup bfloat4*>(weights+at+j)=bfloat4(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr(BK==128){
            if(base+BK<=p[0]){auto input=a.template slice<BK,BM>(base,0);op.run(input,b,acc);}
            else {
                constexpr auto tail_desc=matmul2d_descriptor(BM,64,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
                matmul2d<tail_desc,execution_simdgroups<4>> tail_op;
                auto input=a.template slice<64,BM>(base,0);auto tail=b.template slice<64,64>(0,0);tail_op.run(input,tail,acc);
            }
        }else {auto input=a.template slice<BK,BM>(base,0);op.run(input,b,acc);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count && nbase+ij[0]<p[1]){
        uint entry=lists[e*p[2]*6+first+ij[1]];float v=mlx_bf(*it);if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));out[ulong(entry)*p[1]+nbase+ij[0]]=v;}}
}
#define NEMO_REUSE_EXPERT(NAME,DOWN,BM) \
kernel void NAME(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device const uint* offsets [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup bfloat weights[64*72];nemo_mlx_expert_reuse<DOWN,BM>(w,x,lists,counts,tiles,offsets,out,p,g,tid,weights);}
NEMO_REUSE_EXPERT(nemo_mlx_up_reuse,false,32)
NEMO_REUSE_EXPERT(nemo_mlx_down_reuse,true,32)
NEMO_REUSE_EXPERT(nemo_mlx_up_reuse64,false,64)
NEMO_REUSE_EXPERT(nemo_mlx_down_reuse64,true,64)
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
NEMO_REUSE_EXPERT(nemo_mlx_up_reuse128,false,128)
NEMO_REUSE_EXPERT(nemo_mlx_down_reuse128,true,128)
#endif
#define NEMO_REUSE_EXPERT_K128(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device const uint* offsets [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup bfloat weights[64*136];nemo_mlx_expert_reuse<DOWN,64,128>(w,x,lists,counts,tiles,offsets,out,p,g,tid,weights);}
NEMO_REUSE_EXPERT_K128(nemo_mlx_up_reuse64_k128,false)
NEMO_REUSE_EXPERT_K128(nemo_mlx_down_reuse64_k128,true)
#undef NEMO_REUSE_EXPERT_K128
#undef NEMO_REUSE_EXPERT
template<bool Down,uint BM=16,uint BN=32,typename WeightTile=float>
inline void nemo_mlx_grouped_exact(device const uchar* w,device const float* x,device const uint* lists,device const uint* counts,
 device const uint* tiles,device float* out,constant uint* p,uint2 g,uint tid,threadgroup float* aa,threadgroup WeightTile* bb){
    if(g.y>=tiles[0])return;uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[e]-first),nbase=g.x*BN;
    auto a=tensor(aa,extents<int,64,BM>(),array<int,2>{1,64});auto b=tensor(bb,extents<int,64,BN>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(BM,BN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid*4;i<BM*64;i+=128*4){uint r=i/64,k=base+i%64,entry=r<count?lists[e*p[2]*6+first+r]:0;
            *reinterpret_cast<threadgroup float4*>(aa+i)=r<count?*reinterpret_cast<device const float4*>(x+ulong(Down?entry:entry/6)*p[0]+k):float4(0);}
        for(uint i=tid*4;i<BN*64;i+=128*4){uint n=nbase+i/64,k=base+i%64;
            if(n<p[1])nemo_mlx_stage4(w,p[0],p[1]*128,(ulong(e)*p[1]+n)*p[0]+k,bb+i);
            else *reinterpret_cast<threadgroup vec<WeightTile,4>*>(bb+i)=vec<WeightTile,4>(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count && nbase+ij[0]<p[1]){
        uint entry=lists[e*p[2]*6+first+ij[1]];float v=mlx_bf(*it);
        if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));out[ulong(entry)*p[1]+nbase+ij[0]]=v;}}
}
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
template<bool Down>
inline void nemo_mlx_grouped_cooperative(device const uchar* w,device const float* x,device const uint* lists,device const uint* counts,
 device const uint* tiles,device float* out,constant uint* p,uint2 g,uint sg) {
    if(g.y>=tiles[0])return;uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(16u,counts[e]-first),nbase=g.x*128+sg*32;
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto a=op.get_left_input_cooperative_tensor<float,float,float>();
    auto b=op.get_right_input_cooperative_tensor<float,float,float>();
    auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=32){
        for(auto it=a.begin();it!=a.end();++it){auto ij=it.get_multidimensional_index();uint r=ij[1],k=base+ij[0];
            if(it.is_valid_element()){
                uint entry=r<count?lists[e*p[2]*6+first+r]:0;
                *it=r<count && k<p[0]?x[ulong(Down?entry:entry/6)*p[0]+k]:0;
            }}
        for(auto it=b.begin();it!=b.end();++it){auto ij=it.get_multidimensional_index();uint n=nbase+ij[1],k=base+ij[0];
            if(it.is_valid_element())*it=n<p[1] && k<p[0]?mlx_bf(dg_weight(w,p[0],p[1]*128,0x108,(ulong(e)*p[1]+n)*p[0]+k)):0;}
        op.run(a,b,acc);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element() && ij[1]<count && nbase+ij[0]<p[1]){
        uint entry=lists[e*p[2]*6+first+ij[1]];float v=mlx_bf(*it);
        if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));out[ulong(entry)*p[1]+nbase+ij[0]]=v;}}
}
#define NEMO_COOPERATIVE_EXPERT(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]]) {nemo_mlx_grouped_cooperative<DOWN>(w,x,lists,counts,tiles,out,p,g,sg);}
NEMO_COOPERATIVE_EXPERT(nemo_mlx_up_cooperative,false)
NEMO_COOPERATIVE_EXPERT(nemo_mlx_down_cooperative,true)
#undef NEMO_COOPERATIVE_EXPERT
#endif
kernel void nemo_mlx_a(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]=-exp(x[i]);}
// Consecutive four state columns per lane match MLX's decode reduction;
// prefill has separate BF16 CB/skip boundaries and never enters this path.
kernel void nemo_mlx_scan(device const float* conv [[buffer(0)]],device const float* proj [[buffer(1)]],
 device const float* a [[buffer(2)]],device const float* d [[buffer(3)]],device const float* bias [[buffer(4)]],
 device float* state [[buffer(5)]],device const uint* seq [[buffer(6)]],device float* out [[buffer(7)]],
 constant uint* p [[buffer(8)]],uint3 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint slot=seq[4*g.z],first=seq[4*g.z+1],count=seq[4*g.z+2];if(seq[4*g.z+3]!=UINT_MAX && !p[0])return;
    uint h=g.y,col=g.x*4+sg,group=h/8;ulong si=(ulong(slot)*4096+h*64+col)*128;
    float s[4];for(uint j=0;j<4;++j)s[j]=state[si+lane*4+j];
    for(uint t=first;t<first+count;++t){
        float dt=nemo_dt(proj[ulong(t)*10304+10240+h]+bias[h]),decay=exp(a[h]*dt);
        float x=conv[ulong(t)*6144+h*64+col],sum=0;
        for(uint j=0;j<4;++j){uint n=lane*4+j;
            float b=conv[ulong(t)*6144+4096+group*128+n],c=conv[ulong(t)*6144+5120+group*128+n];
            s[j]=fma(decay,s[j],(x*dt)*b);sum+=s[j]*c;}
        sum=simd_sum(sum);if(lane==0)out[ulong(t)*4096+h*64+col]=mlx_bf(sum+d[h]*x);
    }
    for(uint j=0;j<4;++j)state[si+lane*4+j]=s[j];
}
kernel void nemo_mlx_ssd_states(device float* state [[buffer(0)]],device const float* delta [[buffer(1)]],
 device const float* decay [[buffer(2)]],device const uint* seq [[buffer(3)]],device const uint* tiles [[buffer(4)]],
 device float* incoming [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint count=seq[4*g.y+2],first=seq[4*g.y+3];if(first==UINT_MAX)return;
    uint i=g.x*256+tid,slot=seq[4*g.y],h=i/8192;float s=state[ulong(slot)*524288+i];
    for(uint t=first;t<first+(count+31)/32;++t){uint n=tiles[4*t+2];ulong off=ulong(t)*524288+i;
        incoming[off]=s;s=fma(exp(decay[(ulong(t)*64+h)*32+n-1]),s,delta[off]);}
    state[ulong(slot)*524288+i]=s;
}
kernel void nemo_mlx_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],device float* x [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])x[i]=mlx_bf(dg_weight(w,p[0],131072,0x108,ulong(ids[i/p[0]])*p[0]+i%p[0]));
}
kernel void nemo_mlx_conv(device const float* proj [[buffer(0)]],device const float* win [[buffer(1)]],
 device const float* w [[buffer(2)]],device const float* b [[buffer(3)]],device const uint* seq [[buffer(4)]],
 device float* out [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint slot=seq[4*g.y],first=seq[4*g.y+1],count=seq[4*g.y+2],i=g.x*256+tid;
    if(i>=count*6144)return;uint row=i/6144,c=i%6144;float v=0;
    for(uint j=0;j<4;++j){int t=int(row)+int(j)-3;
        float x=t<0?win[(ulong(slot)*3+uint(t+3))*6144+c]:proj[ulong(first+uint(t))*10304+4096+c];v=fma(x,w[c*4+j],v);}
    v=mlx_bf(mlx_bf(v)+b[c]);out[ulong(first+row)*6144+c]=mlx_bf(v*mlx_sigmoid_bf(v));
}
kernel void nemo_mlx_ssd_prepare(device const float* proj [[buffer(0)]],device const float* a [[buffer(1)]],
 device const float* bias [[buffer(2)]],device const uint* tiles [[buffer(3)]],device float* decay [[buffer(4)]],
 device float* dt [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint first=tiles[4*g.y+1],count=tiles[4*g.y+2],h=g.x;float step=lane<count?nemo_dt(proj[ulong(first+lane)*10304+10240+h]+bias[h]):0;
    decay[(ulong(g.y)*64+h)*32+lane]=simd_prefix_inclusive_sum(step*mlx_bf(a[h]));if(lane<count)dt[ulong(first+lane)*64+h]=step;
}
kernel void nemo_mlx_gated_norm(device const float* y [[buffer(0)]],device const float* proj [[buffer(1)]],
 device const float* w [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float sums[8];uint c=g.x*512;ulong at=ulong(g.y)*4096+c;float v[2],sum=0;
    for(uint j=0;j<2;++j){uint d=tid*2+j;float z=proj[ulong(g.y)*10304+c+d];v[j]=mlx_bf(mlx_bf(y[at+d])*mlx_bf(z*mlx_sigmoid_bf(z)));sum+=v[j]*v[j];}
    sum=simd_sum(sum);if(tid%32==0)sums[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=precise::rsqrt(simd_sum(tid%32<8?sums[tid%32]:0)/512+as_type<float>(p[0]));
    for(uint j=0;j<2;++j){uint d=tid*2+j;out[at+d]=mlx_bf(mlx_bf(v[j]*inv)*w[c+d]);}
}
kernel void nemo_mlx_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],device bfloat* kc [[buffer(2)]],device bfloat* vc [[buffer(3)]],device const uint* meta [[buffer(4)]],device const uint* pages [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*256)return;uint row=i/256,pos=meta[row*2+1],slot=meta[row*2];ulong at=(ulong(pages[slot*p[1]+pos/16])*16+pos%16)*256+i%256;kc[at]=bfloat(k[i]);vc[at]=bfloat(v[i]);
}
kernel void nemo_mlx_attention_decode(device const float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[16*32],prob[16*32],highs[16],sums[16];gemma_decode<128,16,true,bfloat>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
kernel void nemo_mlx_prefill_tensor(device bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat kv[32*128];threadgroup float scores[32*32],maximum[32],denom[32],correction[32];
    gemma_prefill<128,32,32,bfloat,false,true,bfloat,true,false,float,true,128>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,scores,scores,maximum,denom,correction);
}
// Reuse the attributed MLX register-fragment primitives already qualified for
// Qwen (qwen_mlx_vision_nax.NOTICE.md). Head-128 needs no cross-SIMD head split:
// each SIMD owns 16 queries, retaining Q, softmax and output in registers. The
// paged KV path never materializes a contiguous context-sized staging buffer.
#ifndef PADDOCK_APPLE9
#define NEMO_UNROLL _Pragma("clang loop unroll(full)")
[[kernel, max_total_threads_per_threadgroup(64)]] void nemo_mlx_prefill_nax(
    device const bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
    device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint first=tiles[2*g.y],count=tiles[2*g.y+1],local=tid/32*16,lane=tid%32;
    if(local>=count)return;
    uint valid=min(16u,count-local),slot=meta[first*2],last=meta[(first+local+valid-1)*2+1];
    uint heads=p[0],kvheads=p[1],kh=g.x/(heads/kvheads),stride=kvheads*128;
    uint r=(lane&16)/4+(lane&6)/2,c=(lane&8)+(lane&1)*4;
    vec<bfloat,8> queries[8];vec<float,8> acc[8];
    NEMO_UNROLL for(short d=0;d<8;++d){
        queries[d]=qmlx_nax::load<false>(q+(ulong(first+local)*heads+g.x)*128+d*16,heads*128,valid,lane);
        acc[d]=vec<float,8>(0);
    }
    float2 hi(-INFINITY),den(0);
    for(uint base=0;base<=last;base+=32){
        uint n0=min(16u,last-base+1),n1=last>=base+16?min(16u,last-base-15):0;
        ulong at0=ulong(pages[slot*p[2]+base/16])*16*stride+kh*128;
        ulong at1=n1?ulong(pages[slot*p[2]+base/16+1])*16*stride+kh*128:at0;
        vec<float,8> score[2]={vec<float,8>(0),vec<float,8>(0)};
        NEMO_UNROLL for(short d=0;d<8;++d){
            auto k0=qmlx_nax::load<false>(k+at0+d*16,stride,n0,lane);
            auto k1=qmlx_nax::load<false>(k+at1+d*16,stride,n1,lane);
            qmlx_nax::mma<bfloat,true>(score[0],score[1],queries[d],k0,k1);
        }
        float2 nh=hi;
        NEMO_UNROLL for(short f=0;f<2;++f)NEMO_UNROLL for(short i=0;i<2;++i){
            uint row=r+i*8,limit=row<valid?meta[(first+local+row)*2+1]:0;
            NEMO_UNROLL for(short j=0;j<4;++j)score[f][i*4+j]=row<valid && base+f*16+c+j<=limit?
                score[f][i*4+j]*(0.08838834764831845f*1.44269504089f):-INFINITY;
            float m=max(max(score[f][i*4],score[f][i*4+1]),max(score[f][i*4+2],score[f][i*4+3]));
            m=max(m,simd_shuffle_xor(m,ushort(1)));m=max(m,simd_shuffle_xor(m,ushort(8)));nh[i]=max(nh[i],m);
        }
        float2 factor;
        NEMO_UNROLL for(short i=0;i<2;++i){
            if(!isfinite(nh[i]))nh[i]=0;
            factor[i]=fast::exp2(hi[i]-nh[i]);hi[i]=nh[i];den[i]*=factor[i];
        }
        NEMO_UNROLL for(short f=0;f<2;++f)NEMO_UNROLL for(short i=0;i<2;++i){
            NEMO_UNROLL for(short j=0;j<4;++j)score[f][i*4+j]=fast::exp2(score[f][i*4+j]-hi[i]);
            float sum=(score[f][i*4]+score[f][i*4+1])+(score[f][i*4+2]+score[f][i*4+3]);
            sum+=simd_shuffle_xor(sum,ushort(1));sum+=simd_shuffle_xor(sum,ushort(8));den[i]+=sum;
        }
        NEMO_UNROLL for(short d=0;d<8;++d)NEMO_UNROLL for(short i=0;i<8;++i)acc[d][i]*=factor[i/4];
        simdgroup_barrier(mem_flags::mem_none);
        NEMO_UNROLL for(short d=0;d<8;d+=2)NEMO_UNROLL for(short f=0;f<2;++f){
            ulong at=f?at1:at0;uint n=f?n1:n0;
            auto v0=qmlx_nax::load<false>(v+at+d*16,stride,n,lane);
            auto v1=qmlx_nax::load<false>(v+at+d*16+16,stride,n,lane);
            qmlx_nax::mma<float,false>(acc[d],acc[d+1],score[f],v0,v1);
        }
    }
    float2 inv=precise::divide(float2(1),den);
    NEMO_UNROLL for(short d=0;d<8;++d)NEMO_UNROLL for(short i=0;i<2;++i)if(r+uint(i)*8<valid)
        NEMO_UNROLL for(short j=0;j<4;++j)
            out[(ulong(first+local+r+i*8)*heads+g.x)*128+d*16+c+j]=mlx_bf(acc[d][i*4+j]*inv[i]);
}
#undef NEMO_UNROLL
#endif
template<bool Down>
inline void nemo_mlx_expert_decode(device const uchar* w,device const float* x,device const uint* ids,device float* out,constant uint* p,uint2 g,uint sg,uint lane) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=g.x*4+sg;if(n>=p[1])return;ulong total=ulong(p[0])*p[1]*128;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+total);device const bfloat* biases=scales+total/64;
    ulong wr=(ulong(ids[g.y])*p[1]+n)*p[0],xr=ulong(Down?g.y:g.y/6)*p[0];float sum=0;
    uint items=p[0]%256==0?8:4;
    for(uint k=lane*items;k<p[0];k+=32*items){float dot=0,xsum=0;
        for(uint j=0;j<items && k+j<p[0];++j){float z=x[xr+k+j];dot+=z*float(w[wr+k+j]);xsum+=z;}
        ulong group=(wr+k)/64;sum+=dot*float(scales[group])+xsum*float(biases[group]);}
    float v=mlx_bf(simd_sum(sum));if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));if(lane==0)out[ulong(g.y)*p[1]+n]=v;
}
#define NEMO_MLX_EXPERT(NAME,DOWN) \
kernel void NAME##_decode(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {nemo_mlx_expert_decode<DOWN>(w,x,ids,out,p,g,sg,lane);} \
kernel void NAME##_grouped(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup float aa[16*64];threadgroup bfloat bb[32*64];nemo_mlx_grouped_exact<DOWN,16,32,bfloat>(w,x,lists,counts,tiles,out,p,g,tid,aa,bb);} \
kernel void NAME##_grouped32(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup float aa[32*64];threadgroup bfloat bb[64*64];nemo_mlx_grouped_exact<DOWN,32,64,bfloat>(w,x,lists,counts,tiles,out,p,g,tid,aa,bb);}
NEMO_MLX_EXPERT(nemo_mlx_up,false)
NEMO_MLX_EXPERT(nemo_mlx_down,true)
#undef NEMO_MLX_EXPERT
template<bool Down,uint Items,bool Sorted=false,uint Cols=4>
inline void nemo_mlx_expert_decode_reuse(device const uchar* w,device const float* x,device const uint* ids,device float* out,device const uint* order,constant uint* p,uint2 g,uint sg,uint lane) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=(g.x*4+sg)*Cols;if(n>=p[1])return;ulong total=ulong(p[0])*p[1]*128;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+total);device const bfloat* biases=scales+total/64;
    uint entry=Sorted?order[g.y]:g.y;
    ulong wr=(ulong(ids[entry])*p[1]+n)*p[0],xr=ulong(Down?entry:entry/6)*p[0];float sums[Cols];for(uint c=0;c<Cols;++c)sums[c]=0;
    for(uint k=lane*Items;k<p[0];k+=32*Items){
        float4 values[Items/4];float xsum=0;
        for(uint j=0;j<Items/4;++j){values[j]=*reinterpret_cast<device const float4*>(x+xr+k+j*4);for(uint i=0;i<4;++i)xsum+=values[j][i];}
        for(uint c=0;c<Cols;++c){if(n+c>=p[1])continue;ulong at=wr+ulong(c)*p[0]+k;float dot=0;
            for(uint j=0;j<Items/4;++j){float4 codes=float4(*reinterpret_cast<device const uchar4*>(w+at+j*4));for(uint i=0;i<4;++i)dot+=values[j][i]*codes[i];}
            sums[c]+=dot*float(scales[at/64])+xsum*float(biases[at/64]);}
    }
    for(uint c=0;c<Cols;++c){float v=mlx_bf(simd_sum(sums[c]));if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));if(lane==0 && n+c<p[1])out[ulong(entry)*p[1]+n+c]=v;}
}
#define NEMO_REUSE_DECODE(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {if(p[0]%256==0)nemo_mlx_expert_decode_reuse<DOWN,8>(w,x,ids,out,nullptr,p,g,sg,lane);else nemo_mlx_expert_decode_reuse<DOWN,4>(w,x,ids,out,nullptr,p,g,sg,lane);}
NEMO_REUSE_DECODE(nemo_mlx_up_decode_reuse,false)
NEMO_REUSE_DECODE(nemo_mlx_down_decode_reuse,true)
#undef NEMO_REUSE_DECODE
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
#define NEMO_COLUMNS_DECODE(NAME,DOWN,COLS) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {if(p[0]%256==0)nemo_mlx_expert_decode_reuse<DOWN,8,false,COLS>(w,x,ids,out,nullptr,p,g,sg,lane);else nemo_mlx_expert_decode_reuse<DOWN,4,false,COLS>(w,x,ids,out,nullptr,p,g,sg,lane);}
NEMO_COLUMNS_DECODE(nemo_mlx_up_decode_cols2,false,2)
NEMO_COLUMNS_DECODE(nemo_mlx_down_decode_cols2,true,2)
NEMO_COLUMNS_DECODE(nemo_mlx_up_decode_cols8,false,8)
NEMO_COLUMNS_DECODE(nemo_mlx_down_decode_cols8,true,8)
NEMO_COLUMNS_DECODE(nemo_mlx_up_decode_cols16,false,16)
NEMO_COLUMNS_DECODE(nemo_mlx_down_decode_cols16,true,16)
#undef NEMO_COLUMNS_DECODE
#endif
// Reuse each affine weight group across up to four routed entries without
// changing the original per-entry SIMD reduction or BF16 graph boundaries.
template<bool Down,uint Items,uint Rows=4>
inline void nemo_mlx_expert_vector_group4(device const uchar* w,device const float* x,device const uint* lists,device const uint* counts,device const uint* tiles,device float* out,constant uint* p,uint2 g,uint sg,uint lane) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if(g.y>=tiles[0])return;
    uint e=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(Rows,counts[e]-first),n=(g.x*4+sg)*2;
    if(n>=p[1])return;
    uint entries[Rows];for(uint r=0;r<Rows;++r)entries[r]=r<count?lists[e*p[2]*6+first+r]:0;
    ulong total=ulong(p[0])*p[1]*128,wr=(ulong(e)*p[1]+n)*p[0];
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+total);device const bfloat* biases=scales+total/64;
    float sums[Rows][2];for(uint r=0;r<Rows;++r)for(uint c=0;c<2;++c)sums[r][c]=0;
    for(uint k=lane*Items;k<p[0];k+=32*Items){
        float4 codes[2][Items/4];float ss[2],bb[2];
        for(uint c=0;c<2;++c){ulong at=wr+ulong(c)*p[0]+k;
            for(uint j=0;j<Items/4;++j)codes[c][j]=n+c<p[1]?float4(*reinterpret_cast<device const uchar4*>(w+at+j*4)):float4(0);
            ss[c]=n+c<p[1]?float(scales[at/64]):0;bb[c]=n+c<p[1]?float(biases[at/64]):0;}
        for(uint r=0;r<Rows;++r){if(r>=count)continue;ulong xr=ulong(Down?entries[r]:entries[r]/6)*p[0];float dot[2]={0,0},xsum=0;
            for(uint j=0;j<Items/4;++j){float4 v=*reinterpret_cast<device const float4*>(x+xr+k+j*4);
                for(uint i=0;i<4;++i){for(uint c=0;c<2;++c)dot[c]+=v[i]*codes[c][j][i];xsum+=v[i];}}
            for(uint c=0;c<2;++c)sums[r][c]+=dot[c]*ss[c]+xsum*bb[c];
        }
    }
    for(uint r=0;r<Rows;++r)for(uint c=0;c<2;++c){float v=mlx_bf(simd_sum(sums[r][c]));if(!Down)v=mlx_bf(max(v,0.0f)*max(v,0.0f));if(lane==0 && r<count && n+c<p[1])out[ulong(entries[r])*p[1]+n+c]=v;}
}
#define NEMO_VECTOR_GROUP4(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {if(p[0]%256==0)nemo_mlx_expert_vector_group4<DOWN,8>(w,x,lists,counts,tiles,out,p,g,sg,lane);else nemo_mlx_expert_vector_group4<DOWN,4>(w,x,lists,counts,tiles,out,p,g,sg,lane);}
NEMO_VECTOR_GROUP4(nemo_mlx_up_vector_group4,false)
NEMO_VECTOR_GROUP4(nemo_mlx_down_vector_group4,true)
#undef NEMO_VECTOR_GROUP4
#define NEMO_VECTOR_GROUP8(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {if(p[0]%256==0)nemo_mlx_expert_vector_group4<DOWN,8,8>(w,x,lists,counts,tiles,out,p,g,sg,lane);else nemo_mlx_expert_vector_group4<DOWN,4,8>(w,x,lists,counts,tiles,out,p,g,sg,lane);}
NEMO_VECTOR_GROUP8(nemo_mlx_up_vector_group8,false)
NEMO_VECTOR_GROUP8(nemo_mlx_down_vector_group8,true)
#undef NEMO_VECTOR_GROUP8
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
#define NEMO_SORTED_DECODE(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],device const uint* order [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {if(p[0]%256==0)nemo_mlx_expert_decode_reuse<DOWN,8,true>(w,x,ids,out,order,p,g,sg,lane);else nemo_mlx_expert_decode_reuse<DOWN,4,true>(w,x,ids,out,order,p,g,sg,lane);}
NEMO_SORTED_DECODE(nemo_mlx_up_decode_sorted,false)
NEMO_SORTED_DECODE(nemo_mlx_down_decode_sorted,true)
#undef NEMO_SORTED_DECODE
#endif
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
#define NEMO_MLX_REFERENCE(NAME,DOWN) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup float aa[16*64],bb[32*64];nemo_expert_grouped<DOWN,0x108>(w,x,lists,counts,tiles,out,p,g,tid,aa,bb);}
NEMO_MLX_REFERENCE(nemo_mlx_up_f32_reference,false)
NEMO_MLX_REFERENCE(nemo_mlx_down_f32_reference,true)
#undef NEMO_MLX_REFERENCE
#endif
kernel void nemo_mlx_relu2(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {if(i<p[0]){float v=max(x[i],0.0f);x[i]=mlx_bf(v*v);}}
kernel void nemo_mlx_fold(device const float* out [[buffer(0)]],device const float* weights [[buffer(1)]],device float* shared [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(i>=p[0]*2688)return;uint row=i/2688,n=i%2688;float v=0;for(uint j=0;j<6;++j)v+=out[(ulong(row)*6+j)*2688+n]*weights[row*6+j];shared[i]=mlx_bf(shared[i]+mlx_bf(v));
}
kernel void nemo_mlx_rms_selected(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* rows [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[32];uint n=p[0],threads=((n+127)/128)*32;ulong b=ulong(rows[row])*n;
    float inv=gmlx_inv(x+b,n,as_type<float>(p[2]),tid,threads,sums);for(uint i=tid;i<n;i+=threads)out[ulong(row)*n+i]=mlx_bf(mlx_bf(x[b+i]*inv)*w[i]);
}
