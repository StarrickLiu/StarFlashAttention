#include <src/flash_attn.h>
// #include <src/utils.h>

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA_16B(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG_16B(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CA_8B(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CA_4B(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

template <typename T, int bytes_per_thread>
__device__ void cp_async_ca(T *dst, const T *src);
template <typename T, int bytes_per_thread>
__device__ void cp_async_cg(T *dst, const T *src);

__device__ half float2T(float x)
{
    return __float2half(x);
}

__device__ float T2float(half x)
{
    return __half2float(x);
}

// __device__ half warpReduceMax(half val, int warpSize);
// __device__ half warpReduceSum(half val, int warpSize);
// __device__ float warpReduceMax(float val, int warpSize);
// __device__ float warpReduceSum(float val, int warpSize);
// __device__ float half_to_float(uint16_t h);
// __device__ float2 half2_to_float2(uint32_t v);
// __device__ uint32_t float2_to_half2(float2 f);
// __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step);
// __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef);
// __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef);
// __device__ void apply_rotary_embedding(uint2 &q, uint2 &k, int tid, int rot_embed_dim, int t_step, half* rotary_cos, half* rotary_sin);
// __device__ void apply_rotary_embedding(uint2 &q, int tid, int rot_embed_dim, int t_step, half* rotary_cos, half* rotary_sin);
// __device__ void apply_rotary_embedding(uint4 &q, uint4 &k, int tid, int rot_embed_dim, int t_step, half* rotary_cos, half* rotary_sin);
// __device__ void apply_rotary_embedding(uint4 &q, int tid, int rot_embed_dim, int t_step, half* rotary_cos, half* rotary_sin);
// __device__ void gemv_qk(half *sQ, half *gK_cache, half *sK_cache, half *sK, float *rS, int n_elem_per_blockN, int head_dim, int n_block, int memory_max_len, int tidx, int n_elem_per_thread, float head_dim_inv, bool last_block);
// __device__ void gemv_pv(half *rP, half *sV, half *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, bool is_first, int m1_in_formula, int m2_in_formula);
// __device__ void clear_shm(half *p, const int n);
// __device__ void clear_reg(half *p, const int n);
// __device__ void clear_reg(float *p, const int n);
// __device__ half __max(half a, half b);
// __device__ half __exp(half x);
// __device__ float __max(float a, float b);
// __device__ float __exp(float x);
// __device__ half float2T(float x);
// __device__ float T2float(half x);

#define N_ELEM_PER_CPY 8
#define WARP_SIZE 32
// * async tools

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
template <>
__forceinline__ __device__ void cp_async_ca<half, 16>(half *dst, const half *src)
{
    uint32_t dst_ = __cvta_generic_to_shared(dst);
    CP_ASYNC_CA_16B(dst_, src, 16);
}

template <>
__forceinline__ __device__ void cp_async_cg<half, 16>(half *dst, const half *src)
{
    uint32_t dst_ = __cvta_generic_to_shared(dst);
    CP_ASYNC_CG_16B(dst_, src, 16);
}
#else
template <>
__device__ void cp_async_ca<half, 16>(half *dst, const half *src)
{
    CP_ASYNC_CA(dst_, src, 16);
}

template <>
__device__ void cp_async_cg<half, 16>(half *dst, const half *src)
{
    CP_ASYNC_CG(dst_, src, 16);
}
#endif
template <>
__forceinline__ __device__ void cp_async_ca<half, 8>(half *dst, const half *src)
{
    uint32_t dst_ = __cvta_generic_to_shared(dst);
    CP_ASYNC_CA_8B(dst_, src, 8);
}

template <>
__forceinline__ __device__ void cp_async_ca<half, 4>(half *dst, const half *src)
{
    uint32_t dst_ = __cvta_generic_to_shared(dst);
    CP_ASYNC_CA_4B(dst_, src, 4);
}

// * For FP16

__forceinline__ __device__ half warpReduceMax(half val, int warpSize)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = __hmax(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__forceinline__ __device__ half warpReduceSum(half val, int warpSize)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__forceinline__ __device__ float half_to_float(uint16_t h)
{
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

__forceinline__ __device__ float2 half2_to_float2(uint32_t v)
{
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
    return make_float2(half_to_float(lo), half_to_float(hi));
}

__forceinline__ __device__ uint32_t float2_to_half2(float2 f)
{
    union
    {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
    return tmp.u32;
}

__forceinline__ __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step)
{
    const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

__forceinline__ __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

__forceinline__ __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
{
    float2 fv = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
}

__forceinline__ __device__ uint32_t rotary_embedding_transform(const uint32_t v, const uint32_t coef)
{
    float2 fv = half2_to_float2(v);
    float2 fcoef = half2_to_float2(coef);
    float2 rot_fv = rotary_embedding_transform(fv, fcoef);
    return float2_to_half2(rot_fv);
}

// 一个线程负责四个元素的旋转编码
__forceinline__ __device__ void apply_rotary_embedding(uint2 &q, uint2 &k, const int tid, const int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
{
    if (4 * tid >= rot_embed_dim)
    {
        return;
    }

    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    k.x = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    k.y = rotary_embedding_transform(k.y, coef1);
}

// 一个线程负责四个元素的旋转编码
__forceinline__ __device__ void apply_rotary_embedding(uint2 &q, const int tid, const int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
{
    if (4 * tid >= rot_embed_dim)
    {
        return;
    }
    half rCOS[2] = {rotary_cos[0], rotary_cos[1]};
    half rSIN[2] = {rotary_sin[0], rotary_sin[1]};
    // const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, {rCOS[0], rSIN[0]});
    // const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, {rCOS[1], rSIN[1]});
}

// 一个线程负责八个元素的旋转编码
__forceinline__ __device__ void apply_rotary_embedding(uint4 &q, uint4 &k, int tid, int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
{
    if (8 * tid >= rot_embed_dim)
    {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    k.x = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    k.y = rotary_embedding_transform(k.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z = rotary_embedding_transform(q.z, coef2);
    k.z = rotary_embedding_transform(k.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w = rotary_embedding_transform(q.w, coef3);
    k.w = rotary_embedding_transform(k.w, coef3);
}

// 一个线程负责八个元素的旋转编码
__forceinline__ __device__ void apply_rotary_embedding(uint4 &q, int tid, int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
{
    if (8 * tid >= rot_embed_dim)
    {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z = rotary_embedding_transform(q.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w = rotary_embedding_transform(q.w, coef3);
}

// for fp16 gemv

__forceinline__ __device__ void gemv_qk(half *sQ, half *gK_cache, half *sK_cache, half *sK, float *rS, int actual_seq_len, int n_elem_per_blockN, int head_dim, int n_block, int memory_max_len, int tidx, int n_elem_per_thread, float head_dim_inv, bool *mask, bool last_block)
{
    // (uint4) sK (n/8, k/8) (uint4) sQ (k/8) (uint4) rS (n/8)
    // 共计算n_elem_per_thread个元素的结果，偏移量为tidx*n_elem_per_thread，确保k一定可以被8整除
    // 找到对应的矩阵行后，沿着k方向循环，每次向量化拷贝8个Vec和n_elem_per_thread*8个Mat的元素，相乘后结果存储到rS，注意Mat不要超出n的范围
    // 修改后逻辑：
    // gK_cache的布局为(headdim / n_elem_per_cpy, memory_max_len, n_elem_per_cpy)
    // 需要取(headdim/n_elem_per_cpy, n_block*n_elem_per_blockN - (n_block+1)*n_elem_per_blockN, n_elem_per_cpy)
    // sK_cache: (head_dim / n_elem_per_cpy, n_elem_per_blockN, n_elem_per_cpy) sK: (head_dim)
    // gK_cache逐元素异步流水线拷贝到sK_cache中
    // 每一轮各线程计算一个元素的结果，结果存储到rS中，若headDim=128，则每个线程计算4个元素的结果，且4个元素是散布的
    // exm: Thr0 0-32-64-96,Thr1 1-33-65-97,..., Thr31 31-63-95-127
    // 如果是最后一个split的最后一个block的情况，需要单独计算最新的rS，使用一个判断给到对应的线程（会导致Warp分叉性能损耗）

    // 每个线程Q在HeadDim维度上的循环
    int offset_dst = tidx * N_ELEM_PER_CPY;
    int offset_src = (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;
    // int count = 0;
#pragma unroll
    for (int i = 0; i < head_dim / N_ELEM_PER_CPY; i++)
    {
        offset_dst = i * n_elem_per_blockN * N_ELEM_PER_CPY + tidx * N_ELEM_PER_CPY;
        offset_src = i * memory_max_len * N_ELEM_PER_CPY + (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;
#pragma unroll
        for (int j = 0; j < n_elem_per_thread; j++)
        {
            cp_async_ca<half, 16>(sK_cache + offset_dst + j * WARP_SIZE * N_ELEM_PER_CPY,
                                  gK_cache + offset_src + j * WARP_SIZE * N_ELEM_PER_CPY);
        }
    }
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_ALL();

#pragma unroll
    for (int i = 0; i < head_dim / N_ELEM_PER_CPY; i++)
    {
        // offset_dst = i * n_elem_per_blockN * N_ELEM_PER_CPY + tidx * N_ELEM_PER_CPY;
        // offset_src = i * memory_max_len * N_ELEM_PER_CPY + (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;

        // ShareMemory的4个Bank执行广播到所有线程
        uint4 sQ_tmp = reinterpret_cast<uint4 *>(sQ)[i];
        half2 *sQ_h1 = (half2 *)&sQ_tmp.x;
        half2 *sQ_h2 = (half2 *)&sQ_tmp.y;
        half2 *sQ_h3 = (half2 *)&sQ_tmp.z;
        half2 *sQ_h4 = (half2 *)&sQ_tmp.w;
#pragma unroll
        for (int j = 0; j < n_elem_per_thread; j++)
        {
            uint4 sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN + j * WARP_SIZE + tidx];
            half2 *sK_h1 = (half2 *)&sK_tmp.x;
            half2 *sK_h2 = (half2 *)&sK_tmp.y;
            half2 *sK_h3 = (half2 *)&sK_tmp.z;
            half2 *sK_h4 = (half2 *)&sK_tmp.w;
            rS[j] += T2float(sK_h1->x * sQ_h1->x);
            rS[j] += T2float(sK_h1->y * sQ_h1->y);
            rS[j] += T2float(sK_h2->x * sQ_h2->x);
            rS[j] += T2float(sK_h2->y * sQ_h2->y);
            rS[j] += T2float(sK_h3->x * sQ_h3->x);
            rS[j] += T2float(sK_h3->y * sQ_h3->y);
            rS[j] += T2float(sK_h4->x * sQ_h4->x);
            rS[j] += T2float(sK_h4->y * sQ_h4->y);
        }
    }
#pragma unroll
    for (int i = 0; i < n_elem_per_thread; i++)
    {
        rS[i] = rS[i] * head_dim_inv;
    }

    if (last_block)
    {
        uint2 rQ = reinterpret_cast<uint2 *>(sQ)[tidx];
        uint2 rK = reinterpret_cast<uint2 *>(sK)[tidx];
        half2 *rQ_h1 = (half2 *)&rQ.x;
        half2 *rQ_h2 = (half2 *)&rQ.y;
        half2 *rK_h1 = (half2 *)&rK.x;
        half2 *rK_h2 = (half2 *)&rK.y;
        float tmpS = T2float(rK_h1->x * rQ_h1->x + rK_h1->y * rQ_h1->y + rK_h2->x * rQ_h2->x + rK_h2->y * rQ_h2->y);
        int last_idx = actual_seq_len % WARP_SIZE;
        if (tidx == last_idx)
        {
            rS[actual_seq_len / WARP_SIZE] = tmpS;
        }
    }
}

__device__ void gemv_pv(half *sP, half *gV_cache, half *sV_cache, half *sV, float *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, int n_elem_per_blockN, int head_dim, int n_block, int memory_max_len, bool is_first, int m1_in_formula, int m2_in_formula)
{
    // gVCache: (headdim / n_elem_per_cpy, memory_max_len, n_elem_per_cpy)
    // rP: (4) ...[tidx*n_elem_per_thread, (tidx+1)*n_elem_per_thread) located in P [seqLen]
    // sV: (headdim / n_elem_per_cpy, n_elem_per_blockN, n_elem_per_cpy)
    // rO: (4)
    int offset_dst = tidx * N_ELEM_PER_CPY;
    int offset_src = (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;
    // int count = 0;
#pragma unroll
    for (int i = 0; i < head_dim / N_ELEM_PER_CPY; i++)
    {
        offset_dst = i * n_elem_per_blockN * N_ELEM_PER_CPY + tidx * N_ELEM_PER_CPY;
        offset_src = i * memory_max_len * N_ELEM_PER_CPY + (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;
#pragma unroll
        for (int j = 0; j < n_elem_per_thread; j++)
        {
            cp_async_ca<half, 16>(sV_cache + offset_dst + j * WARP_SIZE * N_ELEM_PER_CPY,
                                  gV_cache + offset_src + j * WARP_SIZE * N_ELEM_PER_CPY);
        }
    }
    CP_ASYNC_COMMIT_GROUP();

    // 非第一次的情况，先修正原先的O
    if (!is_first)
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            rO[i] = rO[i] * exp(m1_in_formula - m2_in_formula);
        }
    }
    CP_ASYNC_WAIT_ALL();
    //     // 每个线程每次沿H维度向量化读取sV的8个元素，读取4行，共读取32个元素进入寄存器
    //     // 使用Warp快速通信的方式将结果累加到rO中
    //     for (int u = 0; u < headDim / N_ELEM_PER_CPY; u ++)
    //     {
    //         float buf[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //         for (int i = 0; i < n_elem_per_thread; i++)
    //         {
    //             // uint4 sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN + j * WARP_SIZE + tidx];
    //             uint4 rV = reinterpret_cast<uint4 *>(sV)[u * n_elem_per_blockN + i * WARP_SIZE + tidx];
    //             half2 *rV_h1 = (half2 *)&rV.x;
    //             half2 *rV_h2 = (half2 *)&rV.y;
    //             half2 *rV_h3 = (half2 *)&rV.z;
    //             half2 *rV_h4 = (half2 *)&rV.w;
    //             // buf[0] += T2float(sP[i] * rV_h1->x);
    //             // buf[1] += T2float(rP[i] * rV_h1->y);
    //             // buf[2] += T2float(rP[i] * rV_h2->x);
    //             // buf[3] += T2float(rP[i] * rV_h2->y);
    //             // buf[4] += T2float(rP[i] * rV_h3->x);
    //             // buf[5] += T2float(rP[i] * rV_h3->y);
    //             // buf[6] += T2float(rP[i] * rV_h4->x);
    //             // buf[7] += T2float(rP[i] * rV_h4->y);
    //         }
    //         // 规约当前warp上的buf，广播到每个线程中，如果是当前线程负责的元素，则相加到寄存器rO上
    //         // for (int i = 0; i < N_ELEM_PER_CPY; ++i)
    //         // {
    //         //     buf[i] = warpReduceSum(buf[i], 32);
    //         // }
    //         // if (tidx == 2 * u)
    //         // {
    //         //     rO[0] += buf[0];
    //         //     rO[1] += buf[1];
    //         //     rO[2] += buf[2];
    //         //     rO[3] += buf[3];
    //         // }
    //         // if (tidx == 2 * u + 1)
    //         // {
    //         //     rO[0] += buf[4];
    //         //     rO[1] += buf[5];
    //         //     rO[2] += buf[6];
    //         //     rO[3] += buf[7];
    //         // }
    //     }
    // sVCache: (n_elem_per_blockN, headdim)

#pragma unroll
    for (int i = 0; i < n_elem_per_blockN; i += N_ELEM_PER_CPY)
    {
        uint4 rP_tmp = reinterpret_cast<uint4 *>(sP)[i];
        half *rP = reinterpret_cast<half*>(&rP_tmp);  // 将rP_tmp的地址转换为指向half的指针

#pragma unroll
        for (int k = 0; k < N_ELEM_PER_CPY; k++)
        {
            int offset = (i + k) * headDim;
#pragma unroll
            for (int j = 0; j < n_elem_per_thread; j++)
            {
                uint2 rV = {reinterpret_cast<uint2 *>(sV_cache + offset)[j]};
                half2 *rV_h1 = (half2 *)&rV.x;
                half2 *rV_h2 = (half2 *)&rV.y;
                rO[0] += T2float(rP[k] * rV_h1->x);
                rO[1] += T2float(rP[k] * rV_h1->y);
                rO[2] += T2float(rP[k] * rV_h2->x);
                rO[3] += T2float(rP[k] * rV_h2->y);
            }
        }
    }
}

__device__ void clear_shm(half *p, const int n)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        p[i] = 0;
    }
}

__device__ void clear_reg(half *p, const int n)
{
    for (int i = 0; i < n; i++)
    {
        p[i] = 0;
    }
}

__device__ void clear_reg(float *p, const int n)
{
    for (int i = 0; i < n; i++)
    {
        p[i] = 0;
    }
}

__device__ half __max(half a, half b)
{
    return a > b ? a : b;
}

__device__ half __exp(half x)
{
    return hexp(x);
}

__device__ float __max(float a, float b)
{
    return a > b ? a : b;
}

__device__ float __exp(float x)
{
    return exp(x);
}

__global__ void init_half_array_kernel(half *array, half value, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        array[idx] = value;
    }
}

void init_half_array(half *array, half value, int n, int numBlocks, int blockSize)
{
    init_half_array_kernel<<<numBlocks, blockSize>>>(array, value, n);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Init CUDA kernel error: %s\n", cudaGetErrorString(error));
    }
}

__global__ void rotary_table_kernel(half *rotary_cos_table, half *rotary_sin_table, int max_seq_len, int rot_embed_dim)
{
    int cur_seq_len = blockIdx.x;
    int zid = threadIdx.x * 2;
    float freq = cur_seq_len / pow(10000.0f, zid / (float)rot_embed_dim);
    rotary_cos_table[cur_seq_len * rot_embed_dim + threadIdx.x] = __float2half(cosf(freq));
    rotary_sin_table[cur_seq_len * rot_embed_dim + threadIdx.x] = __float2half(sinf(freq));
    // if (zid < rot_embed_dim / 2)
    // { // 保证只有有效的zid参与计算
    //     float freq = powf(10000.0f, -((float)zid / rot_embed_dim));
    //     for (int pos = 0; pos < max_seq_len; pos++)
    //     {
    //         float angle = freq * pos;
    //         rotary_cos_table[pos * rot_embed_dim + zid] = __float2half(cosf(angle));
    //         rotary_sin_table[pos * rot_embed_dim + zid] = __float2half(sinf(angle));
    //     }
    // }
}

template <>
void compute_rotary_table(half *rotary_cos_table, half *rotary_sin_table, int max_seq_len, int rot_embed_dim)
{
    int threadsPerBlock = rot_embed_dim / 2;
    int blocksPerGrid = max_seq_len;

    rotary_table_kernel<<<blocksPerGrid, threadsPerBlock>>>(rotary_cos_table, rotary_sin_table, max_seq_len, rot_embed_dim);
}

__forceinline__ __device__ float warpReduceMax(float val, int warpSize)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = __max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__forceinline__ __device__ float warpReduceSum(float val, int warpSize)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <typename T, typename Traits>
__global__ void flash_decoder_kernel(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers)
{

    extern __shared__ char smem[];
    const int memory_max_len = input.memory_max_len;
    const int num_heads = input.num_heads;
    const int head_dim = input.head_dim;
    const int qkv_stride = input.stride;
    const int num_layer = input.num_layer;
    const int idx_layer = input.idx_layer;
    const int rot_embed_dim = input.rotary_embedding_dim;
    const int bidb = blockIdx.z / num_heads;
    const int bidh = blockIdx.z - bidb * num_heads;
    const int tidx = threadIdx.x;

    const int seq_len = reinterpret_cast<int *>(input.seq_len)[bidb];
    const int padding_seq_len = seq_len + 1; // 加上当前的Token

    const int n_split_idx = blockIdx.y;
    const int num_splits = params.num_splits;
    bool last_split_is_first = n_split_idx == num_splits - 1;
    bool is_first = true;

    const int n_elem_per_blockN = params.kBlockN; // 128

    const int n_blockN = (padding_seq_len + n_elem_per_blockN - 1) / n_elem_per_blockN;
    const int n_block_per_split = (n_blockN + num_splits - 1) / num_splits;

    // 定位到起始和结束Block
    const int n_block_min = n_split_idx * n_block_per_split;
    const int n_block_max = std::min(n_blockN, (n_split_idx + 1) * n_block_per_split);
    int n_block = n_block_max - 1;

    // 获得当前batch,head下的qkv的起始位置
    // qkv_stride == 3 * num_heads * head_dim
    // QKV (batch_size, 3, num_heads, head_dim)
    const uint32_t offset_q = bidb * qkv_stride + bidh * head_dim;
    const uint32_t offset_k = bidb * qkv_stride + num_heads * head_dim + bidh * head_dim;
    const uint32_t offset_v = bidb * qkv_stride + 2 * num_heads * head_dim + bidh * head_dim;
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("[DEBUG] Here3 \n");
    T *gQ = reinterpret_cast<T *>(input.qkv) + offset_q;
    T *gK = reinterpret_cast<T *>(input.qkv) + offset_k;
    T *gV = reinterpret_cast<T *>(input.qkv) + offset_v;

    T *rotary_cos = reinterpret_cast<T *>(input.rotary_cos_table) + seq_len * head_dim;
    T *rotary_sin = reinterpret_cast<T *>(input.rotary_sin_table) + seq_len * head_dim;

    T *gO;
    if (num_splits == 1)
    {
        // o_split (batch_size, num_heads, head_dim)
        const uint32_t offset_o = bidb * num_heads * head_dim + bidh * head_dim;
        gO = reinterpret_cast<T *>(input.o) + offset_o;
    }
    else
    {
        // o_split (batch_size, num_heads, num_splits, head_dim)
        const uint32_t offset_o = bidb * num_heads * num_splits * head_dim + bidh * num_splits * head_dim + n_split_idx * head_dim;
        gO = reinterpret_cast<T *>(buffers.o_split) + offset_o;
    }

    const uint32_t offset_cache = bidb * num_layer * num_heads * head_dim * memory_max_len +
                                  idx_layer * num_heads * head_dim * memory_max_len +
                                  bidh * head_dim * memory_max_len;
    T *gKCache = reinterpret_cast<T *>(input.k_cache_table) + offset_cache;
    T *gVCache = reinterpret_cast<T *>(input.v_cache_table) + offset_cache;

    T *sQ = reinterpret_cast<T *>(smem);            // (1, head_dim)
    T *sK = sQ + head_dim;                          // (1, head_dim)
    T *sV = sK + head_dim;                          // (1, head_dim)
    T *sO = sV + head_dim;                          // (1, head_dim)
    T *sKCache = sO + head_dim;                     // (n_elem_per_blockN, head_dim)
    T *sVCache = sK + n_elem_per_blockN * head_dim; // (n_elem_per_blockN, head_dim)
    int offset_rotary = tidx * 4;

    cp_async_ca<T, 4>(sKCache, rotary_cos + offset_rotary / 2);
    cp_async_ca<T, 4>(sVCache, rotary_sin + offset_rotary / 2);
    CP_ASYNC_COMMIT_GROUP();
    cp_async_ca<T, 8>(sQ + offset_rotary, gQ + offset_rotary);
    CP_ASYNC_COMMIT_GROUP();

    // TODO: 根据模板变
    bool last_block_qv_mask[4];
    if (last_split_is_first)
    {
        int last_seq_len = seq_len % n_elem_per_blockN;
        last_block_qv_mask[0] = last_seq_len / tidx > 0;
        last_block_qv_mask[1] = (last_seq_len - WARP_SIZE) / tidx > 0;
        last_block_qv_mask[2] = (last_seq_len - 2 * WARP_SIZE) / tidx > 0;
        last_block_qv_mask[3] = (last_seq_len - 3 * WARP_SIZE) / tidx > 0;
    }

    const int n_elem_per_thread = Traits::n_elem_per_thread;

    float rS[Traits::n_elem_per_thread];
    half rP_1[Traits::n_elem_per_thread];
    half rP_2[Traits::n_elem_per_thread];
    float rO[Traits::n_elem_per_thread];
    clear_reg(rO, n_elem_per_thread);
    float rEll_1 = 0;
    float rEll_2 = 0;
    float rEll_2_inv = 0;

    float m1_in_formula;
    float m2_in_formula;

    // TODO: 添加偏置
    // Convert gQ and gK to uint2 pointers (向量化读取)
    // uint2 *gQ_uint2 = reinterpret_cast<uint2 *>(gQ);
    // uint2 *gK_uint2 = reinterpret_cast<uint2 *>(gK);
    // uint2 *gV_uint2 = reinterpret_cast<uint2 *>(gV);

    if (rot_embed_dim != 0 && n_split_idx == num_splits - 1)
    {
        // 对于最后一个spilit，先将gQ和gK向量化拷贝到smem的特定位置后，再进行旋转编码，在后面直接使用，原因是内核写回全局内存有延迟，存入后再读数据不正确
        // 顺手把相应的gV也拷贝到smem特定位置并存入gVCache中
        // if (tidx * 8 < head_dim)
        // {
        // printf("offset: %d, max_offset: %d cur_len: %d, n_elem_per_blockN: %d\n", cur_len * head_dim + tidx * 8, 2 * n_elem_per_blockN * head_dim, cur_len, n_elem_per_blockN);
        // reinterpret_cast<uint2 *>(sK)[tidx] = gK_uint2[tidx];
        cp_async_ca<T, 8>(sK + offset_rotary, gK + offset_rotary);
        CP_ASYNC_COMMIT_GROUP();
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31 && tidx == 0)
        // {
        //     printf("[DEBUG] before rotary: %f %f %f %f\n",
        //            __half2float(sQ[tidx * 4]), __half2float(sQ[tidx * 4 + 1]), __half2float(sQ[tidx * 4 + 2]), __half2float(sQ[tidx * 4 + 3]));
        // }
        // TODO: 其实可以直接把sQ和sK拷到寄存器？
        CP_ASYNC_WAIT_GROUP(1);
        apply_rotary_embedding(reinterpret_cast<uint2 *>(sQ)[tidx], tidx, rot_embed_dim, seq_len, rotary_cos, rotary_sin);
        CP_ASYNC_WAIT_ALL();
        apply_rotary_embedding(reinterpret_cast<uint2 *>(sK)[tidx], tidx, rot_embed_dim, seq_len, rotary_cos, rotary_sin);
        // gQ_uint2[tidx] = reinterpret_cast<uint2 *>(sQ)[tidx];
        // gK_uint2[tidx] = reinterpret_cast<uint2 *>(sK + cur_len * head_dim)[tidx];

        // gKCache的布局为(Batch, num_layer, num_heads, headdim / n_elem_per_cpy, memory_max_len, n_elem_per_cpy)
        // gVCache的布局为(Batch, num_layer, num_heads, memory_max_len, headdim)
        cp_async_ca<T, 8>(sV + offset_rotary, gV + offset_rotary);
        CP_ASYNC_COMMIT_GROUP();
        // int cur_len = seq_len % n_elem_per_blockN;
        int idx_1 = tidx / 2;
        int idx_3 = (tidx % 2) * 4;
        uint2 *gKCache_uint2 = reinterpret_cast<uint2 *>(gKCache + idx_1 * memory_max_len * N_ELEM_PER_CPY + seq_len * N_ELEM_PER_CPY + idx_3);
        uint2 *gVCache_uint2 = reinterpret_cast<uint2 *>(gVCache + seq_len * head_dim);
        gKCache_uint2[0] = reinterpret_cast<uint2 *>(sK)[tidx];
        gVCache_uint2[tidx] = reinterpret_cast<uint2 *>(sV)[tidx];
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31 && tidx == 0)
        // {
        //     printf("[DEBUG] after rotary: %f %f %f %f\n",
        //            __half2float(sQ[tidx * 4]), __half2float(sQ[tidx * 4 + 1]), __half2float(sQ[tidx * 4 + 2]), __half2float(sQ[tidx * 4 + 3]));
        // }
        // }
    }
    else
    {
        CP_ASYNC_WAIT_ALL();
        // 非最后一个split只对Q进行旋转编码
        // printf("offset: %d, max_offset: %d cur_len: %d, n_elem_per_blockN: %d\n", cur_len * head_dim + tidx * 8, 2 * n_elem_per_blockN * head_dim, cur_len, n_elem_per_blockN);
        apply_rotary_embedding(reinterpret_cast<uint2 *>(sQ)[tidx], tidx, rot_embed_dim, seq_len, sKCache, sVCache);
    }

    // 获得当前batch,layer下的KV cache的起始位置
    // CacheTable (Batch, num_layer, memory_max_len, num_heads, headdim)
    // __syncthreads();
    // TODO: DoubleBuffer优化
    // 处理最后一个seqLen不能被整除的块
    if (n_split_idx == num_splits - 1 && padding_seq_len % n_elem_per_blockN != 0)
    {
        const int actual_seq_len = padding_seq_len % n_elem_per_blockN;
        // KV从gCache向量化拷贝到sK和sV中，只拷贝(seq_len % n_elem_per_block, head_dim)的数据，每个线程拷贝8个元素
        // 现改为每个线程执行n_elem_per_blockN个seq的拷贝，且进行异步拷贝，需要使用数据时再确保已经拷贝结束
        // TODO: n_elem_per_thread有混淆，需要分成两个量，分别针对head_dim和kBlockN
        for (int i = tidx * n_elem_per_thread; i < (tidx + 1) * n_elem_per_thread && i < actual_seq_len - 1; i++)
            for (int j = 0; j < head_dim / 8; j++)
            {
                // uint2 *gKCache_uint2 = reinterpret_cast<uint2 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 4);
                // uint2 *gVCache_uint2 = reinterpret_cast<uint2 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 4);
                // reinterpret_cast<uint2 *>(sK + i * head_dim)[tidx] = gKCache_uint2[0];
                // reinterpret_cast<uint2 *>(sV + i * head_dim)[tidx] = gVCache_uint2[0];
                cp_async_cg<half, 16>(sK + i * head_dim + j * 8, gKCache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
                cp_async_cg<half, 16>(sV + i * head_dim + j * 8, gVCache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
            }
        // __syncthreads();
        // 计算Attention中的Q * K^T，Q (1, H) K^T (H, S) Score (1, S)，因此采用GEMV
        // 假设共有32个Thread, Q(1, HeadDim), K^T (HeadDim, n_elem_per_blockN) Score (1, n_elem_per_blockN), 所有数据已经在Share Memory中
        // 考虑1个Thread最多处理8个元素，但n_elem_per_blockN有128个元素，因此每个Thread最多负责4个元素
        clear_reg(rS, n_elem_per_thread);
        // gemv_qk(sK, sQ, rS, actual_seq_len, head_dim, tidx, n_elem_per_thread, input.head_dim_inv);
        // 现在每个线程的rS都有4个元素，代表tid*4 - (tid+1)*4 的相乘累加结果，现执行Softmax，先找到最大值
        // 由于S已经清零，且e的任意幂均为正数，因此不需要考虑越位问题
        // TODO: 转换到FP32做softmax
        float maxVal = rS[0];
#pragma unroll
        for (int i = 1; i < n_elem_per_thread; i++)
        {
            maxVal = __max(maxVal, rS[i]);
        }
        m1_in_formula = warpReduceMax(maxVal, 32);
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP_2[i] = __exp(rS[i] - m1_in_formula);
            rEll_1 += __half2float(rP_1[i]);
        }
        rEll_2 = rEll_1;
        rEll_1 = 0;
        // 对P(1,S) V(S,H)进行Gemv，此时每个线程拥有自己的P，而V在ShareMemory上，最终把结果写入rO中
        // gemv_pv(half *rP, half *gV_cache, half *sV_cache, half *sV, float *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, int n_elem_per_blockN, int head_dim, int n_block, int memory_max_len, bool is_first, int m1_in_formula, int m2_in_formula)
        // gemv_pv(rP_2, gVCache, sVCache, sV, rO, actual_seq_len, head_dim, tidx, n_elem_per_thread, is_first, m1_in_formula, m2_in_formula);
        is_first = false;
        last_split_is_first = false;
    }
#pragma unroll
    for (; n_block >= n_block_min; --n_block)
    {
        // 计算Attention中的Q * K^T，Q (1, H) K^T (H, S) Score (1, S)，因此采用GEMV
        // 假设共有32个Thread, Q(1, HeadDim), K^T (HeadDim, n_elem_per_blockN) Score (1, n_elem_per_blockN), 所有数据已经在Share Memory中
        // 考虑1个Thread最多处理8个元素，但n_elem_per_blockN有128个元素，因此每个Thread最多负责4个元素
        // CP_ASYNC_COMMIT_GROUP();
        clear_reg(rS, n_elem_per_thread);
        // gKCache的布局为(Batch, num_layer, num_heads, headdim / n_elem_per_cpy, memory_max_len, n_elem_per_cpy)
        gemv_qk(sQ, gKCache, sKCache, sK, rS, n_elem_per_blockN, n_elem_per_blockN, head_dim, n_block, memory_max_len, tidx, n_elem_per_thread, input.head_dim_inv, last_block_qv_mask, n_block_max == n_blockN);

        float maxVal = rS[0];
#pragma unroll
        for (int i = 1; i < n_elem_per_thread; i++)
        {
            maxVal = __max(maxVal, rS[i]);
        }

        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31)
        // {
        //     // printf("rEll: %f\n", __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     // printf("m1_in_formula: %f\n", __half2float(reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));

        //     printf("[1 tidx=%d] m1_in_formula: %f, m2_in_formula: %f, rEll: %f, maxVal: %f\n", tidx, __half2float(m1_in_formula), __half2float(m2_in_formula), __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]), __half2float(maxVal));
        // }
        if (is_first)
        {
            m2_in_formula = warpReduceMax((float)maxVal, 32);
        }
        else
        {
            m2_in_formula = __max((float)warpReduceMax(maxVal, 32), m1_in_formula);
        }
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP_1[i] = __exp(rS[i] - m2_in_formula);
            rEll_1 += __half2float(rP_1[i]);
        }

        if (is_first)
        {
            rEll_2 = warpReduceSum(rEll_1, 32);
        }
        else
        {
            rEll_2 = __exp(m1_in_formula - m2_in_formula) * rEll_2 + warpReduceSum(rEll_1, 32);
        }
        rEll_2_inv = 1.0 / rEll_2;

        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP_2[i] = rP_1[i] * __float2half(rEll_2_inv);
        }

        for (int i = 0; i < n_elem_per_thread; i++)
        {
            // 复用sK，存储rP_2
            sK[i * WARP_SIZE + tidx] = rP_2[i];
        }
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31)
        // {
        //     // printf("rEll: %f\n", __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     // printf("m1_in_formula: %f\n", __half2float(reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     printf("rEll_2_inv: %f, rEll_2: %f, rP_2[%d, %d, %d, %d] = %f, %f, %f, %f\n", __half2float(rEll_2_inv), __half2float(rEll_2), tidx * 4, tidx * 4 + 1, tidx * 4 + 2, tidx * 4 + 3, __half2float(reinterpret_cast<T *>(rP_2)[0]), __half2float(reinterpret_cast<T *>(rP_2)[1]), __half2float(reinterpret_cast<T *>(rP_2)[2]), __half2float(reinterpret_cast<T *>(rP_2)[3]));
        // }
        // 对于sO先修正再加和，因此不用再置0

        // CP_ASYNC_WAIT_ALL();
        // gemv_pv(half *rP, half *gV_cache, half *sV_cache, half *sV, float *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, int n_elem_per_blockN, int head_dim, int n_block, int memory_max_len, bool is_first, int m1_in_formula, int m2_in_formula)

        gemv_pv(sK, gVCache, sVCache, sV, rO, n_elem_per_blockN, head_dim, tidx, n_elem_per_thread, n_elem_per_blockN, head_dim, n_block, memory_max_len, is_first, m1_in_formula, m2_in_formula);

        m1_in_formula = m2_in_formula;
        is_first = false;
        last_split_is_first = false;
    }
    // rO向量化拷贝回全局内存中，每个线程拷贝4个元素

    // reinterpret_cast<uint2 *>(gO)[tidx] = reinterpret_cast<uint2 *>(rO)[0];

    for (int i = 0; i < n_elem_per_thread; i++)
    {
        gO[tidx + WARP_SIZE * i] = __float2half(rO[i]);
    }

    // printf("[here!]\n");
    if (num_splits > 1)
    {
        // buffers.ell Shape: (batch_size, num_heads, n_split)
        // buffers.m_formula shape: (batch_size, num_heads, n_split)
        // buffers.o_spilit shape: (batch_size, num_heads, n_split, head_dim)
        reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx] = rEll_2;
        reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx] = m1_in_formula;

        // if (blockIdx.x == 0 && blockIdx.y == 2 && threadIdx.x == 0)
        // {
        //     printf("[DEBUG] [head:%d] rEll: %f, m1_in_formula: %f\n", bidh, __half2float(rEll_2), __half2float(m1_in_formula));
        // }
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31)
        // {
        //     // printf("rEll: %f\n", __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     // printf("m1_in_formula: %f\n", __half2float(reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     printf("[tidx=%d] gO ptr: %p \n", tidx, gO);
        //     // printf("rO[%d, %d, %d, %d] = %f, %f, %f, %f\n", tidx * 4, tidx * 4 + 1, tidx * 4 + 2, tidx * 4 + 3, __half2float(reinterpret_cast<T *>(rO)[0]), __half2float(reinterpret_cast<T *>(rO)[1]), __half2float(reinterpret_cast<T *>(rO)[2]), __half2float(reinterpret_cast<T *>(rO)[3]));
        // }
    }
    // printf("[DEBUG] Here End\n");
}

template <typename T, typename Traits>
__global__ void flash_combine_kernel(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers)
{
    const int n_split = params.num_splits;
    const int num_heads = input.num_heads;
    const int head_dim = input.head_dim;

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;

    const int num_per_thread = Traits::n_elem_per_thread;

    // Shape: (batch_size, num_heads, n_split)
    T *gEll = reinterpret_cast<T *>(buffers.ell);
    T *gM = reinterpret_cast<T *>(buffers.m_formula);
    T *gO_split = reinterpret_cast<T *>(buffers.o_split);
    // Shape: (batch_size, hidden_size)
    T *gO = reinterpret_cast<T *>(input.o);

    // 将当前batch、head的所有Ell拷贝至寄存器中
    T rM = -256;
    for (int i = 0; i < n_split; i++)
    {
        rM = __max(rM, gM[bidb * num_heads * n_split + bidh * n_split + i]);
    }
    // 将Ell修正后加和
    T rEll = 0;
    for (int i = 0; i < n_split; i++)
    {
        rEll += __exp(gM[bidb * num_heads * n_split + bidh * n_split + i] - rM) * gEll[bidb * num_heads * n_split + bidh * n_split + i];
        // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        //     printf("[DEBUG] gM: %f rEll: %f gEll[%d]: %f\n", __half2float(gM[bidb * num_heads * n_split + bidh * n_split + i]), __half2float(rEll), i, __half2float(gEll[bidb * num_heads * n_split + bidh * n_split + i]));
    }

    T rO[Traits::n_elem_per_thread];
    clear_reg(rO, num_per_thread);
    // 每个线程沿n_split方向遍历，对O修正后加和在rO中
    for (int i = 0; i < n_split; i++)
    {
        // 每个线程负责N个Thread，在这里是FP16，因此每个线程负责4个元素，使用uint2向量化访存
        // Shape: (batch_size, num_heads, n_split, head_dim)
        uint2 *rO_split = reinterpret_cast<uint2 *>(gO_split + bidb * num_heads * n_split * head_dim + bidh * n_split * head_dim + i * head_dim + threadIdx.x * num_per_thread);
        for (int j = 0; j < num_per_thread; j++)
        {
            rO[j] += __exp(gM[bidb * num_heads * n_split + bidh * n_split + i] - rM) * reinterpret_cast<T *>(rO_split)[j];
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            //     printf("[DEBUG] expxx: %f gM: %f rM: %f rO[%d]: %f rO_acc: %f\n", __half2float(__exp(gM[bidb * num_heads * n_split + bidh * n_split + i] - rM)), __half2float(gM[bidb * num_heads * n_split + bidh * n_split + i]), __half2float(rM), j, __half2float(reinterpret_cast<T *>(rO_split)[j]), __half2float(rO[j]));
        }
    }
    T invREll = half(1) / rEll; // 计算倒数
    // 除以ell后，将结果写回全局内存
    for (int i = 0; i < num_per_thread; i++)
    {
        rO[i] *= invREll;
    }

    // rO向量化拷贝回全局内存中，每个线程拷贝4个元素
    reinterpret_cast<uint2 *>(gO + bidb * num_heads * head_dim + bidh * head_dim + threadIdx.x * num_per_thread)[0] = reinterpret_cast<uint2 *>(rO)[0];
}

template <typename T>
void run_flash_decoder(Flash_decoder_input &input, Flash_decoder_params &params, cudaStream_t stream)
{
    using traits = Traits<half, 128, 128>;
    // grid (1, num_splits, b*h)
    // block(kNThreads) 128 = 4 Warps
    // 先定义一个在设备端使用的结构体指针
    Flash_decoder_buffers buffers;

    // 为设备端结构体分配内存
    cudaMalloc((void **)&buffers, sizeof(Flash_decoder_buffers));

    // 为每个内部指针分别分配内存，并设置设备端结构体的指针
    cudaMalloc(&(buffers.o_split), sizeof(T) * input.batch_size * input.num_heads * params.num_splits * input.head_dim);
    cudaMalloc(&(buffers.ell), sizeof(T) * input.batch_size * input.num_heads * params.num_splits);
    cudaMalloc(&(buffers.m_formula), sizeof(T) * input.batch_size * input.num_heads * params.num_splits);
    dim3 grid(1, params.num_splits, input.batch_size * input.num_heads);
    dim3 block(params.kNThreads);
    cudaError_t err;
    // printf("Share Memory Size: %d\n", traits::smemSize);
    if (traits::smemSize >= 48 * 1024)
    {
        cudaFuncSetAttribute(
            flash_decoder_kernel<T, traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::smemSize);
        cudaFuncSetAttribute(
            flash_combine_kernel<T, traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::smemSize);
    }

    Flash_decoder_input *d_input;
    Flash_decoder_params *d_params;
    Flash_decoder_buffers *d_buffers;

    cudaMalloc((void **)&d_input, sizeof(Flash_decoder_input));
    cudaMalloc((void **)&d_params, sizeof(Flash_decoder_params));
    cudaMalloc((void **)&d_buffers, sizeof(Flash_decoder_buffers));

    cudaMemcpy(d_input, &input, sizeof(Flash_decoder_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(Flash_decoder_params), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffers, &buffers, sizeof(Flash_decoder_buffers), cudaMemcpyHostToDevice);

    // 将d_input->q 拷贝到host中，并输出
    flash_decoder_kernel<T, traits><<<grid, block, traits::smemSize, stream>>>(*d_input, *d_params, *d_buffers);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA错误: %s\n", cudaGetErrorString(err));
        return; // 发生错误，提前退出函数
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA设备同步错误: %s\n", cudaGetErrorString(err));
        return; // 发生错误，提前退出函数
    }

    // T *q_host = new T[input.batch_size * input.num_heads * input.head_dim];
    // cudaMemcpy(q_host, reinterpret_cast<T *>(buffers.o_split) + 31 * params.num_splits * input.head_dim + 3 * input.head_dim, sizeof(T) * input.head_dim, cudaMemcpyDeviceToHost);
    // std::cout << "o_split:" << std::endl;
    // for (int i = 0; i < input.head_dim; i++)
    // {
    //     printf("%f ", __half2float(q_host[i]));
    // }
    // std::cout << std::endl;

    dim3 grid_combine(input.batch_size, input.num_heads);
    dim3 block_combine(params.kNThreads);

    flash_combine_kernel<T, traits><<<grid_combine, block_combine, traits::smemSize, stream>>>(*d_input, *d_params, *d_buffers);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA错误 (组合核函数): %s\n", cudaGetErrorString(err));
    }
    // 销毁d_buffers
    cudaFree(buffers.o_split);
    cudaFree(buffers.ell);
    cudaFree(buffers.m_formula);

    cudaFree(d_input);
    cudaFree(d_params);
}

template __global__ void flash_decoder_kernel<half, Traits<half, 128, 64>>(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers);
template __global__ void flash_combine_kernel<half, Traits<half, 128, 64>>(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers);
template void run_flash_decoder<half>(Flash_decoder_input &input, Flash_decoder_params &params, cudaStream_t);

// TODO: 目前仅针对3090单卡和llama 7B设置分割数，后续需要智能分割