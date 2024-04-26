// #pragma once

// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cstdint>
// #include <device_functions.h>


// #if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
// #define CP_ASYNC_CA_16B(dst, src, Bytes) \
//     asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

// #define CP_ASYNC_CG_16B(dst, src, Bytes) \
//     asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

// #define CP_ASYNC_CA_8B(dst, src, Bytes) \
//     asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

// #define CP_ASYNC_CA_4B(dst, src, Bytes) \
//     asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
// #endif
// #define CP_ASYNC_CA(dst, src, Bytes) \
//     asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(Bytes))

// #define CP_ASYNC_CG(dst, src, Bytes) \
//     asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(Bytes))

// #define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

// #define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

// #define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

// template<typename T, int bytes_per_thread>
// __device__ void cp_async_ca(T *dst, const T *src);
// template<typename T, int bytes_per_thread>
// __device__ void cp_async_cg(T *dst, const T *src);


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

// void init_half_array(half *array, half value, int n, int numBlocks, int blockSize);
// template<typename T>
// void compute_rotary_table(T *rotary_cos_table, T *rotary_sin_table, int max_seq_len, int rot_embed_dim);


// #define N_ELEM_PER_CPY 8
// #define WARP_SIZE 32
// // * async tools

// #if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
// template <>
// __device__ void cp_async_ca<half, 16>(half *dst, const half *src)
// {
//     uint32_t dst_ = __cvta_generic_to_shared(dst);
//     CP_ASYNC_CA_16B(dst_, src, 16);
// }

// template <>
// __device__ void cp_async_cg<half, 16>(half *dst, const half *src)
// {
//     uint32_t dst_ = __cvta_generic_to_shared(dst);
//     CP_ASYNC_CG_16B(dst_, src, 16);
// }
// #else
// template <>
// __device__ void cp_async_ca<half, 16>(half *dst, const half *src)
// {
//     CP_ASYNC_CA(dst_, src, 16);
// }

// template <>
// __device__ void cp_async_cg<half, 16>(half *dst, const half *src)
// {
//     CP_ASYNC_CG(dst_, src, 16);
// }
// #endif
// template <>
// __device__ void cp_async_ca<half, 8>(half *dst, const half *src)
// {
//     uint32_t dst_ = __cvta_generic_to_shared(dst);
//     CP_ASYNC_CA_8B(dst_, src, 8);
// }

// template <>
// __device__ void cp_async_ca<half, 4>(half *dst, const half *src)
// {
//     uint32_t dst_ = __cvta_generic_to_shared(dst);
//     CP_ASYNC_CA_4B(dst_, src, 4);
// }

// // * For FP16

// __device__ half warpReduceMax(half val, int warpSize)
// {
//     for (int offset = warpSize / 2; offset > 0; offset /= 2)
//         val = __hmax(val, __shfl_xor_sync(0xffffffff, val, offset));
//     return val;
// }

// __device__ half warpReduceSum(half val, int warpSize)
// {
//     for (int offset = warpSize / 2; offset > 0; offset /= 2)
//         val += __shfl_xor_sync(0xffffffff, val, offset);
//     return val;
// }

// __device__ float half_to_float(uint16_t h)
// {
//     float f;
//     asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
//     return f;
// }

// __device__ float2 half2_to_float2(uint32_t v)
// {
//     uint16_t lo, hi;
//     asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
//     return make_float2(half_to_float(lo), half_to_float(hi));
// }

// __device__ uint32_t float2_to_half2(float2 f)
// {
//     union
//     {
//         uint32_t u32;
//         uint16_t u16[2];
//     } tmp;
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//     asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
// #else
//     asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
//     asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
// #endif
//     return tmp.u32;
// }

// __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step)
// {
//     const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
//     return {cos(inv_freq), sin(inv_freq)};
// }

// __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
// {
//     float2 rot_v;
//     rot_v.x = coef.x * v.x - coef.y * v.y;
//     rot_v.y = coef.x * v.y + coef.y * v.x;
//     return rot_v;
// }

// __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
// {
//     float2 fv = half2_to_float2(v);
//     float2 rot_fv = rotary_embedding_transform(fv, coef);
//     return float2_to_half2(rot_fv);
// }

// __device__ uint32_t rotary_embedding_transform(const uint32_t v, const uint32_t coef)
// {
//     float2 fv = half2_to_float2(v);
//     float2 fcoef = half2_to_float2(coef);
//     float2 rot_fv = rotary_embedding_transform(fv, fcoef);
//     return float2_to_half2(rot_fv);
// }

// // 一个线程负责四个元素的旋转编码
// __device__ void apply_rotary_embedding(uint2 &q, uint2 &k, const int tid, const int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
// {
//     if (4 * tid >= rot_embed_dim)
//     {
//         return;
//     }

//     const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
//     q.x = rotary_embedding_transform(q.x, coef0);
//     k.x = rotary_embedding_transform(k.x, coef0);
//     const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
//     q.y = rotary_embedding_transform(q.y, coef1);
//     k.y = rotary_embedding_transform(k.y, coef1);
// }

// // 一个线程负责四个元素的旋转编码
// __device__ void apply_rotary_embedding(uint2 &q, const int tid, const int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
// {
//     if (4 * tid >= rot_embed_dim)
//     {
//         return;
//     }
//     half rCOS[2] = {rotary_cos[0], rotary_cos[1]};
//     half rSIN[2] = {rotary_sin[0], rotary_sin[1]};
//     // const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
//     q.x = rotary_embedding_transform(q.x, {rCOS[0], rSIN[0]});
//     // const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
//     q.y = rotary_embedding_transform(q.y, {rCOS[1], rSIN[1]});
// }

// // 一个线程负责八个元素的旋转编码
// __device__ void apply_rotary_embedding(uint4 &q, uint4 &k, int tid, int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
// {
//     if (8 * tid >= rot_embed_dim)
//     {
//         return;
//     }
//     const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
//     q.x = rotary_embedding_transform(q.x, coef0);
//     k.x = rotary_embedding_transform(k.x, coef0);
//     const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
//     q.y = rotary_embedding_transform(q.y, coef1);
//     k.y = rotary_embedding_transform(k.y, coef1);
//     const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
//     q.z = rotary_embedding_transform(q.z, coef2);
//     k.z = rotary_embedding_transform(k.z, coef2);
//     const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
//     q.w = rotary_embedding_transform(q.w, coef3);
//     k.w = rotary_embedding_transform(k.w, coef3);
// }

// // 一个线程负责八个元素的旋转编码
// __device__ void apply_rotary_embedding(uint4 &q, int tid, int rot_embed_dim, int t_step, half *rotary_cos, half *rotary_sin)
// {
//     if (8 * tid >= rot_embed_dim)
//     {
//         return;
//     }
//     const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
//     q.x = rotary_embedding_transform(q.x, coef0);
//     const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
//     q.y = rotary_embedding_transform(q.y, coef1);
//     const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
//     q.z = rotary_embedding_transform(q.z, coef2);
//     const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
//     q.w = rotary_embedding_transform(q.w, coef3);
// }

// // for fp16 gemv

// __device__ void gemv_qk(half *sQ, half *gK_cache, half *sK_cache, half *sK, float *rS, int n_elem_per_blockN, int head_dim, int n_block, int memory_max_len, int tidx, int n_elem_per_thread, float head_dim_inv, bool last_block)
// {
//     // (uint4) sK (n/8, k/8) (uint4) sQ (k/8) (uint4) rS (n/8)
//     // 共计算n_elem_per_thread个元素的结果，偏移量为tidx*n_elem_per_thread，确保k一定可以被8整除
//     // 找到对应的矩阵行后，沿着k方向循环，每次向量化拷贝8个Vec和n_elem_per_thread*8个Mat的元素，相乘后结果存储到rS，注意Mat不要超出n的范围
//     // 修改后逻辑：
//     // gK_cache的布局为(headdim / n_elem_per_cpy, memory_max_len, n_elem_per_cpy)
//     // 需要取(headdim/n_elem_per_cpy, n_block*n_elem_per_blockN - (n_block+1)*n_elem_per_blockN, n_elem_per_cpy)
//     // sK_cache: (head_dim / n_elem_per_cpy, n_elem_per_blockN, n_elem_per_cpy) sK: (head_dim)
//     // gK_cache逐元素异步流水线拷贝到sK_cache中
//     // 每一轮各线程计算一个元素的结果，结果存储到rS中，若headDim=128，则每个线程计算4个元素的结果，且4个元素是散布的
//     // exm: Thr0 0-32-64-96,Thr1 1-33-65-97,..., Thr31 31-63-95-127
//     // 如果是最后一个split的最后一个block的情况，需要单独计算最新的rS，使用一个判断给到对应的线程（会导致Warp分叉性能损耗）

//     // 每个线程Q在HeadDim维度上的循环
//     int offset_dst = tidx * N_ELEM_PER_CPY;
//     int offset_src = (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;
//     int count = 0;
//     for (int i = 0; i < head_dim / N_ELEM_PER_CPY; i++)
//     {
//         offset_dst = i * n_elem_per_blockN * N_ELEM_PER_CPY + tidx * N_ELEM_PER_CPY;
//         offset_src = i * memory_max_len * N_ELEM_PER_CPY + (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;
//         for (int j = 0; j < n_elem_per_thread; j++)
//         {
//             cp_async_ca<half, 16>(sK_cache + offset_dst + j * WARP_SIZE * N_ELEM_PER_CPY,
//                                   gK_cache + offset_src + j * WARP_SIZE * N_ELEM_PER_CPY);
//         }
//     }
//     CP_ASYNC_COMMIT_GROUP();
//     CP_ASYNC_WAIT_ALL();

// #pragma unroll
//     for (int i = 0; i < head_dim / N_ELEM_PER_CPY; i++)
//     {
//         offset_dst = i * n_elem_per_blockN * N_ELEM_PER_CPY + tidx * N_ELEM_PER_CPY;
//         offset_src = i * memory_max_len * N_ELEM_PER_CPY + (n_block * n_elem_per_blockN + tidx) * N_ELEM_PER_CPY;

//         // if (i < head_dim / N_ELEM_PER_CPY - 1)
//         // {
//         //     for (int j = 0; j < n_elem_per_thread; j++)
//         //     {
//         //         cp_async_ca<half, 16>(sK_cache + offset_dst + j * WARP_SIZE * N_ELEM_PER_CPY,
//         //                               gK_cache + offset_src + j * WARP_SIZE * N_ELEM_PER_CPY);
//         //         CP_ASYNC_COMMIT_GROUP();
//         //     }
//         // }

//         // ShareMemory的4个Bank执行广播到所有线程
//         uint4 sQ_tmp = reinterpret_cast<uint4 *>(sQ)[i];
//         half2 *sQ_h1 = (half2 *)&sQ_tmp.x;
//         half2 *sQ_h2 = (half2 *)&sQ_tmp.y;
//         half2 *sQ_h3 = (half2 *)&sQ_tmp.z;
//         half2 *sQ_h4 = (half2 *)&sQ_tmp.w;
//         // for (int j = 0; j < n_elem_per_thread; j++)
//         // {
//         //     uint4 sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN + j];
//         //     half2 *sK_h1 = (half2 *)&sK_tmp.x;
//         //     half2 *sK_h2 = (half2 *)&sK_tmp.y;
//         //     half2 *sK_h3 = (half2 *)&sK_tmp.z;
//         //     half2 *sK_h4 = (half2 *)&sK_tmp.w;
//         //     rS[j] += T2float(sK_h1->x * sQ_h1->x);
//         //     rS[j] += T2float(sK_h1->y * sQ_h1->y);
//         //     rS[j] += T2float(sK_h2->x * sQ_h2->x);
//         //     rS[j] += T2float(sK_h2->y * sQ_h2->y);
//         //     rS[j] += T2float(sK_h3->x * sQ_h3->x);
//         //     rS[j] += T2float(sK_h3->y * sQ_h3->y);
//         //     rS[j] += T2float(sK_h4->x * sQ_h4->x);
//         //     rS[j] += T2float(sK_h4->y * sQ_h4->y);
//         // } / 

//         // if (i == head_dim / N_ELEM_PER_CPY - 1)
//         // {
//         //     CP_ASYNC_WAIT_GROUP(7);
//         // }
//         // else
//         // {
//         //     CP_ASYNC_WAIT_GROUP(3);
//         // }
//         uint4 sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN];
//         half2 *sK_h1 = (half2 *)&sK_tmp.x;
//         half2 *sK_h2 = (half2 *)&sK_tmp.y;
//         half2 *sK_h3 = (half2 *)&sK_tmp.z;
//         half2 *sK_h4 = (half2 *)&sK_tmp.w;
//         rS[0] += T2float(sK_h1->x * sQ_h1->x);
//         rS[0] += T2float(sK_h1->y * sQ_h1->y);
//         rS[0] += T2float(sK_h2->x * sQ_h2->x);
//         rS[0] += T2float(sK_h2->y * sQ_h2->y);
//         rS[0] += T2float(sK_h3->x * sQ_h3->x);
//         rS[0] += T2float(sK_h3->y * sQ_h3->y);
//         rS[0] += T2float(sK_h4->x * sQ_h4->x);
//         rS[0] += T2float(sK_h4->y * sQ_h4->y);

//         // if (i == head_dim / N_ELEM_PER_CPY - 1)
//         // {
//         //     CP_ASYNC_WAIT_GROUP(6);
//         // }
//         // else
//         // {
//         //     CP_ASYNC_WAIT_GROUP(2);
//         // }
//         sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN + 1];
//         sK_h1 = (half2 *)&sK_tmp.x;
//         sK_h2 = (half2 *)&sK_tmp.y;
//         sK_h3 = (half2 *)&sK_tmp.z;
//         sK_h4 = (half2 *)&sK_tmp.w;
//         rS[1] += T2float(sK_h1->x * sQ_h1->x);
//         rS[1] += T2float(sK_h1->y * sQ_h1->y);
//         rS[1] += T2float(sK_h2->x * sQ_h2->x);
//         rS[1] += T2float(sK_h2->y * sQ_h2->y);
//         rS[1] += T2float(sK_h3->x * sQ_h3->x);
//         rS[1] += T2float(sK_h3->y * sQ_h3->y);
//         rS[1] += T2float(sK_h4->x * sQ_h4->x);
//         rS[1] += T2float(sK_h4->y * sQ_h4->y);

//         // if (i == head_dim / N_ELEM_PER_CPY - 1)
//         // {
//         //     CP_ASYNC_WAIT_GROUP(5);
//         // }
//         // else
//         // {
//         //     CP_ASYNC_WAIT_GROUP(1);
//         // }
//         sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN + 2];
//         sK_h1 = (half2 *)&sK_tmp.x;
//         sK_h2 = (half2 *)&sK_tmp.y;
//         sK_h3 = (half2 *)&sK_tmp.z;
//         sK_h4 = (half2 *)&sK_tmp.w;
//         rS[2] += T2float(sK_h1->x * sQ_h1->x);
//         rS[2] += T2float(sK_h1->y * sQ_h1->y);
//         rS[2] += T2float(sK_h2->x * sQ_h2->x);
//         rS[2] += T2float(sK_h2->y * sQ_h2->y);
//         rS[2] += T2float(sK_h3->x * sQ_h3->x);
//         rS[2] += T2float(sK_h3->y * sQ_h3->y);
//         rS[2] += T2float(sK_h4->x * sQ_h4->x);
//         rS[2] += T2float(sK_h4->y * sQ_h4->y);

//         // if (i == head_dim / N_ELEM_PER_CPY - 1)
//         // {
//         //     CP_ASYNC_WAIT_GROUP(4);
//         // }
//         // else
//         // {
//         //     CP_ASYNC_WAIT_GROUP(0);
//         // }
//         sK_tmp = reinterpret_cast<uint4 *>(sK_cache)[i * n_elem_per_blockN + 3];
//         sK_h1 = (half2 *)&sK_tmp.x;
//         sK_h2 = (half2 *)&sK_tmp.y;
//         sK_h3 = (half2 *)&sK_tmp.z;
//         sK_h4 = (half2 *)&sK_tmp.w;
//         rS[3] += T2float(sK_h1->x * sQ_h1->x);
//         rS[3] += T2float(sK_h1->y * sQ_h1->y);
//         rS[3] += T2float(sK_h2->x * sQ_h2->x);
//         rS[3] += T2float(sK_h2->y * sQ_h2->y);
//         rS[3] += T2float(sK_h3->x * sQ_h3->x);
//         rS[3] += T2float(sK_h3->y * sQ_h3->y);
//         rS[3] += T2float(sK_h4->x * sQ_h4->x);
//         rS[3] += T2float(sK_h4->y * sQ_h4->y);
//         // if (blockIdx.x == 0 && blockIdx.y == 2 && blockIdx.z == 31 && tidx == 31)
//         // {
//         //     printf("sQ[%d]: %f\n", i*8 + 0, __half2float(sQ_h1->x));
//         //     printf("sQ[%d]: %f\n", i*8 + 1, __half2float(sQ_h1->y));
//         //     printf("sQ[%d]: %f\n", i*8 + 2, __half2float(sQ_h2->x));
//         //     printf("sQ[%d]: %f\n", i*8 + 3, __half2float(sQ_h2->y));
//         //     printf("sQ[%d]: %f\n", i*8 + 4, __half2float(sQ_h3->x));
//         //     printf("sQ[%d]: %f\n", i*8 + 5, __half2float(sQ_h3->y));
//         //     printf("sQ[%d]: %f\n", i*8 + 6, __half2float(sQ_h4->x));
//         //     printf("sQ[%d]: %f\n", i*8 + 7, __half2float(sQ_h4->y));
//         //     printf("sK[%d]: %f\n", j*8 + 0, __half2float(sK_h1->x));
//         //     printf("sK[%d]: %f\n", j*8 + 1, __half2float(sK_h1->y));
//         //     printf("sK[%d]: %f\n", j*8 + 2, __half2float(sK_h2->x));
//         //     printf("sK[%d]: %f\n", j*8 + 3, __half2float(sK_h2->y));
//         //     printf("sK[%d]: %f\n", j*8 + 4, __half2float(sK_h3->x));
//         //     printf("sK[%d]: %f\n", j*8 + 5, __half2float(sK_h3->y));
//         //     printf("sK[%d]: %f\n", j*8 + 6, __half2float(sK_h4->x));
//         //     printf("sK[%d]: %f\n", j*8 + 7, __half2float(sK_h4->y));
//         //     printf("rS[%d]: %f\n", idx, __half2float(rS[idx]));
//         // }
//         // }
//     }

//     // if (last_block) {
//     //     uint2 rQ = reinterpret_cast<uint2 *>(sQ)[tidx];
//     //     uint2 rK = reinterpret_cast<uint2 *>(sK)[tidx];
//     //     half2 *rQ_h1 = (half2 *)&rQ.x;
//     //     half2 *rQ_h2 = (half2 *)&rQ.y;
//     //     half2 *rK_h1 = (half2 *)&rK.x;
//     //     half2 *rK_h2 = (half2 *)&rK.y;
//     //     float tmpS = T2float(rK_h1->x * rQ_h1->x + rK_h1->y * rQ_h1->y + rK_h2->x * rQ_h2->x + rK_h2->y * rQ_h2->y);

//     // }
//     for (int i = 0; i < n_elem_per_thread; i++)
//     {
//         rS[i] = rS[i] * head_dim_inv;
//     }
// }

// __device__ void gemv_pv(half *rP, half *sV, half *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, bool is_first, int m1_in_formula, int m2_in_formula)
// {
//     // rP: (4) ...[tidx*n_elem_per_thread, (tidx+1)*n_elem_per_thread) located in P [seqLen]
//     // sV: (seqLen, headDim)
//     // rO: (4)

//     // 非第一次的情况，先修正原先的O
//     if (!is_first)
//     {
// #pragma unroll
//         for (int i = 0; i < 4; i++)
//         {
//             rO[i] = rO[i] * __double2half(exp(m1_in_formula - m2_in_formula));
//         }
//     }
//     // 每个线程每次沿H维度向量化读取sV的8个元素，读取4行，共读取32个元素进入寄存器
//     // 使用Warp快速通信的方式将结果累加到rO中
//     int s_start = tidx * n_elem_per_thread;
//     int s_end = (tidx + 1) * n_elem_per_thread;
// #pragma unroll
//     for (int u = 0; u < headDim / 8; u += 1)
//     {
//         half buf[8] = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f),
//                        __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
//         for (int i = s_start; i < s_end && i < seqLen; i++)
//         {
//             int idx = i - s_start;
//             uint4 rV = reinterpret_cast<uint4 *>(sV + i * headDim)[u];
//             half2 *rV_h1 = (half2 *)&rV.x;
//             half2 *rV_h2 = (half2 *)&rV.y;
//             half2 *rV_h3 = (half2 *)&rV.z;
//             half2 *rV_h4 = (half2 *)&rV.w;
//             buf[0] += rP[idx] * rV_h1->x;
//             buf[1] += rP[idx] * rV_h1->y;
//             buf[2] += rP[idx] * rV_h2->x;
//             buf[3] += rP[idx] * rV_h2->y;
//             buf[4] += rP[idx] * rV_h3->x;
//             buf[5] += rP[idx] * rV_h3->y;
//             buf[6] += rP[idx] * rV_h4->x;
//             buf[7] += rP[idx] * rV_h4->y;
//         }
//         // 规约当前warp上的buf，广播到每个线程中，如果是当前线程负责的元素，则相加到寄存器rO上
//         for (int i = 0; i < 8; ++i)
//         {
//             buf[i] = warpReduceSum(buf[i], 32);
//         }
//         if (tidx == 2 * u)
//         {
//             rO[0] += buf[0];
//             rO[1] += buf[1];
//             rO[2] += buf[2];
//             rO[3] += buf[3];
//         }
//         if (tidx == 2 * u + 1)
//         {
//             rO[0] += buf[4];
//             rO[1] += buf[5];
//             rO[2] += buf[6];
//             rO[3] += buf[7];
//         }
//     }
// }

// __device__ void clear_shm(half *p, const int n)
// {
//     for (int i = threadIdx.x; i < n; i += blockDim.x)
//     {
//         p[i] = 0;
//     }
// }

// __device__ void clear_reg(half *p, const int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         p[i] = 0;
//     }
// }

// __device__ void clear_reg(float *p, const int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         p[i] = 0;
//     }
// }

// __device__ half __max(half a, half b)
// {
//     return a > b ? a : b;
// }

// __device__ half __exp(half x)
// {
//     return hexp(x);
// }

// __device__ float __max(float a, float b)
// {
//     return a > b ? a : b;
// }

// __device__ float __exp(float x)
// {
//     return exp(x);
// }

// __device__ half float2T(float x)
// {
//     return __float2half(x);
// }

// __device__ float T2float(half x)
// {
//     return __half2float(x);
// }

// __global__ void init_half_array_kernel(half *array, half value, int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n)
//     {
//         array[idx] = value;
//     }
// }

// void init_half_array(half *array, half value, int n, int numBlocks, int blockSize)
// {
//     init_half_array_kernel<<<numBlocks, blockSize>>>(array, value, n);
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess)
//     {
//         printf("Init CUDA kernel error: %s\n", cudaGetErrorString(error));
//     }
// }

// __global__ void rotary_table_kernel(half *rotary_cos_table, half *rotary_sin_table, int max_seq_len, int rot_embed_dim)
// {
//     int cur_seq_len = blockIdx.x;
//     int zid = threadIdx.x * 2;
//     float freq = cur_seq_len / pow(10000.0f, zid / (float)rot_embed_dim);
//     rotary_cos_table[cur_seq_len * rot_embed_dim + threadIdx.x] = __float2half(cosf(freq));
//     rotary_sin_table[cur_seq_len * rot_embed_dim + threadIdx.x] = __float2half(sinf(freq));
//     // if (zid < rot_embed_dim / 2)
//     // { // 保证只有有效的zid参与计算
//     //     float freq = powf(10000.0f, -((float)zid / rot_embed_dim));
//     //     for (int pos = 0; pos < max_seq_len; pos++)
//     //     {
//     //         float angle = freq * pos;
//     //         rotary_cos_table[pos * rot_embed_dim + zid] = __float2half(cosf(angle));
//     //         rotary_sin_table[pos * rot_embed_dim + zid] = __float2half(sinf(angle));
//     //     }
//     // }
// }

// template <>
// void compute_rotary_table(half *rotary_cos_table, half *rotary_sin_table, int max_seq_len, int rot_embed_dim)
// {
//     int threadsPerBlock = rot_embed_dim / 2;
//     int blocksPerGrid = max_seq_len;

//     rotary_table_kernel<<<blocksPerGrid, threadsPerBlock>>>(rotary_cos_table, rotary_sin_table, max_seq_len, rot_embed_dim);
// }

// __forceinline__ __device__ float warpReduceMax(float val, int warpSize)
// {
//     for (int offset = warpSize / 2; offset > 0; offset /= 2)
//         val = __max(val, __shfl_xor_sync(0xffffffff, val, offset));
//     return val;
// }

// __forceinline__ __device__ float warpReduceSum(float val, int warpSize)
// {
//     for (int offset = warpSize / 2; offset > 0; offset /= 2)
//         val += __shfl_xor_sync(0xffffffff, val, offset);
//     return val;
// }