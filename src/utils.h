#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(Bytes))
#endif

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

template<typename T, int bytes_per_thread>
__device__ void cp_async_ca(T *dst, const T *src);
template<typename T, int bytes_per_thread>
__device__ void cp_async_cg(T *dst, const T *src);


__device__ half warpReduceMax(half val, int warpSize);
__device__ half warpReduceSum(half val, int warpSize);
__device__ float half_to_float(uint16_t h);
__device__ float2 half2_to_float2(uint32_t v);
__device__ uint32_t float2_to_half2(float2 f);
__device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step);
__device__ float2 rotary_embedding_transform(const float2 v, const float2 coef);
__device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef);
__device__ void apply_rotary_embedding(uint2 &q, uint2 &k, int tid, int rot_embed_dim, int t_step);
__device__ void apply_rotary_embedding(uint2 &q, int tid, int rot_embed_dim, int t_step);
__device__ void apply_rotary_embedding(uint4 &q, uint4 &k, int tid, int rot_embed_dim, int t_step);
__device__ void apply_rotary_embedding(uint4 &q, int tid, int rot_embed_dim, int t_step);
__device__ void gemv_qk(half *gK, half *sK, half *sQ, half *rS, int n, int k, int tidx, int n_elem_per_thread, float head_dim_inv);
__device__ void gemv_pv(half *rP, half *sV, half *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, bool is_first, int m1_in_formula, int m2_in_formula);
__device__ void clear_shm(half *p, const int n);
__device__ void clear_reg(half *p, const int n);
__device__ half __max(half a, half b);
__device__ half __exp(half x);

void init_half_array(half *array, half value, int n, int numBlocks, int blockSize);