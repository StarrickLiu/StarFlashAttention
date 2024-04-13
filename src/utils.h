#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Declare device functions for external use
__device__ half warpReduceMax(half val, int warpSize);
__device__ half warpReduceSum(half val, int warpSize);
__device__ float half_to_float(uint16_t h);
__device__ float2 half2_to_float2(uint32_t v);
__device__ uint32_t float2_to_half2(float2 f);
__device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step);
__device__ float2 rotary_embedding_transform(const float2 v, const float2 coef);
__device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef);
__device__ void apply_rotary_embedding(uint4 &q, uint4 &k, int tid, int rot_embed_dim, int t_step);
__device__ void gemv_qk(half *mat, half *vec, half *res, int n, int k, int tidx, int n_elem_per_thread);
__device__ void gemv_pv(half *rP, half *sV, half *rO, int seqLen, int headDim, int tidx, int n_elem_per_thread, bool is_first, int m1_in_formula, int m2_in_formula);
__device__ void clear_shm(half *p, const int n);
__device__ void clear_reg(half *p, const int n);
__device__ half __max(half a, half b);
__device__ half __exp(half x);

void init_half_array(half *array, half value, int n, int numBlocks, int blockSize);