#pragma once
#include <src/params.h>
template <typename T, typename Traits>
__global__ void flash_decoder_kernel(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers);
template <typename T, typename Traits>
__global__ void flash_combine_kernel(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers);
template <typename T>
void run_flash_decoder(Flash_decoder_input &input, Flash_decoder_params &params, cudaStream_t stream);
template<typename T>
void compute_rotary_table(T *rotary_cos_table, T *rotary_sin_table, int max_seq_len, int rot_embed_dim);
void init_half_array(half *array, half value, int n, int numBlocks, int blockSize);
