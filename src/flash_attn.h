#pragma once
#include <src/params.h>
template <typename T, typename Traits>
__global__ void flash_decoder_kernel(Flash_decoder_input<T> &input, Flash_decoder_params &params);
template <typename T, typename Traits>
__global__ void flash_combine_kernel(Flash_decoder_input<T> &input, Flash_decoder_params &params);
template <typename T>
void run_flash_decoder(Flash_decoder_input<T> *input, Flash_decoder_params *params, cudaStream_t stream);