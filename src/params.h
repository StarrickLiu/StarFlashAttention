#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <src/traits.h>

struct Flash_decoder_input
{

    // Shape: (batch_size, 3, hidden_size)
    void * __restrict__ qkv = nullptr;
    void * __restrict__ q_bias = nullptr;
    void * __restrict__ k_bias = nullptr;
    void * __restrict__ v_bias = nullptr;
    void * __restrict__ o = nullptr;
    // The len of every sequence in the batch.
    void * __restrict__ seq_len = nullptr;

    // The cache for the Ks. The shape must be (Batch, Layer, Seq_len, nheads_k, headdim)
    void * __restrict__ k_cache_table = nullptr;
    // The cache for the Vs. The shape must be (Batch, Layer, Seq_len, nheads_k, headdim)
    void * __restrict__ v_cache_table = nullptr;

    // The table for cos and sin the rotary coef, shape: (seqlen, headdim/2)
    void * __restrict__ rotary_cos_table = nullptr;
    void * __restrict__ rotary_sin_table = nullptr;

    // The batch size.
    int batch_size = 0;
    // The sequence length.
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // The hidden dimension per head (Dh).
    int head_dim = 0;
    // The inv of the hidden dimension per head.
    float head_dim_inv = 0;
    // The per-head latent space reserved for rotary embeddings.
    int rotary_embedding_dim = 0;
    // The maximum length of input sentences.
    int max_input_length = 0;
    // The stride of the input QKV.
    int stride = 0;
    // The num of layer
    int num_layer = 0;
    // The current layer
    int idx_layer = 0;
};

struct Flash_decoder_params
{
    int kBlockN; // 128
    int num_splits;
    int kNThreads;
};

struct Flash_decoder_buffers
{
    // Shape: (batch_size, num_heads, n_split, head_dim)
    void *o_split = nullptr;
    // Shape: (batch_size, num_heads, n_split)
    void *ell = nullptr;
    // shape: (batch_size, num_heads, n_split)
    void *m_formula = nullptr;
};