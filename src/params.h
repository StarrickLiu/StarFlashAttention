#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <src/traits.h>

template <typename T>
struct Flash_decoder_input
{

    T *out = nullptr;
    // Shape: (batch_size, 3, hidden_size)
    T *qkv = nullptr;
    T *q_bias = nullptr;
    T *k_bias = nullptr;
    T *v_bias = nullptr;
    T *o = nullptr;
    // Shape: (batch_size, num_heads, n_split, head_dim)
    T *o_split = nullptr;
    // Shape: (batch_size, num_heads, n_split)
    T *ell = nullptr;
    // shape: (batch_size, num_heads, n_split)
    T *m_formula = nullptr;

    // The cache for the Ks. The shape must be (Batch, Layer, Seq_len, nheads_k, headdim)
    T *k_cache_table = nullptr;
    // The cache for the Vs. The shape must be (Batch, Layer, Seq_len, nheads_k, headdim)
    T *v_cache_table = nullptr;

    // The batch size.
    int batch_size = 0;
    // The sequence length.
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // The hidden dimension per head (Dh).
    int head_dim = 0;
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
    // The len of every sequence in the batch.
    int *seq_len = nullptr;
};

struct Flash_decoder_params
{
    int kBlockN; // 128
    int num_splits;
    int combine_parallel;
    int kNThreads;
    int kNWarps;
};