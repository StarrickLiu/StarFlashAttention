#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

template<typename elem_type, int head_dim, int n_elem_per_blockN>
struct Traits {
    static constexpr int warp_size = 32;
    static constexpr int n_elem_per_vec = 16 / sizeof(elem_type);
    static constexpr int n_elem_per_thread = head_dim / warp_size;

    // T *sQ = reinterpret_cast<T *>(smem);           // (1, head_dim)
    // T *sK = sQ + head_dim;                         // (2 * n_elem_per_blockN, head_dim)
    // T *sV = sK + 2 * n_elem_per_blockN * head_dim; // (2 * n_elem_per_blockN, head_dim)
    static constexpr int smemQ = (head_dim + 1) * sizeof(elem_type);
    static constexpr int smemK = n_elem_per_blockN * (head_dim + 4) * sizeof(elem_type);
    static constexpr int smemV = n_elem_per_blockN * (head_dim + 4) * sizeof(elem_type);
    static constexpr int smemSize = smemQ + smemK + smemV;
};

// Instantiate Traits for half type and different head dimensions