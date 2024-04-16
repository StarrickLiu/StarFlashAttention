#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <src/params.h>
#include <src/flash_attn.h>
#include <cmath>

void set_input(Flash_decoder_input &input,
               at::Tensor qkv, at::Tensor q_bias,
               at::Tensor k_bias, at::Tensor v_bias,
               at::Tensor o,
               at::Tensor k_cache_table, at::Tensor v_cache_table,
               at::Tensor seq_len, int batch_size,
               int memory_max_len, int num_heads,
               int head_dim, int rotary_embedding_dim,
               int max_input_length,
               int num_layer, int idx_layer)
{
    input.qkv = qkv.data_ptr();
    input.o = o.data_ptr();
    input.seq_len = seq_len.data_ptr();
    input.k_cache_table = k_cache_table.data_ptr();
    input.v_cache_table = v_cache_table.data_ptr();
    input.batch_size = batch_size;
    input.memory_max_len = memory_max_len;
    input.num_heads = num_heads;
    input.head_dim = head_dim;
    input.head_dim_inv = 1.0f / std::sqrt(head_dim);
    input.rotary_embedding_dim = rotary_embedding_dim;
    input.max_input_length = max_input_length;
    input.stride = 3 * num_heads * head_dim;
    input.num_layer = num_layer;
    input.idx_layer = idx_layer;
}

void set_default_params(Flash_decoder_params &params)
{
    params.kBlockN = 32;
    params.num_splits = 4;
    params.kNThreads = 32;
}

at::Tensor mha_fwd_cuda(
    at::Tensor &qkv,
    at::Tensor &q_bias,
    at::Tensor &k_bias,
    at::Tensor &v_bias,
    at::Tensor &k_cache_table,
    at::Tensor &v_cache_table,
    at::Tensor &seq_len,
    at::Tensor &o,
    int batch_size,
    int memory_max_len,
    int num_heads,
    int head_dim,
    int rotary_embedding_dim,
    int max_input_length,
    int num_layer,
    int idx_layer)
{

    Flash_decoder_input input;
    set_input(input, qkv, q_bias, k_bias, v_bias, o, k_cache_table, v_cache_table, seq_len, batch_size, memory_max_len, num_heads, head_dim, rotary_embedding_dim, max_input_length, num_layer, idx_layer);
    Flash_decoder_params params;
    set_default_params(params);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_flash_decoder<half>(input, params, stream);
    return o;
}

PYBIND11_MODULE(star_flash_attn, m)
{
    m.doc() = "API for Flash Attention using CUDA"; // optional module docstring

    m.def("mha_fwd_cuda", &mha_fwd_cuda, "A function that performs multi-head attention forward pass using CUDA",
          py::arg("qkv"), py::arg("q_bias"), py::arg("k_bias"), py::arg("v_bias"),
          py::arg("k_cache_table"), py::arg("v_cache_table"), py::arg("seq_len"),
          py::arg("o"), py::arg("batch_size"), py::arg("memory_max_len"),
          py::arg("num_heads"), py::arg("head_dim"), py::arg("rotary_embedding_dim"),
          py::arg("max_input_length"), py::arg("num_layer"), py::arg("idx_layer"));
}