#include <cuda_runtime.h>
#include <iostream>
#include <src/flash_attn.h>
#include <src/utils.h>
#include "cuda_fp16.h"

int main()
{
    const int batch_size = 1;
    const int num_heads = 32;
    const int head_dim = 128;
    const int seq_len = 512;
    const int max_seq_len = 1024;
    const int num_splits = 4;
    const int kNThreads = 32;
    const int num_layer = 4;
    const int idx_layer = 0;

    // 分配并初始化输入和参数结构
    Flash_decoder_input<half> *input = new Flash_decoder_input<half>;
    Flash_decoder_params *params = new Flash_decoder_params;

    // 设置参数
    params->num_splits = num_splits;
    params->kNThreads = kNThreads;
    params->kBlockN = 64;

    // Setup input sizes
    input->batch_size = batch_size;
    input->num_heads = num_heads;
    input->head_dim = head_dim;
    input->memory_max_len = max_seq_len;
    input->max_input_length = max_seq_len;
    input->rotary_embedding_dim = head_dim;
    input->stride = 3 * num_heads * head_dim;
    input->idx_layer = idx_layer;
    input->num_layer = num_layer;

    size_t qkv_matrix_size = 3 * batch_size * num_heads * seq_len * head_dim * sizeof(half);
    size_t matrix_size = batch_size * num_heads * seq_len * head_dim * sizeof(half);
    size_t split_matrix_size = batch_size * num_heads * num_splits * head_dim * sizeof(half);
    size_t vector_size = batch_size * num_heads * num_splits * sizeof(half);
    size_t cache_size = batch_size * num_layer * max_seq_len * num_heads * head_dim * sizeof(half);

    std::cout << "[INFO]"
              << " 内存开始分配" << std::endl;
    // 分配设备内存
    cudaMalloc((void **)&input->qkv, qkv_matrix_size);
    cudaMalloc((void **)&input->o, matrix_size);
    cudaMalloc((void **)&input->o_split, split_matrix_size);
    cudaMalloc((void **)&input->ell, vector_size);
    cudaMalloc((void **)&input->m_formula, vector_size);
    cudaMalloc((void **)&input->k_cache_table, cache_size);
    cudaMalloc((void **)&input->v_cache_table, cache_size);
    cudaMalloc((void **)&input->seq_len, batch_size * sizeof(int));

    // 创建大小为seq_len的host数组，拷贝到device上的input->seq_len中
    int *seq_len_host = new int[batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        seq_len_host[i] = seq_len;
    }
    cudaMemcpy(input->seq_len, seq_len_host, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    delete[] seq_len_host;

    half one = __float2half(1.0);
    // 用1初始化矩阵
    int num_elements = batch_size * 3 * num_heads * head_dim; // 总元素数量

    // 设置CUDA内核的执行配置
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    // 初始化数组
    init_half_array(input->qkv, one, num_elements, numBlocks, blockSize);

    // (batch_size * num_layer * max_seq_len * num_heads * head_dim)
    num_elements = batch_size * num_layer * max_seq_len * num_heads * head_dim;
    numBlocks = (num_elements + blockSize - 1) / blockSize;
    init_half_array(input->k_cache_table, one, num_elements, numBlocks, blockSize);
    init_half_array(input->v_cache_table, one, num_elements, numBlocks, blockSize);
    std::cout << "[INFO]"
              << " 内存分配完毕且初始化完毕" << std::endl;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "[INFO]"
              << " 测试开始" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error before kernel: %s\n", cudaGetErrorString(error));
    }
    // 运行解码器
    run_flash_decoder<half>(input, params, stream);

    std::cout << "[INFO]"
              << " 测试结束" << std::endl;

    // 同步CUDA流以确保所有操作完成
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // 清理资源
    cudaFree(input->qkv);
    cudaFree(input->o);
    cudaFree(input->o_split);
    cudaFree(input->ell);
    cudaFree(input->m_formula);

    delete input;
    delete params;

    return 0;
}
