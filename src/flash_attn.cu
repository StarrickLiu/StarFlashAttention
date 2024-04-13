#include <src/flash_attn.h>
#include <src/utils.h>

template <typename T, typename Traits>
__global__ void flash_decoder_kernel(Flash_decoder_input<T> &input, Flash_decoder_params &params)
{
    extern __shared__ char smem[];
    const int memory_max_len = input.memory_max_len;
    const int num_heads = input.num_heads;
    const int head_dim = input.head_dim;
    const int qkv_stride = input.stride;
    const int num_layer = input.num_layer;
    const int idx_layer = input.idx_layer;
    const int rot_embed_dim = input.rotary_embedding_dim;
    const int bidb = blockIdx.z / num_heads;
    const int bidh = blockIdx.z - bidb * num_heads;
    const int tidx = threadIdx.x;

    const int seq_len = input.seq_len[bidb];

    const int n_split_idx = blockIdx.y;
    const int num_splits = params.num_splits;

    const int n_elem_per_blockN = params.kBlockN; // 128

    const int n_blockN = (seq_len + n_elem_per_blockN - 1) / n_elem_per_blockN;
    const int n_block_per_split = (n_blockN + num_splits - 1) / num_splits;

    // 定位到起始和结束Block
    const int n_block_min = n_split_idx * n_block_per_split;
    const int n_block_max = std::min((seq_len + n_elem_per_blockN - 1) / n_elem_per_blockN, (n_split_idx + 1) * n_block_per_split);
    int n_block = n_block_max - 1;
    int memory_buffer_idx = 0;

    // 获得当前batch,head下的qkv的起始位置
    // qkv_stride == 3 * num_heads * head_dim
    // QKV (batch_size, 3, num_heads, head_dim)
    const uint32_t offset_q = bidb * qkv_stride + bidh * head_dim;
    const uint32_t offset_k = bidb * qkv_stride + num_heads * head_dim + bidh * head_dim;
    const uint32_t offset_v = bidb * qkv_stride + 2 * num_heads * head_dim + bidh * head_dim;
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("[DEBUG] Here3 \n");
    T *gQ = input.qkv + offset_q;
    T *gK = input.qkv + offset_k;
    T *gV = input.qkv + offset_v;
    T *gO;
    if (num_splits == 1)
    {
        // o_split (batch_size, num_heads, head_dim)
        const uint32_t offset_o = bidb * num_heads * head_dim + bidh * head_dim;
        gO = input.o + offset_o;
    }
    else
    {
        // o_split (batch_size, num_heads, num_splits, head_dim)
        const uint32_t offset_o = bidb * num_heads * num_splits * head_dim + bidh * num_splits * head_dim + n_split_idx * head_dim;
        gO = input.o_split + offset_o;
    }

    T *gKCache = input.k_cache_table;
    T *gVCache = input.v_cache_table;

    T *sQ = reinterpret_cast<T *>(smem);           // (1, head_dim)
    T *sK = sQ + head_dim;                         // (2 * n_elem_per_blockN, head_dim)
    T *sV = sK + 2 * n_elem_per_blockN * head_dim; // (2 * n_elem_per_blockN, head_dim)

    const int n_elem_per_thread = Traits::n_elem_per_thread;
    T rS[Traits::n_elem_per_thread];
    T rP[Traits::n_elem_per_thread];
    T rO[Traits::n_elem_per_thread];
    clear_reg(rO, n_elem_per_thread);
    T rEll = 0;

    T m1_in_formula;
    T m2_in_formula;

    // TODO: 添加偏置
    // TODO: 先将QKV拷贝到shared memory中
    // Convert gQ and gK to uint4 pointers (向量化读取)
    uint4 *gQ_uint4 = reinterpret_cast<uint4 *>(gQ);
    uint4 *gK_uint4 = reinterpret_cast<uint4 *>(gK);
    uint4 *gV_uint4 = reinterpret_cast<uint4 *>(gV);
    const uint32_t offset_cache = bidb * num_layer * memory_max_len * num_heads * head_dim + idx_layer * memory_max_len * num_heads * head_dim;

    // printf("[DEBUG] Here\n");
    if (rot_embed_dim != 0)
    {
        // 先将gQ和gK向量化拷贝到smem的特定位置后，再进行旋转编码，在后面直接使用，原因是内核写回全局内存有延迟，存入后再读数据不正确
        // 顺手把相应的gV也拷贝到smem特定位置并存入gVCache中
        if (tidx * 8 < head_dim)
        {
            int cur_len = (seq_len - 1) % n_elem_per_blockN;
            //printf("offset: %d, max_offset: %d cur_len: %d, n_elem_per_blockN: %d\n", memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim + tidx * 8, 2 * n_elem_per_blockN * head_dim, cur_len, n_elem_per_blockN);
            reinterpret_cast<uint4 *>(sQ)[tidx] = gQ_uint4[tidx];
            reinterpret_cast<uint4 *>(sK + memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim)[tidx] = gK_uint4[tidx];
            reinterpret_cast<uint4 *>(sV + memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim)[tidx] = gV_uint4[tidx];
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0)
            {
                printf("[DEBUG] before rotary: %f %f %f %f %f %f %f %f\n",
                       __half2float(sQ[tidx * 8]), __half2float(sQ[tidx * 8 + 1]), __half2float(sQ[tidx * 8 + 2]), __half2float(sQ[tidx * 8 + 3]), __half2float(sQ[tidx * 8 + 4]), __half2float(sQ[tidx * 8 + 5]), __half2float(sQ[tidx * 8 + 6]), __half2float(sQ[tidx * 8 + 7]));
            }
            apply_rotary_embedding(reinterpret_cast<uint4 *>(sQ)[tidx], reinterpret_cast<uint4 *>(sK + memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim)[tidx], tidx, rot_embed_dim, seq_len);
            // gQ_uint4[tidx] = reinterpret_cast<uint4 *>(sQ)[tidx];
            // gK_uint4[tidx] = reinterpret_cast<uint4 *>(sK + memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim)[tidx];
            
            uint4 *gKCache_uint4 = reinterpret_cast<uint4 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + cur_len) * num_heads * head_dim + bidh * head_dim);
            uint4 *gVCache_uint4 = reinterpret_cast<uint4 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + cur_len) * num_heads * head_dim + bidh * head_dim);
            gKCache_uint4[tidx] = reinterpret_cast<uint4 *>(sK + memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim)[tidx];
            gVCache_uint4[tidx] = reinterpret_cast<uint4 *>(sV + memory_buffer_idx * n_elem_per_blockN * head_dim + cur_len * head_dim)[tidx];
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0)
            {
                printf("[DEBUG] after rotary: %f %f %f %f %f %f %f %f\n",
                       __half2float(sQ[tidx * 8]), __half2float(sQ[tidx * 8 + 1]), __half2float(sQ[tidx * 8 + 2]), __half2float(sQ[tidx * 8 + 3]), __half2float(sQ[tidx * 8 + 4]), __half2float(sQ[tidx * 8 + 5]), __half2float(sQ[tidx * 8 + 6]), __half2float(sQ[tidx * 8 + 7]));
            }
        }
    }
    // 获得当前batch,layer下的KV cache的起始位置
    // CacheTable (Batch, num_layer, memory_max_len, num_heads, headdim)

    bool is_first = true;
    __syncthreads();
    // TODO: 异步拷贝优化 DoubleBuffer优化
    // 处理最后一个seqLen不能被整除的块
    if (n_split_idx == num_splits - 1 && seq_len % n_elem_per_blockN != 0)
    {
        const int actual_seq_len = seq_len % n_elem_per_blockN;
        // KV从gCache向量化拷贝到sK和sV中，只拷贝(seq_len % n_elem_per_block, head_dim)的数据，每个线程拷贝8个元素
        for (int i = 0; i < n_elem_per_blockN && i < actual_seq_len - 1; i++)
        {
            if (tidx * 8 < head_dim)
            {
                uint4 *gKCache_uint4 = reinterpret_cast<uint4 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 8);
                uint4 *gVCache_uint4 = reinterpret_cast<uint4 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 8);
                reinterpret_cast<uint4 *>(sK + memory_buffer_idx * n_elem_per_blockN * head_dim + i * head_dim)[tidx] = gKCache_uint4[0];
                reinterpret_cast<uint4 *>(sV + memory_buffer_idx * n_elem_per_blockN * head_dim + i * head_dim)[tidx] = gVCache_uint4[0];
            }
        }
        __syncthreads();
        // 计算Attention中的Q * K^T，Q (1, H) K^T (H, S) Score (1, S)，因此采用GEMV
        // 假设共有32个Thread, Q(1, 128), K^T (128, 128) Score (1, 128), 所有数据已经在Share Memory中
        // 考虑1个Thread最多处理8个元素，但Score只有128个元素，因此每个Thread最多负责4个元素
        clear_reg(rS, n_elem_per_thread);
        gemv_qk(sK, sQ, rS, head_dim, actual_seq_len, tidx, n_elem_per_thread);
        // 现在每个线程的rS都有4个元素，代表tid*4 - (tid+1)*4 的相乘累加结果，现执行Softmax，先找到最大值
        // 由于S已经清零，且e的任意幂均为正数，因此不需要考虑越位问题
        // TODO: 转换到FP32做softmax
        T maxVal = rS[0];
        for (int i = 1; i < n_elem_per_thread; i++)
        {
            maxVal = __max(maxVal, rS[i]);
        }
        m1_in_formula = warpReduceMax(maxVal, 32);
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP[i] = __exp(rS[i] - m1_in_formula);
            rEll += rP[i];
        }
        // 对P(1,S) V(S,H)进行Gemv，此时每个线程拥有自己的P，而V在ShareMemory上，最终把结果写入rO中
        gemv_pv(rP, sV, rO, actual_seq_len, head_dim, tidx, n_elem_per_thread, is_first, m1_in_formula, m2_in_formula);
        is_first = false;
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        printf("[DEBUG] Here 7\n");
    for (; n_block >= n_block_min; --n_block)
    {
        // 拷贝KV到sK和sV中，每个线程拷贝8个元素
        for (int i = 0; i < n_elem_per_blockN - int(is_first); i++)
        {
            // printf("[tidx %d]\n", i);
            if (tidx * 8 < head_dim)
            {
                uint4 *gKCache_uint4 = reinterpret_cast<uint4 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim);
                uint4 *gVCache_uint4 = reinterpret_cast<uint4 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim);
                // (Batch, num_layer, memory_max_len, num_heads, headdim)
                // printf("offset: [%d \\ %d]\n", offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 8, gridDim.z * num_layer * memory_max_len * head_dim);
                // printf("gK[%d]: %f\n", offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 8, __half2float(gKCache[offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 8]));
                // printf("sK[%d]: %f\n", memory_buffer_idx * n_elem_per_blockN * head_dim + i * head_dim + tidx *8, __half2float(sK[memory_buffer_idx * n_elem_per_blockN * head_dim + i * head_dim + tidx*8]));
                reinterpret_cast<uint4 *>(sK + memory_buffer_idx * n_elem_per_blockN * head_dim + i * head_dim)[tidx] = gKCache_uint4[tidx];
                reinterpret_cast<uint4 *>(sV + memory_buffer_idx * n_elem_per_blockN * head_dim + i * head_dim)[tidx] = gVCache_uint4[tidx];
                // printf("!!!!!\n");
            }
        }
        __syncthreads();
        // 计算Attention中的Q * K^T，Q (1, H) K^T (H, S) Score (1, S)，因此采用GEMV
        // 假设共有32个Thread, Q(1, 128), K^T (128, 128) Score (1, 128), 所有数据已经在Share Memory中
        // 考虑1个Thread最多处理8个元素，但Score只有128个元素，因此每个Thread最多负责4个元素
        clear_reg(rS, n_elem_per_thread);
        gemv_qk(sK, sQ, rS, head_dim, n_elem_per_blockN, tidx, n_elem_per_thread);
        T maxVal = rS[0];
        for (int i = 1; i < n_elem_per_thread; i++)
        {
            maxVal = __max(maxVal, rS[i]);
        }
        if (is_first)
        {

            m2_in_formula = __max(warpReduceMax(maxVal, 32), m1_in_formula);
        }
        else
        {

            m2_in_formula = warpReduceMax(maxVal, 32);
        }
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP[i] = __exp(rS[i] - m1_in_formula);
            if (is_first)
            {
                rEll = rP[i];
            }
            else
            {
                rEll = __exp(m1_in_formula - m2_in_formula) * rEll + rP[i];
            }
        }
        // 对于sO先修正再加和，因此不用再置0
        gemv_pv(rP, sV, rO, n_elem_per_blockN, head_dim, tidx, n_elem_per_thread, is_first, m1_in_formula, m2_in_formula);
        is_first = false;
        m1_in_formula = m2_in_formula;
    }
    rEll = warpReduceSum(rEll, 32);
    // rO向量化拷贝回全局内存中，每个线程拷贝4个元素
    reinterpret_cast<uint2 *>(gO)[tidx] = reinterpret_cast<uint2 *>(rO)[0];

    if (num_splits > 1)
    {
        // input.ell Shape: (batch_size, num_heads, n_split)
        // input.m_formula shape: (batch_size, num_heads, n_split)
        input.ell[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx] = rEll;
        input.m_formula[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx] = m1_in_formula;
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        printf("[DEBUG] Here End\n");
}

template <typename T, typename Traits>
__global__ void flash_combine_kernel(Flash_decoder_input<T> &input, Flash_decoder_params &params)
{
    const int n_split = params.num_splits;
    const int num_heads = input.num_heads;
    const int head_dim = input.head_dim;

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;

    const int num_per_thread = Traits::n_elem_per_thread;

    // Shape: (batch_size, num_heads, n_split)
    T *gEll = input.ell;
    T *gM = input.m_formula;
    T *gO_split = input.o_split;
    // Shape: (batch_size, hidden_size)
    T *gO = input.o;

    // 将当前batch、head的所有Ell拷贝至寄存器中
    T rM = -256;
    for (int i = 0; i < n_split; i++)
    {
        rM = __max(rM, gM[bidb * num_heads * n_split + bidh * n_split + i]);
    }
    // 将Ell修正后加和
    T rEll = 0;
    for (int i = 0; i < n_split; i++)
    {
        rEll += __exp(gM[bidb * num_heads * n_split + bidh * n_split + i] - rM) * gEll[bidb * num_heads * n_split + bidh * n_split + i];
    }

    T rO[Traits::n_elem_per_thread];
    clear_reg(rO, num_per_thread);
    // 每个线程沿n_split方向遍历，对O修正后加和在rO中
    for (int i = 0; i < n_split; i++)
    {
        // 每个线程负责N个Thread，在这里是FP16，因此每个线程负责4个元素，使用uint2向量化访存
        // Shape: (batch_size, num_heads, n_split, head_dim)
        uint2 *rO_split = reinterpret_cast<uint2 *>(gO_split + bidb * num_heads * n_split * head_dim + bidh * n_split * head_dim + i * head_dim + threadIdx.x * num_per_thread);
        for (int j = 0; j < num_per_thread; j++)
        {
            rO[j] += __exp(gM[bidb * num_heads * n_split + bidh * n_split + i] - rM) * reinterpret_cast<T *>(rO_split)[j];
        }
    }
    // 除以ell后，将结果写回全局内存
    for (int i = 0; i < num_per_thread; i++)
    {
        rO[i] /= rEll;
    }

    // rO向量化拷贝回全局内存中，每个线程拷贝4个元素
    reinterpret_cast<uint2 *>(gO + bidb * num_heads * head_dim + bidh * head_dim + threadIdx.x * num_per_thread)[0] = reinterpret_cast<uint2 *>(rO)[0];
}

template <typename T>
void run_flash_decoder(Flash_decoder_input<T> *input, Flash_decoder_params *params, cudaStream_t stream)
{
    using traits = Traits<half, 128, 64>;
    // grid (1, num_splits, b*h)
    // block(kNThreads) 128 = 4 Warps
    Flash_decoder_input<T> *d_input;
    Flash_decoder_params *d_params;

    cudaMalloc((void **)&d_input, sizeof(Flash_decoder_input<T>));
    cudaMalloc((void **)&d_params, sizeof(Flash_decoder_params));

    // Copy data from host to device
    cudaMemcpy(d_input, input, sizeof(Flash_decoder_input<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params, sizeof(Flash_decoder_params), cudaMemcpyHostToDevice);

    dim3 grid(1, params->num_splits, input->batch_size * input->num_heads);
    dim3 block(params->kNThreads);
    cudaError_t err;
    printf("Share Memory Size: %d\n", traits::smemSize);
    if (traits::smemSize >= 48 * 1024)
    {
        cudaFuncSetAttribute(
            flash_decoder_kernel<T, traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::smemSize);
        cudaFuncSetAttribute(
            flash_combine_kernel<T, traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::smemSize);
    }
    flash_decoder_kernel<T, traits><<<grid, block, traits::smemSize, stream>>>(*d_input, *d_params);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA错误: %s\n", cudaGetErrorString(err));
        return; // 发生错误，提前退出函数
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA设备同步错误: %s\n", cudaGetErrorString(err));
        return; // 发生错误，提前退出函数
    }

    dim3 grid_combine(input->batch_size, input->num_heads);
    dim3 block_combine(params->kNThreads);

    flash_combine_kernel<T, traits><<<grid_combine, block_combine, traits::smemSize, stream>>>(*d_input, *d_params);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA错误 (组合核函数): %s\n", cudaGetErrorString(err));
        return; // 发生错误，提前退出函数
    }
}

template __global__ void flash_decoder_kernel<half, Traits<half, 128, 64>>(Flash_decoder_input<half> &input, Flash_decoder_params &params);
template __global__ void flash_combine_kernel<half, Traits<half, 128, 64>>(Flash_decoder_input<half> &input, Flash_decoder_params &params);
template void run_flash_decoder<half>(Flash_decoder_input<half> *, Flash_decoder_params *, cudaStream_t);

// TODO: 目前仅针对3090单卡和llama 7B设置分割数，后续需要智能分割