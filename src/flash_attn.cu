#include <src/flash_attn.h>
#include <src/utils.h>

template <typename T, typename Traits>
__global__ void flash_decoder_kernel(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers)
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

    const int seq_len = reinterpret_cast<int *>(input.seq_len)[bidb];

    const int n_split_idx = blockIdx.y;
    const int num_splits = params.num_splits;
    bool last_split_is_first = n_split_idx == num_splits - 1;
    bool is_first = true;

    const int n_elem_per_blockN = params.kBlockN; // 128

    const int n_blockN = (seq_len + n_elem_per_blockN - 1) / n_elem_per_blockN;
    const int n_block_per_split = (n_blockN + num_splits - 1) / num_splits;

    // 定位到起始和结束Block
    const int n_block_min = n_split_idx * n_block_per_split;
    const int n_block_max = std::min((seq_len + n_elem_per_blockN - 1) / n_elem_per_blockN, (n_split_idx + 1) * n_block_per_split);
    int n_block = n_block_max - 1;

    // 获得当前batch,head下的qkv的起始位置
    // qkv_stride == 3 * num_heads * head_dim
    // QKV (batch_size, 3, num_heads, head_dim)
    const uint32_t offset_q = bidb * qkv_stride + bidh * head_dim;
    const uint32_t offset_k = bidb * qkv_stride + num_heads * head_dim + bidh * head_dim;
    const uint32_t offset_v = bidb * qkv_stride + 2 * num_heads * head_dim + bidh * head_dim;
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("[DEBUG] Here3 \n");
    T *gQ = reinterpret_cast<T *>(input.qkv) + offset_q;
    T *gK = reinterpret_cast<T *>(input.qkv) + offset_k;
    T *gV = reinterpret_cast<T *>(input.qkv) + offset_v;
    T *gO;
    if (num_splits == 1)
    {
        // o_split (batch_size, num_heads, head_dim)
        const uint32_t offset_o = bidb * num_heads * head_dim + bidh * head_dim;
        gO = reinterpret_cast<T *>(input.o) + offset_o;
    }
    else
    {
        // o_split (batch_size, num_heads, num_splits, head_dim)
        const uint32_t offset_o = bidb * num_heads * num_splits * head_dim + bidh * num_splits * head_dim + n_split_idx * head_dim;
        gO = reinterpret_cast<T *>(buffers.o_split) + offset_o;
    }

    T *gKCache = reinterpret_cast<T *>(input.k_cache_table);
    T *gVCache = reinterpret_cast<T *>(input.v_cache_table);

    T *sQ = reinterpret_cast<T *>(smem);       // (1, head_dim)
    T *sK = sQ + head_dim;                     // (n_elem_per_blockN, head_dim)
    T *sV = sK + n_elem_per_blockN * head_dim; // (n_elem_per_blockN, head_dim)

    const int n_elem_per_thread = Traits::n_elem_per_thread;
    T rS[Traits::n_elem_per_thread];
    T rP_1[Traits::n_elem_per_thread];
    T rP_2[Traits::n_elem_per_thread];
    T rO[Traits::n_elem_per_thread];
    clear_reg(rO, n_elem_per_thread);
    T rEll_1 = 0;
    T rEll_2 = 0;
    T rEll_2_inv = 0;

    T m1_in_formula;
    T m2_in_formula;

    // TODO: 添加偏置
    // Convert gQ and gK to uint2 pointers (向量化读取)
    uint2 *gQ_uint2 = reinterpret_cast<uint2 *>(gQ);
    uint2 *gK_uint2 = reinterpret_cast<uint2 *>(gK);
    uint2 *gV_uint2 = reinterpret_cast<uint2 *>(gV);
    const uint32_t offset_cache = bidb * num_layer * memory_max_len * num_heads * head_dim + idx_layer * memory_max_len * num_heads * head_dim;

    if (rot_embed_dim != 0 && n_split_idx == num_splits - 1)
    {
        // 对于最后一个spilit，先将gQ和gK向量化拷贝到smem的特定位置后，再进行旋转编码，在后面直接使用，原因是内核写回全局内存有延迟，存入后再读数据不正确
        // 顺手把相应的gV也拷贝到smem特定位置并存入gVCache中
        // if (tidx * 8 < head_dim)
        // {
        int cur_len = (seq_len - 1) % n_elem_per_blockN;
        // printf("offset: %d, max_offset: %d cur_len: %d, n_elem_per_blockN: %d\n", cur_len * head_dim + tidx * 8, 2 * n_elem_per_blockN * head_dim, cur_len, n_elem_per_blockN);
        reinterpret_cast<uint2 *>(sQ)[tidx] = gQ_uint2[tidx];
        reinterpret_cast<uint2 *>(sK + cur_len * head_dim)[tidx] = gK_uint2[tidx];
        reinterpret_cast<uint2 *>(sV + cur_len * head_dim)[tidx] = gV_uint2[tidx];
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31 && tidx == 0)
        // {
        //     printf("[DEBUG] before rotary: %f %f %f %f\n",
        //            __half2float(sQ[tidx * 4]), __half2float(sQ[tidx * 4 + 1]), __half2float(sQ[tidx * 4 + 2]), __half2float(sQ[tidx * 4 + 3]));
        // }
        apply_rotary_embedding(reinterpret_cast<uint2 *>(sQ)[tidx], reinterpret_cast<uint2 *>(sK + cur_len * head_dim)[tidx], tidx, rot_embed_dim, seq_len - 1);
        // gQ_uint2[tidx] = reinterpret_cast<uint2 *>(sQ)[tidx];
        // gK_uint2[tidx] = reinterpret_cast<uint2 *>(sK + cur_len * head_dim)[tidx];

        uint2 *gKCache_uint2 = reinterpret_cast<uint2 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + cur_len) * num_heads * head_dim + bidh * head_dim);
        uint2 *gVCache_uint2 = reinterpret_cast<uint2 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + cur_len) * num_heads * head_dim + bidh * head_dim);
        gKCache_uint2[tidx] = reinterpret_cast<uint2 *>(sK + cur_len * head_dim)[tidx];
        gVCache_uint2[tidx] = reinterpret_cast<uint2 *>(sV + cur_len * head_dim)[tidx];
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31 && tidx == 0)
        // {
        //     printf("[DEBUG] after rotary: %f %f %f %f\n",
        //            __half2float(sQ[tidx * 4]), __half2float(sQ[tidx * 4 + 1]), __half2float(sQ[tidx * 4 + 2]), __half2float(sQ[tidx * 4 + 3]));
        // }
        // }
    }
    else
    {
        // 非最后一个split只对Q进行旋转编码
        // printf("offset: %d, max_offset: %d cur_len: %d, n_elem_per_blockN: %d\n", cur_len * head_dim + tidx * 8, 2 * n_elem_per_blockN * head_dim, cur_len, n_elem_per_blockN);
        reinterpret_cast<uint2 *>(sQ)[tidx] = gQ_uint2[tidx];
        apply_rotary_embedding(reinterpret_cast<uint2 *>(sQ)[tidx], tidx, rot_embed_dim, seq_len - 1);
    }

    // 获得当前batch,layer下的KV cache的起始位置
    // CacheTable (Batch, num_layer, memory_max_len, num_heads, headdim)
    // __syncthreads();
    // TODO: DoubleBuffer优化
    // 处理最后一个seqLen不能被整除的块
    if (n_split_idx == num_splits - 1 && seq_len % n_elem_per_blockN != 0)
    {
        const int actual_seq_len = seq_len % n_elem_per_blockN;
        // KV从gCache向量化拷贝到sK和sV中，只拷贝(seq_len % n_elem_per_block, head_dim)的数据，每个线程拷贝8个元素
        // 现改为每个线程执行n_elem_per_blockN个seq的拷贝，且进行异步拷贝，需要使用数据时再确保已经拷贝结束
        // TODO: n_elem_per_thread有混淆，需要分成两个量，分别针对head_dim和kBlockN
        for (int i = tidx * n_elem_per_thread; i < (tidx + 1) * n_elem_per_thread && i < actual_seq_len - 1; i++)
            for (int j = 0; j < head_dim / 8; j++)
            {
                
                // uint2 *gKCache_uint2 = reinterpret_cast<uint2 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 4);
                // uint2 *gVCache_uint2 = reinterpret_cast<uint2 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 4);
                // reinterpret_cast<uint2 *>(sK + i * head_dim)[tidx] = gKCache_uint2[0];
                // reinterpret_cast<uint2 *>(sV + i * head_dim)[tidx] = gVCache_uint2[0];
                cp_async_cg<half, 16>(sK + i * head_dim + j * 8, gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
                cp_async_cg<half, 16>(sV + i * head_dim + j * 8, gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
            }
        // __syncthreads();
        // 计算Attention中的Q * K^T，Q (1, H) K^T (H, S) Score (1, S)，因此采用GEMV
        // 假设共有32个Thread, Q(1, HeadDim), K^T (HeadDim, n_elem_per_blockN) Score (1, n_elem_per_blockN), 所有数据已经在Share Memory中
        // 考虑1个Thread最多处理8个元素，但n_elem_per_blockN有128个元素，因此每个Thread最多负责4个元素
        clear_reg(rS, n_elem_per_thread);
        gemv_qk(sK, sQ, rS, actual_seq_len, head_dim, tidx, n_elem_per_thread, input.head_dim_inv);
        // 现在每个线程的rS都有4个元素，代表tid*4 - (tid+1)*4 的相乘累加结果，现执行Softmax，先找到最大值
        // 由于S已经清零，且e的任意幂均为正数，因此不需要考虑越位问题
        // TODO: 转换到FP32做softmax
        T maxVal = rS[0];
#pragma unroll
        for (int i = 1; i < n_elem_per_thread; i++)
        {
            maxVal = __max(maxVal, rS[i]);
        }
        m1_in_formula = warpReduceMax(maxVal, 32);
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP_2[i] = __exp(rS[i] - m1_in_formula);
            rEll_1 += rP_2[i];
        }
        rEll_2 = rEll_1;
        rEll_1 = 0;
        // 对P(1,S) V(S,H)进行Gemv，此时每个线程拥有自己的P，而V在ShareMemory上，最终把结果写入rO中
        gemv_pv(rP_2, sV, rO, actual_seq_len, head_dim, tidx, n_elem_per_thread, is_first, m1_in_formula, m2_in_formula);
        is_first = false;
        last_split_is_first = false;
    }
#pragma unroll
    for (; n_block >= n_block_min; --n_block)
    {
                
                // uint2 *gKCache_uint2 = reinterpret_cast<uint2 *>(gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 4);
                // uint2 *gVCache_uint2 = reinterpret_cast<uint2 *>(gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 4);
                // reinterpret_cast<uint2 *>(sK + i * head_dim)[tidx] = gKCache_uint2[0];
                // reinterpret_cast<uint2 *>(sV + i * head_dim)[tidx] = gVCache_uint2[0];
        for (int i = tidx * n_elem_per_thread; i < (tidx + 1) * n_elem_per_thread - int(last_split_is_first); i++)
            for (int j = 0; j < head_dim / 8; j++)
            {
                cp_async_cg<half, 16>(sK + i * head_dim + j * 8, gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
            }

        // TODO: 整合进入qk后看看性能
        // for (int i = tidx * n_elem_per_thread; i < (tidx + 1) * n_elem_per_thread - int(last_split_is_first); i++)
        //     for (int j = 0; j < head_dim / 8; j++)
        //     {
        //         cp_async_cg<half, 16>(sK + i * head_dim + j * 8, gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
        //     }
        // 保证全局连续性，修改为一次拷贝两个seq
        // for (int i = 0; i < n_elem_per_blockN; i+= 2) {
        //     if (tidx < 16)
        //         cp_async_cg<half, 16>(sK + i * head_dim + tidx * 8, gKCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + tidx * 8);
        //     else
        //         cp_async_cg<half, 16>(sK + i * head_dim + tidx * 8, gKCache + offset_cache + (n_block * n_elem_per_blockN + i + 1) * num_heads * head_dim + bidh * head_dim + (tidx-16) * 8);
        // }
        // __syncthreads();
        // 计算Attention中的Q * K^T，Q (1, H) K^T (H, S) Score (1, S)，因此采用GEMV
        // 假设共有32个Thread, Q(1, HeadDim), K^T (HeadDim, n_elem_per_blockN) Score (1, n_elem_per_blockN), 所有数据已经在Share Memory中
        // 考虑1个Thread最多处理8个元素，但n_elem_per_blockN有128个元素，因此每个Thread最多负责4个元素
        CP_ASYNC_COMMIT_GROUP();
        clear_reg(rS, n_elem_per_thread);

        CP_ASYNC_WAIT_ALL();
#pragma unroll
        for (int i = tidx * n_elem_per_thread; i < (tidx + 1) * n_elem_per_thread - int(last_split_is_first); i++)
#pragma unroll
            for (int j = 0; j < head_dim / 8; j++)
            {
                cp_async_cg<half, 16>(sV + i * head_dim + j * 8, gVCache + offset_cache + (n_block * n_elem_per_blockN + i) * num_heads * head_dim + bidh * head_dim + j * 8);
            }
        CP_ASYNC_COMMIT_GROUP();
        gemv_qk(sK, sQ, rS, n_elem_per_blockN, head_dim, tidx, n_elem_per_thread, input.head_dim_inv);

        T maxVal = rS[0];
#pragma unroll
        for (int i = 1; i < n_elem_per_thread; i++)
        {
            maxVal = __max(maxVal, rS[i]);
        }

        
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31)
        // {
        //     // printf("rEll: %f\n", __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     // printf("m1_in_formula: %f\n", __half2float(reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
            
        //     printf("[1 tidx=%d] m1_in_formula: %f, m2_in_formula: %f, rEll: %f, maxVal: %f\n", tidx, __half2float(m1_in_formula), __half2float(m2_in_formula), __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]), __half2float(maxVal));
        // }
        if (is_first)
        {
            m2_in_formula = warpReduceMax(maxVal, 32);
        }
        else
        {
            m2_in_formula = __max(warpReduceMax(maxVal, 32), m1_in_formula);
        }
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP_1[i] = __exp(rS[i] - m2_in_formula);
            rEll_1 += rP_1[i];
        }

        if (is_first)
        {
            rEll_2 = warpReduceSum(rEll_1, 32);
        }
        else
        {
            rEll_2 = __exp(m1_in_formula - m2_in_formula) * rEll_2 + warpReduceSum(rEll_1, 32);
        }
        rEll_2_inv = T(1) / rEll_2;
        
        for (int i = 0; i < n_elem_per_thread; i++)
        {
            rP_2[i] = rP_1[i] * rEll_2_inv;
        }
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31)
        // {
        //     // printf("rEll: %f\n", __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     // printf("m1_in_formula: %f\n", __half2float(reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     printf("rEll_2_inv: %f, rEll_2: %f, rP_2[%d, %d, %d, %d] = %f, %f, %f, %f\n", __half2float(rEll_2_inv), __half2float(rEll_2), tidx * 4, tidx * 4 + 1, tidx * 4 + 2, tidx * 4 + 3, __half2float(reinterpret_cast<T *>(rP_2)[0]), __half2float(reinterpret_cast<T *>(rP_2)[1]), __half2float(reinterpret_cast<T *>(rP_2)[2]), __half2float(reinterpret_cast<T *>(rP_2)[3]));
        // }
        // 对于sO先修正再加和，因此不用再置0

        CP_ASYNC_WAIT_ALL();
        // gemv_pv(rP_2, sV, rO, n_elem_per_blockN, head_dim, tidx, n_elem_per_thread, is_first, m1_in_formula, m2_in_formula);


        m1_in_formula = m2_in_formula;
        is_first = false;
        last_split_is_first = false;
    }
    // rO向量化拷贝回全局内存中，每个线程拷贝4个元素

    reinterpret_cast<uint2 *>(gO)[tidx] = reinterpret_cast<uint2 *>(rO)[0];
    // printf("[here!]\n");
    if (num_splits > 1)
    {
        // buffers.ell Shape: (batch_size, num_heads, n_split)
        // buffers.m_formula shape: (batch_size, num_heads, n_split)
        // buffers.o_spilit shape: (batch_size, num_heads, n_split, head_dim)
        reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx] = rEll_2;
        reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx] = m1_in_formula;

        // if (blockIdx.x == 0 && blockIdx.y == 2 && threadIdx.x == 0)
        // {
        //     printf("[DEBUG] [head:%d] rEll: %f, m1_in_formula: %f\n", bidh, __half2float(rEll_2), __half2float(m1_in_formula));
        // }
        // if (blockIdx.x == 0 && blockIdx.y == 3 && blockIdx.z == 31)
        // {
        //     // printf("rEll: %f\n", __half2float(reinterpret_cast<T *>(buffers.ell)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     // printf("m1_in_formula: %f\n", __half2float(reinterpret_cast<T *>(buffers.m_formula)[bidb * num_heads * num_splits + bidh * num_splits + n_split_idx]));
        //     printf("[tidx=%d] gO ptr: %p \n", tidx, gO);
        //     // printf("rO[%d, %d, %d, %d] = %f, %f, %f, %f\n", tidx * 4, tidx * 4 + 1, tidx * 4 + 2, tidx * 4 + 3, __half2float(reinterpret_cast<T *>(rO)[0]), __half2float(reinterpret_cast<T *>(rO)[1]), __half2float(reinterpret_cast<T *>(rO)[2]), __half2float(reinterpret_cast<T *>(rO)[3]));
        // }
    }
    // printf("[DEBUG] Here End\n");
}

template <typename T, typename Traits>
__global__ void flash_combine_kernel(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers)
{
    const int n_split = params.num_splits;
    const int num_heads = input.num_heads;
    const int head_dim = input.head_dim;

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;

    const int num_per_thread = Traits::n_elem_per_thread;

    // Shape: (batch_size, num_heads, n_split)
    T *gEll = reinterpret_cast<T *>(buffers.ell);
    T *gM = reinterpret_cast<T *>(buffers.m_formula);
    T *gO_split = reinterpret_cast<T *>(buffers.o_split);
    // Shape: (batch_size, hidden_size)
    T *gO = reinterpret_cast<T *>(input.o);

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
        // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        //     printf("[DEBUG] gM: %f rEll: %f gEll[%d]: %f\n", __half2float(gM[bidb * num_heads * n_split + bidh * n_split + i]), __half2float(rEll), i, __half2float(gEll[bidb * num_heads * n_split + bidh * n_split + i]));
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
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            //     printf("[DEBUG] expxx: %f gM: %f rM: %f rO[%d]: %f rO_acc: %f\n", __half2float(__exp(gM[bidb * num_heads * n_split + bidh * n_split + i] - rM)), __half2float(gM[bidb * num_heads * n_split + bidh * n_split + i]), __half2float(rM), j, __half2float(reinterpret_cast<T *>(rO_split)[j]), __half2float(rO[j]));
        }
    }
    T invREll = half(1) / rEll; // 计算倒数
    // 除以ell后，将结果写回全局内存
    for (int i = 0; i < num_per_thread; i++)
    {
        rO[i] *= invREll;
    }

    // rO向量化拷贝回全局内存中，每个线程拷贝4个元素
    reinterpret_cast<uint2 *>(gO + bidb * num_heads * head_dim + bidh * head_dim + threadIdx.x * num_per_thread)[0] = reinterpret_cast<uint2 *>(rO)[0];
}

template <typename T>
void run_flash_decoder(Flash_decoder_input &input, Flash_decoder_params &params, cudaStream_t stream)
{
    using traits = Traits<half, 128, 128>;
    // grid (1, num_splits, b*h)
    // block(kNThreads) 128 = 4 Warps
    // 先定义一个在设备端使用的结构体指针
    Flash_decoder_buffers buffers;

    // 为设备端结构体分配内存
    cudaMalloc((void **)&buffers, sizeof(Flash_decoder_buffers));

    // 为每个内部指针分别分配内存，并设置设备端结构体的指针
    cudaMalloc(&(buffers.o_split), sizeof(T) * input.batch_size * input.num_heads * params.num_splits * input.head_dim);
    cudaMalloc(&(buffers.ell), sizeof(T) * input.batch_size * input.num_heads * params.num_splits);
    cudaMalloc(&(buffers.m_formula), sizeof(T) * input.batch_size * input.num_heads * params.num_splits);

    dim3 grid(1, params.num_splits, input.batch_size * input.num_heads);
    dim3 block(params.kNThreads);
    cudaError_t err;
    // printf("Share Memory Size: %d\n", traits::smemSize);
    if (traits::smemSize >= 48 * 1024)
    {
        cudaFuncSetAttribute(
            flash_decoder_kernel<T, traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::smemSize);
        cudaFuncSetAttribute(
            flash_combine_kernel<T, traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::smemSize);
    }

    Flash_decoder_input *d_input;
    Flash_decoder_params *d_params;
    Flash_decoder_buffers *d_buffers;

    cudaMalloc((void **)&d_input, sizeof(Flash_decoder_input));
    cudaMalloc((void **)&d_params, sizeof(Flash_decoder_params));
    cudaMalloc((void **)&d_buffers, sizeof(Flash_decoder_buffers));

    cudaMemcpy(d_input, &input, sizeof(Flash_decoder_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(Flash_decoder_params), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffers, &buffers, sizeof(Flash_decoder_buffers), cudaMemcpyHostToDevice);

    // 将d_input->q 拷贝到host中，并输出
    flash_decoder_kernel<T, traits><<<grid, block, traits::smemSize, stream>>>(*d_input, *d_params, *d_buffers);
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

    // T *q_host = new T[input.batch_size * input.num_heads * input.head_dim];
    // cudaMemcpy(q_host, reinterpret_cast<T *>(buffers.o_split) + 31 * params.num_splits * input.head_dim + 3 * input.head_dim, sizeof(T) * input.head_dim, cudaMemcpyDeviceToHost);
    // std::cout << "o_split:" << std::endl;
    // for (int i = 0; i < input.head_dim; i++)
    // {
    //     printf("%f ", __half2float(q_host[i]));
    // }
    // std::cout << std::endl;

    dim3 grid_combine(input.batch_size, input.num_heads);
    dim3 block_combine(params.kNThreads);

    flash_combine_kernel<T, traits><<<grid_combine, block_combine, traits::smemSize, stream>>>(*d_input, *d_params, *d_buffers);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA错误 (组合核函数): %s\n", cudaGetErrorString(err));
    }
    // 销毁d_buffers
    cudaFree(buffers.o_split);
    cudaFree(buffers.ell);
    cudaFree(buffers.m_formula);

    cudaFree(d_input);
    cudaFree(d_params);
}

template __global__ void flash_decoder_kernel<half, Traits<half, 128, 64>>(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers);
template __global__ void flash_combine_kernel<half, Traits<half, 128, 64>>(Flash_decoder_input &input, Flash_decoder_params &params, Flash_decoder_buffers &buffers);
template void run_flash_decoder<half>(Flash_decoder_input &input, Flash_decoder_params &params, cudaStream_t);

// TODO: 目前仅针对3090单卡和llama 7B设置分割数，后续需要智能分割