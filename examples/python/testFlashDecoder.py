import torch
import math
import star_flash_attn
import torch.nn.functional as F
import torch.nn as nn

class LlamaRotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len, device):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            print(freqs)
            emb = freqs.repeat_interleave(2, dim=-1).to(device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached
        
def rotate_half(x):
    # x_even取偶数索引元素（即原来的x0, x2, x4,...）
    x_even = x[..., 0::2]  # 从索引0开始，步长为2取元素（偶数索引）
    # x_odd取奇数索引元素（即原来的x1, x3, x5,...）
    x_odd = x[..., 1::2]  # 从索引1开始，步长为2取元素（奇数索引）

    # 交替重新组合x_odd和x_even，以实现x1,x0,x3,x2,...,xN,xN-1的效果
    # 我们创建一个新的空tensor，其总元素数等于x_odd和x_even总和的两倍
    reordered = torch.empty_like(x)

    # 将x_odd放在偶数位置（即x1, x3, x5,...）
    reordered[..., 0::2] = -x_odd
    # 将x_even放在奇数位置（即x0, x2, x4,...）
    reordered[..., 1::2] = x_even

    return reordered

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    # 保存原始数据类型
    original_dtype = q.dtype
    
    # 计算旋转位置编码后的q和k
    # 注意：这里假设rotate_half已经定义在其他地方，且兼容Tensor操作
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    # 将结果转换回原始数据类型
    q_rot = q_rot.to(original_dtype)
    k_rot = k_rot.to(original_dtype)
    return q_rot, k_rot
    

class LlamaAttention(nn.Module):
    
    def __init__(self, head_dim):
        super().__init__()
        self.rotary_emb = LlamaRotaryEmbedding(dim=head_dim)

    def forward(
        self, qkv, key_cache, value_cache, current_seq_len, layer_idx
    ) -> torch.tensor:
        _, _, head_dim = qkv.size(0), qkv.size(2), qkv.size(3)
        queries, keys, values = torch.chunk(qkv, 3, dim=1)

        # 在这里应用旋转位置编码
        cos, sin = self.rotary_emb(current_seq_len, queries.device)
        print(cos[-1:,:,:,:1], sin[-1:,:,:,:1])
        queries, keys = apply_rotary_pos_emb(queries, keys, cos[-1:,:,:,:], sin[-1:,:,:,:])
        print(queries[0,0,0,:8])
        
        keys = torch.cat([key_cache[:, layer_idx, :current_seq_len - 1, :, :], keys], dim=1)
        values = torch.cat([value_cache[:, layer_idx, :current_seq_len - 1, :, :], values], dim=1)
        # print(queries.dtype, keys.dtype, values.dtype)
        # print(queries.shape, keys.shape, values.shape)
        # Calculate scores
        scores = torch.matmul(queries.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3)) / math.sqrt(head_dim)
        # print(scores)
        print("scores value: ", scores[0,0,0,-8:], "score shape: ", scores.shape)
        # Apply softmax to get probabilities
        probs = F.softmax(scores, dim=-1)
        print("prob value:", probs[0,0,0,:])
        print("probs shape: ", probs.shape, "keys shape: ", keys.shape, "valuesT shape: ", values.transpose(1,2).shape)
        # Weighted sum of values based on the probabilities
        attn_output = torch.matmul(probs, values.transpose(1,2))
        print("attn_output value: ", attn_output)
        return attn_output

def main():
    batch_size = 1
    num_heads = 32
    head_dim = 128
    seq_len_value = 512
    max_seq_len = 1024
    num_splits = 4
    kNThreads = 32
    num_layer = 4
    idx_layer = 0

    # 设备配置
    device = torch.device("cuda")

    # 分配并初始化输入和参数结构
    qkv = torch.ones((batch_size, 3, num_heads, head_dim), dtype=torch.float16, device=device)
    q_bias = torch.zeros((num_heads, head_dim), dtype=torch.float16, device=device)
    k_bias = torch.zeros((num_heads, head_dim), dtype=torch.float16, device=device)
    v_bias = torch.zeros((num_heads, head_dim), dtype=torch.float16, device=device)
    k_cache_table = torch.ones((batch_size, num_layer, max_seq_len, num_heads, head_dim), dtype=torch.float16, device=device)
    v_cache_table = torch.ones((batch_size, num_layer, max_seq_len, num_heads, head_dim), dtype=torch.float16, device=device)
    seq_len = torch.full((batch_size,), seq_len_value, dtype=torch.int32, device=device)
    o = torch.zeros((batch_size, num_heads, head_dim), dtype=torch.float16, device=device)

    print("[INFO] 测试开始")

    # 运行解码器
    
    # o = star_flash_attn.mha_fwd_cuda(qkv, q_bias, k_bias, v_bias, k_cache_table, v_cache_table, seq_len, o,
    #                                  batch_size, max_seq_len, num_heads, head_dim, head_dim,
    #                                  max_seq_len, num_layer, idx_layer)
    
    attn = LlamaAttention(head_dim)
    o = attn.forward(qkv, k_cache_table, v_cache_table, seq_len_value, idx_layer)

    print("[INFO] 测试结束")

    # 将输出拷贝至主机内存，并打印第一个head和最后一个head
    o_host = o.cpu()
    print("First head:")
    print(o_host[0, 0, :])
    print("Last head:")
    print(o_host[0, -1, :])

if __name__ == "__main__":
    main()
