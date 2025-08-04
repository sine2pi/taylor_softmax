
class attention(nn.Module):
    def __init__(self, dims: int, head: int, dropout_rate: float = 0.1):
        super().__init__()

        self.head = head
        self.dims = dims
        self.que = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)
        self.ln = nn.LayerNorm(dims) 
        self.rope = rotary(dims, head) 

    def taylor_softmax(self, x, order=2):
        taylor_approx = 1.0
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            taylor_approx += x**i / factorial_i
        return taylor_approx / torch.sum(taylor_approx, dim=-1, keepdim=True)

    def taylor_softmax_2nd_order(x):
        exp_approx = 1 + x + (x**2) / 2
        return exp_approx / torch.sum(exp_approx, dim=-1, keepdim=True)

    def forward(self, x, xa = None, mask = None):

        q = self.que(self.ln(x))
        k, v = self.kv(self.ln(x if xa is None else xa)).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        q = self.rope(q)
        k = self.rope(k)

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale 
 #       qk = torch.nn.functional.softmax(qk, dim=-1) For comparison
        qk = self.taylor_softmax(qk, order=2)        

        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out


# import torch
# import torch.nn.functional as F

# batch_size = 2
# num_heads = 4
# sequence_length_k = 10
# sequence_length_q = 5
# embedding_dim = 64

# q = torch.randn(batch_size, num_heads, sequence_length_k, embedding_dim)
# k = torch.randn(batch_size, num_heads, sequence_length_q, embedding_dim)
# v = torch.randn(batch_size, num_heads, sequence_length_q, embedding_dim)

# scale = 1.0 / (embedding_dim**0.5)

# def taylor_softmax(x, order=2):
#     taylor_approx = 1.0
#     for i in range(1, order + 1):
#         factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
#         taylor_approx += x**i / factorial_i
#     return taylor_approx / torch.sum(taylor_approx, dim=-1, keepdim=True)

# qk_original = torch.einsum('b h k d, b h q d -> b h k q', q, k) * scale
# qk_original_softmax = F.softmax(qk_original, dim=-1)

# qk_taylor = torch.einsum('b h k d, b h q d -> b h k q', q, k) * scale
# qk_taylor_softmax = taylor_softmax(qk_taylor, order=2)

# frobenius_norm_diff = torch.linalg.matrix_norm(qk_original_softmax - qk_taylor_softmax, ord='fro')

# frobenius_norm_original_qk = torch.linalg.matrix_norm(qk_original_softmax, ord='fro')

# frobenius_norm_taylor_qk = torch.linalg.matrix_norm(qk_taylor_softmax, ord='fro')

# wv_original = torch.einsum('b h k q, b h q d -> b h k d', qk_original_softmax, v)
# wv_taylor = torch.einsum('b h k q, b h q d -> b h k d', qk_taylor_softmax, v)

# print("Frobenius Norm of Difference (Original - Taylor Softmax):", frobenius_norm_diff)
# print("Frobenius Norm of Original Softmax Attention Weights:", frobenius_norm_original_qk)
# print("Frobenius Norm of Taylor Softmax Attention Weights (order 2):", frobenius_norm_taylor_qk)
# print("\nOriginal Softmax Output (wv):", wv_original)
# print("Taylor Softmax Output (wv, order 2):", wv_taylor)
