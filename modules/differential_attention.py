import torch
import math
from torch import nn
import torch.nn.functional as F


"""
Differential attention code from:
https://github.com/microsoft/unilm/tree/master/Diff-Transformer
"""   
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class DiffAttention(nn.Module):
    def __init__(self, dim, depth, num_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x, attn_mask=None):
        # x: (B, S, dim)
        # attn_mask: t, (B, S, S, 2 * H)
        B, S, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, S, 2 * self.num_heads, self.head_dim)
        k = k.view(B, S, 2 * self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, 2 * self.head_dim)

        q = q.transpose(1, 2) * self.scaling  # (B, 2 * H, S, head_dim)
        k = k.transpose(1, 2)  # (B, 2 * H, S, head_dim)
        v = v.transpose(1, 2)  # (B, H, S, 2 * head_dim)

        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (B, 2 * H, S, S)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([S, S])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1,
            )
        else:
            attn_mask = attn_mask.permute(0, -1, 1, 2)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(B, self.num_heads, 2, S, S)  # (B, H, 2, S, S)
        self.A1 = attn_weights[:, :, 0].cpu().detach()
        self.A2 = (lambda_full * attn_weights[:, :, 1]).cpu().detach()
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # (B, H, S, S)
        
        attn = torch.matmul(attn_weights, v)  # (B, H, S, 2 * head_dim)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, S, self.num_heads * 2 * self.head_dim)
        return self.out_proj(attn)