import torch
import math
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from opt_einsum import contract as einsum
from einops import rearrange
from torchvision import transforms
from modules.differential_attention import DiffAttention, RMSNorm


class ConvEncoder(nn.Module):
    def __init__(self, d, resize_shape=48, in_channels=6):
        super().__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape), 
            #* Other transforms
        ])
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, 
                               padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,
                               padding='same')
        self.global_avg_pool = nn.AdaptiveAvgPool2d(16)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16 * 256, d)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_avg_pool(x)
        x = self.fc(self.flatten(x))
        return x


class Image2Seq(nn.Module):
    def __init__(self, d, resnet_name='resnet18',
                 in_channels=6, conv_resize_shape=48):
        super().__init__()
        self.resnets = {
            'resnet18': (resnet18, 512),
            'resnet34': (resnet34, 512),
            'resnet50': (resnet50, 2048),
            'resnet101': (resnet101, 2048),
        }
        if resnet_name is not None:
            assert resnet_name in self.resnets.keys(),\
                f'Unknown ResNet version: {resnet_name}. Set resnet_name to one of {list(self.resnets.keys())} or None.'
            resnet = self.resnets[resnet_name][0](weights=None)
            d1 = self.resnets[resnet_name][1]
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1),
                nn.Sequential(*list(resnet.children())[:-1]),
                nn.Flatten(),
                nn.Linear(d1, d)
            )
        else:
            self.encoder = ConvEncoder(d, conv_resize_shape, in_channels)
        
    def forward(self, x):
        # x: (B, S, 6, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        x = self.encoder(x)
        x = x.view(B, S, -1)
        return x
    
    
class SinusoidalTimePosEmbedding(nn.Module):
    def __init__(self, hidden_size, d, frequency_embedding_size=256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, d, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        # t: (B, S)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, :, None].float() * freqs[None, None]  # (B, S, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1], device=t.device)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_pos = self.mlp(t_freq)
        return t_pos


class OuterProduct(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.emb_a = nn.Linear(dim, hidden_dim, bias=False)  # left
        self.emb_b = nn.Linear(dim, hidden_dim, bias=False)  # right
        self.to_out = nn.Linear(hidden_dim * hidden_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x (B, L, dim)
        Return:
            z (B, L, L, out_dim)
        """
        x = self.norm(x)
        a = self.emb_a(x)
        b = self.emb_b(x)
        o = einsum('bic,bjd->bijcd', a, b)  # (B, L, L, d, d), outer product
        o = rearrange(o, 'b i j c d -> b i j (c d)')
        z = self.to_out(o)
        return z


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim, bias=False)
        self.proj_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x_proj = self.proj_2(x)
        swish_x = x_proj * F.sigmoid(x_proj)
        x = self.proj_1(x) * swish_x
        return x
    
    
class Embedding(nn.Module):
    def __init__(self, dim, d_pair,
                 hidden_size=32, frequency_embedding_size=256, p_drop=0.1) -> None:
        super().__init__()
        self.emb_seq = nn.Linear(dim, dim)
        self.emb_pair_right = nn.Embedding(1, d_pair)  # d_pair should be 2 * num_heads
        self.emb_pair_left = nn.Embedding(1, d_pair)
        self.drop = nn.Dropout(p_drop)
        if frequency_embedding_size > 0:
            self.pos = SinusoidalTimePosEmbedding(hidden_size=hidden_size, d=dim,
                                                  frequency_embedding_size=frequency_embedding_size)
        self.frequency_embedding_size = frequency_embedding_size
        
    def forward(self, x, t):
        # * seq embedding
        x = self.emb_seq(x)
        if self.frequency_embedding_size > 0:
            x = x + self.pos(t)
            # * pair embedding
            t = t[:, :, None]
            left = (t @ self.emb_pair_left.weight)[:, None]  # (B, 1, L, d_pair)
            right = (t @ self.emb_pair_right.weight)[:, :, None]  # (B, L, 1, d_pair)
            t = left + right
            return self.drop(x), self.drop(t)
        else:
            return self.drop(x), None
    

class DyBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, dropout=0., 
                 apply_attn=True, outer=True) -> None:
        super().__init__()
        self.apply_attn = apply_attn
        self.outer = outer
        if self.apply_attn:
            self.ln_t = RMSNorm(dim, eps=1e-5, elementwise_affine=False)
            self.ln_x = RMSNorm(dim, eps=1e-5, elementwise_affine=False)
            self.ln_y = RMSNorm(dim, eps=1e-5, elementwise_affine=False)
            self.diff_attn = DiffAttention(dim=dim, depth=depth, num_heads=num_heads, dropout=dropout)
            self.swi_glu = SwiGLU(dim)
            self.outer_product = OuterProduct(dim=dim, hidden_dim=dim, out_dim=2 * num_heads)
        else:
            self.proj = nn.Liner(dim, dim)

    def diff_attn_forward(self, x, t=None):
        if t is not None:
            t = self.ln_t(t)
        y = x + self.diff_attn(self.ln_x(x), t)
        x = y + self.swi_glu(self.ln_y(y))
        if self.outer and t is not None:
            t = t + self.outer_product(x)
        return x, t

    def linear_forward(self, x, t=None):
        return x + self.proj(x), t
    
    def forward(self, x, t=None):
        if self.apply_attn:
            x, t = self.diff_attn_forward(x, t)
        else:
            x, t = self.linear_forward(x, t)
        return x, t

        
class DynamicFormer(nn.Module):
    def __init__(self, dim, num_heads, depth, out_dim=1,
                 resnet_name='resnet18',
                 in_channels=6, conv_resize_shape=48,
                 frequency_embedding_size=256,
                 apply_attn=True, outer=True,
                 dropout=0.0, device='cuda') -> None:
        super().__init__()
        self.device = torch.device(device)

        self.img2seq = Image2Seq(d=dim, resnet_name=resnet_name,
                                 in_channels=in_channels, 
                                 conv_resize_shape=conv_resize_shape).to(self.device)
        
        self.embedding = Embedding(dim=dim, d_pair=2 * num_heads, hidden_size=dim,
                                   frequency_embedding_size=frequency_embedding_size, 
                                   p_drop=dropout).to(self.device)

        self.dyblocks = nn.ModuleList()
        for depth_ in range(depth):
            self.dyblocks.append(DyBlock(dim, depth_, num_heads, dropout=dropout,
                                         apply_attn=apply_attn, outer=outer).to(self.device))

        assert out_dim >= 1
        if out_dim == 1:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, out_dim),
                nn.Sigmoid()
            ).to(self.device)
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, out_dim)
            ).to(self.device)

    def forward(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)

        x = self.img2seq(x)
        x, t = self.embedding(x, t)

        for dyblock in self.dyblocks:
            x, t = dyblock(x, t)

        x = torch.mean(x, dim=-2)
        out = self.mlp_head(x)

        return out


def build_model(args):
    model = DynamicFormer(dim=args.dim, 
                          num_heads=args.num_heads, 
                          depth=args.depth, 
                          out_dim=args.out_dim,
                          resnet_name=args.resnet_name, 
                          in_channels=args.in_channels,
                          conv_resize_shape=args.resize_shape,
                          frequency_embedding_size=args.frequency_embedding_size,
                          apply_attn=args.apply_attn,
                          outer=args.outer,
                          dropout=args.dropout,
                          device=args.device)
    return model