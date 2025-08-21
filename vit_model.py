from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops.layers.torch import Rearrange
from einops import rearrange, repeat


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)

        mean = torch.mean(x, dim = 1, keepdim = True)

        d = (x - mean) / (var + self.eps).sqrt()

        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.2):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)# change to dots
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 31):
        super().__init__()
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * 13 * channels


        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (d d1) (h h1) (w w1) -> b (d h w) (d1 h1 w1 c)', d1=patch_size[0], h1=patch_size[1],
                      w1=patch_size[2]),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((3, -3, 0, 0, 0, 0), (-3, 3, 0, 0, 0, 0), (0, 0, 3, -3, 0, 0),
                  (0, 0, -3, 3, 0, 0), (0, 0, 0, 0, 3, -3), (0, 0, 0, 0, -3, 3),
                  (3, -3, 3, -3, 0, 0), (-3, 3, 0, 0, -3, 3), (3, -3, 0, 0, 3, -3),
                  (-3, 3, -3, 3, 0, 0), (0, 0, -3, 3, -3, 3), (0, 0, 3, -3, 3, -3))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))

        x_with_shifts = torch.cat((x, *shifted_x), dim=1)

        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, grid_size, patch_size,  dim, depth, heads, mlp_dim, pool = 'cls', channels = 31, dim_head = 64, dropout = 0.25, emb_dropout = 0.25):
        super().__init__()

        grid_depth, grid_height, grid_width = grid_size
        patch_depth, patch_height, patch_width = patch_size

        print(grid_depth, grid_height, grid_width)
        print(patch_depth, patch_height, patch_width)

        print(grid_height % patch_height == 0 and grid_width % patch_width == 0 and grid_depth % patch_depth == 0)
        # assert grid_height % patch_height == 0 and grid_width % patch_width == 0 and grid_depth % patch_depth, 'Image dimensions must be divisible by the patch size.'

        num_patches = (grid_depth // patch_depth) * (grid_height // patch_height) * (grid_width // patch_width)
        self.p_dim = grid_depth // patch_depth
        patch_dim = channels * patch_depth * patch_height * patch_width
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, grid):
        grid = rearrange(grid, "b d h w c -> b c d h w")

        x = self.to_patch_embedding(grid)

        b, n, _ = x.shape



        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
