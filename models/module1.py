import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x



class GSSM(nn.Module):
    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(nn.Conv1d(self.dim, hidden, 1, 1, 0))

        self.CPE = nn.Sequential(nn.Conv1d(hidden, hidden, 3, 1, 1, groups=hidden)) # 卷积位置编码（Convolutional Positional Encoding, CPE）

        self.route = nn.Sequential(nn.Linear(self.dim, self.dim // 3),
                                   nn.GELU(),
                                   nn.Linear(self.dim // 3, self.num_tokens),
                                   nn.LogSoftmax(dim=-1))

    def forward(self, x, x_size, global_weight):
        B, n, C = x.shape
        L = x_size

        #### SAR ####
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, L).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x)) # 空间注意力门控：根据每个像素周围的局部上下文信息，动态地生成一个 0~1 之间的软掩码（Soft Mask），然后用这个掩码去重新加权（放大或抑制）原始特征。
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
        #### SAR ####
        y = self.selectiveScan(semantic_x, global_weight)
        y = self.out_proj(self.out_norm(y))

        #### Invert ####
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    ## 初始化dt的权重和偏置，将其值限定在一定范围内
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    ## 该函数初始化SSM中的A，A是【1, d_state】的序列，函数返回log(A),并将其作为可训练的参数
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous() # repeat: 用于对张量的某一个维度进行复制
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)  #nn.Parameter 是 PyTorch 中的一个类，用于将不可训练的 Tensor 转换为可训练的参数，并将其绑定到模型中。
        A_log._no_weight_decay = True  # 在常见的 AdamW 优化器中，默认会对所有参数应用权重衰减，这个标记专门告诉优化器：请跳过这个参数，不要对它施加权重衰减，保证它保持固有的多尺度衰减特性。
        return A_log

    ## 该函数初始化SSM中的D，D的元素全为1，函数返回D,并将其作为可训练的参数
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, global_weight):

        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        ######## Global Enhancement ########
        Cs = Cs.float().view(B, K, -1, L) + global_weight
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # A_ba = -exp(A_log)有讲究，在状态空间模型（SSM）的连续时间表示中，状态矩阵 A 的特征值必须具有负实部，以保证系统是渐近稳定的，exp 始终为正，加负号后 A 始终为负，无论 A_log 被 SGD 更新成多大，A 永远小于 0，从而100% 确保了系统的稳定性。
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, global_weight,  **kwargs):
        y = self.forward_core(x, global_weight)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class FeedForward(nn.Module):
    def __init__(self, dim, expand=2., bias=True):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * expand)
        self.project_in = nn.Conv1d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.dwconv2 = nn.Conv1d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                 groups=hidden_features, bias=bias)

        self.dwconv3 = nn.Conv1d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)

        self.dwconv4 = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1, bias=bias)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x_in, x_p):

        x = self.project_in(x_in)
        x = self.dwconv(x)
        x_p = self.dwconv2(x_p)
        T, A = self.dwconv3(x_p).chunk(2, dim=1)
        T = self.sigmoid(T)
        x = x * T + A * (1 - T)
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)

        return x



class PMGMamba_Block(nn.Module):
    def __init__(self, dim, d_state, inner_rank, num_tokens, mlp_ratio, norm_layer=nn.LayerNorm):
        super(PMGMamba_Block, self).__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.norm3 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.gssm = GSSM(self.dim, d_state, num_tokens=num_tokens, inner_rank=inner_rank, mlp_ratio=mlp_ratio)

        sample_rate = 2
        self.sampler = nn.MaxPool1d(kernel_size=sample_rate, stride=sample_rate)
        self.kernel_size = sample_rate
        self.patch_size = sample_rate
        self.LocalProp = nn.ConvTranspose1d(dim, dim, kernel_size=self.kernel_size, padding=(self.kernel_size // sample_rate - 1),
                                            stride=sample_rate, groups=dim, bias=False)

        self.global_proj = nn.Linear(dim, d_state)
        self.gate = nn.Linear(dim, d_state)

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.norm4 = norm_layer(dim)
        self.ffn = FeedForward(dim)
        self.conv1d = nn.Conv1d(int(dim * mlp_ratio), int(dim * mlp_ratio), kernel_size=3, stride=1, padding=1,
                                groups=dim, bias=False)


    def forward(self, x, x_p):

        ####GlobalGate####
        b, c, l = x.shape # 1 32 256 256
        x_ = self.pool(x).view(b, c, -1).permute(0, 2, 1).contiguous() # 1 128*128 32 #这一行代码把图像特征图 [B, C, H, W] 降采样后，拉直成 [B, N, C] 的 Token 序列，并确保内存连续，为后续的 Transformer 或动态路由排序做好格式准备。
        global_x = self.global_proj(x_.mean(dim=1, keepdim=True)).permute(0, 2, 1).contiguous() # 1 8 1 对x_第二维求均值，然后第三维由dim变换为d_state，再交换第二维和第三维
        global_gate = torch.sigmoid(self.gate(x_)).permute(0, 2, 1).contiguous() # 1 8 128*128 torch.sigmoid(...)作用：把投影后的数值压缩到 (0, 1) 区间。

        #global_x[c]：整个图像在通道 c 上的平均水平（例如“红色的平均强度”）。
        # global_gate[j]：第 j 个 Token 位置的重要性权重（例如“这是前景物体”的置信度 0.9）。
        # 结果解读：
        # 如果某个 Token 很重要（gate 接近 1），那么该位置就会继承全局平均特征的全部强度。
        # 如果某个 Token 是背景噪声（gate 接近 0），那么该位置的全局特征就被压制到接近 0。
        global_weight = (global_gate * global_x).unsqueeze(1) # *会将不匹配的维度相乘，匹配不动；unsqueeze(1)扩展第二维；这

        ####SAGMamba####
        xs = self.sampler(x)
        b, c, ls = xs.shape
        xs_size = (ls)
        xs = xs.view(b, c, -1).permute(0, 2, 1).contiguous()
        x_aca = self.gssm(self.norm3(xs), xs_size, global_weight) + xs  # 核心步骤，mamba处理
        x_aca = self.LocalProp(x_aca.permute(0, 2, 1).view(b, c, ls).contiguous())  # 对经过Mamba后的结果逆卷积
        x_aca = x_aca.view(b, c, -1).permute(0, 2, 1).contiguous()
        x = x_aca + self.scale2 + x.view(b, c, -1).permute(0, 2, 1).contiguous()

        ####PMGFFN####
        x_p = self.conv1d(x_p)
        x = self.ffn(self.norm4(x).permute(0, 2, 1).view(b, c, l).contiguous(), x_p) + x.permute(0, 2, 1).view(b, c, l).contiguous()

        return x



