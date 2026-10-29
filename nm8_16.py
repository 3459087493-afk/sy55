import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model

__all__ = ['Spikingformer', 'RIASparseConv', 'apply_ria_pruning', 'ria_sparsity_report']


# ============================ RIA 剪枝模块（N:M 半结构化稀疏） ============================
# 参考论文: Plug-and-Play (ICLR 2024), Zhang et al.
#   RI_ij  = |W_ij| / sum_i'|W_i'j| + |W_ij| / sum_j'|W_ij'|
#   RIA_ij = RI_ij * (||X_j||_2) ** a          (a 默认 0.5)
#   N:M    = 每一行(输出通道)沿输入通道每 M 个连续元素中, 仅保留 RIA 分数最高的 N 个
# 实现方式: 前向始终使用 masked weight; 梯度天然被 mask 门控(等价 STE);
#           每 update_interval 步用最新 weight + 激活 EMA 重算一次 mask。

class RIASparseConv(nn.Module):
    """RIA 指标驱动的 N:M 稀疏卷积包装器(兼容 Conv1d / Conv2d)。

    怎么读: 它包住原来的 conv(如 q_conv/mlp1_conv), 对外仍是普通 conv;
            内部维护一个与 weight 同形的 0/1 mask, forward 时计算 conv(x, weight*mask)。
    """

    def __init__(self, conv, n=4, m=8, alpha=0.5, update_interval=100,
                 warmup_steps=500, act_ema=0.9, name=''):
        # 是什么: 用原 conv + 一组 RIA 超参数构造稀疏包装层
        # 为什么: 论文式 plug-and-play, 不改变原 conv 结构, 只门控其权重
        # 怎么读: n/m 为稀疏比(4:8 即 50%); alpha 为激活幂因子; warmup 步内保持稠密
        super().__init__()
        assert n < m, f'N:M 稀疏要求 n < m, 收到 n={n}, m={m}'
        self.conv = conv
        self.n, self.m = int(n), int(m)
        self.alpha = float(alpha)
        self.update_interval = int(update_interval)
        self.warmup_steps = int(warmup_steps)
        self.act_ema = float(act_ema)
        self.name = name or conv.__class__.__name__

        # 是什么: 输出/输入通道数(1x1 卷积的 kernel 维均为 1, 不影响通道语义)
        # 为什么: RIA 分数在 (out, in) 矩阵上计算, mask 再广播回 kernel 维
        out_c, in_c = conv.weight.shape[0], conv.weight.shape[1]
        self.out_c, self.in_c = out_c, in_c
        self.register_buffer('mask', torch.ones_like(conv.weight))   # 初始全 1 = 稠密
        self.register_buffer('act_sq_ema', torch.zeros(in_c))        # 每输入通道激活平方的 EMA
        self.register_buffer('steps', torch.zeros((), dtype=torch.long))
        self.register_buffer('num_mask_updates', torch.zeros((), dtype=torch.long))

        # 是什么: 前向预钩子, 在 conv 计算前抓取输入, 统计每输入通道的激活能量
        # 为什么: RIA 需要 ||X_j||_2, 训练中逐 batch EMA 估计, 与论文校准集思想一致
        conv.register_forward_pre_hook(self._collect_activation)

    @torch.no_grad()
    def _collect_activation(self, module, inputs):
        # 输入 x: (B, in_c, L) 或 (B, in_c, H, W), 对 in_c 之外的维求均值
        x = inputs[0].detach()
        reduce_dims = [0] + list(range(2, x.dim()))
        per_ch_sq = (x.float() ** 2).mean(dim=reduce_dims)  # (in_c,)
        if self.act_sq_ema.sum() == 0 and self.steps.item() == 0:
            self.act_sq_ema.copy_(per_ch_sq)                # 首个 batch 直接初始化
        else:
            self.act_sq_ema.mul_(self.act_ema).add_(per_ch_sq, alpha=1.0 - self.act_ema)

    def _weight_matrix(self):
        # 是什么: 把 (out,in,1[,1]) 的卷积权重压成 (out,in) 矩阵
        # 为什么: RIA 公式定义在矩阵上; 1x1 卷积压缩后无信息损失
        return self.conv.weight.detach().abs().reshape(self.out_c, self.in_c)

    @torch.no_grad()
    def _update_mask(self):
        # 是什么: 按当前权重与激活统计重算 N:M mask(论文公式 2/3 + 每块 top-N)
        # 为什么: 训练中权重持续变化, 周期性重选重要连接, 避免早期选择固化
        w = self._weight_matrix()                                   # |W| (out,in)
        col_sum = w.sum(dim=0).clamp_min(1e-12)                     # sum_i'|W_i'j| 每输入通道
        row_sum = w.sum(dim=1).clamp_min(1e-12)                     # sum_j'|W_ij'| 每输出通道
        act_norm = (self.act_sq_ema.clamp_min(0) + 1e-12).sqrt()    # ||X_j||_2
        # RI = 输入通道归一 + 输出通道归一; RIA 再乘激活项 ** alpha
        ri = w / col_sum.unsqueeze(0) + w / row_sum.unsqueeze(1)
        ria = ri * (act_norm.unsqueeze(0) ** self.alpha)
        self.mask = self._nm_topn(ria).reshape_as(self.conv.weight)
        self.num_mask_updates += 1

    def _nm_topn(self, score):
        # 是什么: 对 (out,in) 分数矩阵做 N:M 约束——每行每 M 个连续输入通道保留 top-N
        # 为什么: 硬件稀疏 tensor core 要求; top-N 最大化保留的重要性(论文式 5)
        # 怎么读: 不足 M 的尾部通道单独成块, 仍按 top-N 规则处理
        out_c, in_c = score.shape
        pad = (-in_c) % self.m
        if pad:
            score = torch.cat([score, torch.full((out_c, pad), -1.0, device=score.device, dtype=score.dtype)], dim=1)
        blocks = score.reshape(out_c, -1, self.m)                   # (out, K, M)
        # 每块内分数排名前 n 的位置置 1(填充的尾部块分数为 -1, 只会被多余名额误选, 随后切掉)
        kth = blocks.topk(self.n, dim=-1).indices                   # (out, K, n)
        keep = torch.zeros_like(blocks).scatter_(-1, kth, 1.0)
        return keep.reshape(out_c, -1)[:, :in_c]

    def forward(self, x):
        # 训练态下推进步数, 到间隔且过 warmup 后重算 mask; 推理态直接用当前 mask
        if self.training:
            self.steps += 1
            if self.steps.item() >= self.warmup_steps and \
                    self.steps.item() % self.update_interval == 0:
                self._update_mask()
        # 是什么: 用 masked weight 做卷积(权重乘法在 autocast 外保持 fp32)
        # 为什么: mask 为常量 buffer, 梯度自动只回传到保留位置, 等价 STE
        w = self.conv.weight * self.mask
        return self.conv._conv_forward(x, w, self.conv.bias)

    def extra_repr(self):
        # 打印层信息时显示稀疏比与已更新次数, 便于训练日志核对
        return (f'{self.name}, {self.out_c}x{self.in_c}, {self.n}:{self.m}, '
                f'updates={int(self.num_mask_updates.item())}, '
                f'sparsity={1.0 - self.mask.mean().item():.3f}')


def apply_ria_pruning(model, n=8, m=16, alpha=0.5, update_interval=100, warmup_steps=500,
                      act_ema=0.9, verbose=True):
    """对模型的 Attention Q/K/V 与 MLP 权重施加 RIA N:M 剪枝(就地替换 conv)。

    是什么: 遍历所有 SpikingSelfAttention / MLP, 把 q/k/v_conv 与 mlp1/mlp2_conv
            包装成 RIASparseConv; 其余层(含 proj_conv)保持稠密。
    为什么: 按任务要求只剪 Attention QKV 权重和 MLP 权重。
    怎么读: 返回被包装的层数量; 重复调用安全(已是 RIASparseConv 时跳过)。
    """
    replaced = []
    for module in model.modules():
        if isinstance(module, RIASparseConv):
            continue  # 已包装过, 避免双重嵌套
        if isinstance(module, SpikingSelfAttention):
            for attr in ('q_conv', 'k_conv', 'v_conv'):
                conv = getattr(module, attr)
                setattr(module, attr, RIASparseConv(
                    conv, n=n, m=m, alpha=alpha, update_interval=update_interval,
                    warmup_steps=warmup_steps, act_ema=act_ema, name=attr))
                replaced.append(f'attn.{attr}')
        elif isinstance(module, MLP):
            for attr in ('mlp1_conv', 'mlp2_conv'):
                conv = getattr(module, attr)
                setattr(module, attr, RIASparseConv(
                    conv, n=n, m=m, alpha=alpha, update_interval=update_interval,
                    warmup_steps=warmup_steps, act_ema=act_ema, name=attr))
                replaced.append(f'mlp.{attr}')
    if verbose:
        print(f'[RIA] N:M = {n}:{m}  alpha={alpha}  update_interval={update_interval}  '
              f'warmup_steps={warmup_steps}')
        print(f'[RIA] 已包装 {len(replaced)} 个卷积层: {replaced[0]} ... {replaced[-1]}')
    return len(replaced)


def ria_sparsity_report(model):
    """统计并打印模型中所有 RIASparseConv 的实际稀疏比例, 用于核对 N:M 约束。"""
    total_elems, total_zeros = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, RIASparseConv):
            zeros = int((module.mask == 0).sum().item())
            elems = module.mask.numel()
            total_elems += elems
            total_zeros += zeros
            print(f'  {name:55s} sparsity={zeros / elems:.4f}')
    overall = total_zeros / max(total_elems, 1)
    print(f'[RIA] 剪枝层整体稀疏度: {overall:.4f}')
    return overall


class TernarySpikeFunction(torch.autograd.Function):
    """Ternary spike: membrane values become -1, 0, or +1."""

    @staticmethod
    def forward(ctx, x, threshold, surrogate_alpha):
        threshold = torch.as_tensor(threshold, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, threshold)
        ctx.surrogate_alpha = float(surrogate_alpha)

        out = torch.zeros_like(x)
        out = torch.where(x >= threshold, torch.ones_like(x), out)
        out = torch.where(x <= -threshold, -torch.ones_like(x), out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        width = 1.0 / max(ctx.surrogate_alpha, 1e-6)

        grad_pos = (1.0 - (x - threshold).abs() / width).clamp(min=0.0)
        grad_neg = (1.0 - (x + threshold).abs() / width).clamp(min=0.0)
        return grad_output * (grad_pos + grad_neg), None, None


class MultiStepTernaryLIFNode(nn.Module):
    """Multi-step LIF node with ternary {-1, 0, +1} spikes."""

    def __init__(self, tau=2.0, v_threshold=1.0, detach_reset=True, surrogate_alpha=2.0):
        super().__init__()
        if tau <= 1.0:
            raise ValueError('tau must be greater than 1.0 for LIF dynamics')
        self.tau = tau
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_alpha = surrogate_alpha
        self.v = None

    def forward(self, x):
        if x.dim() < 2:
            raise ValueError('MultiStepTernaryLIFNode expects input with time dimension first')

        if self.v is None or self.v.shape != x[0].shape or self.v.device != x.device:
            self.v = torch.zeros_like(x[0])

        spikes = []
        for t in range(x.shape[0]):
            self.v = self.v + (x[t] - self.v) / self.tau
            spike = TernarySpikeFunction.apply(self.v, self.v_threshold, self.surrogate_alpha)
            reset_spike = spike.detach() if self.detach_reset else spike
            self.v = self.v - reset_spike * self.v_threshold
            spikes.append(spike)

        return torch.stack(spikes, dim=0)

    def reset(self):
        self.v = None


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 ternary_threshold=1.0, ternary_surrogate_alpha=2.0, attn_scale=0.125,
                 ternary_v_mode='ternary', score_norm='none', score_scale=1.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert score_norm in ('none', 'softmax')

        self.dim = dim
        self.num_heads = num_heads
        self.score_norm = score_norm
        self.score_scale = score_scale
        self.ternary_threshold = ternary_threshold

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)

        self.q_lif = MultiStepTernaryLIFNode(
            tau=2.0, v_threshold=ternary_threshold, detach_reset=True,
            surrogate_alpha=ternary_surrogate_alpha)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)

        self.k_lif = MultiStepTernaryLIFNode(
            tau=2.0, v_threshold=ternary_threshold, detach_reset=True,
            surrogate_alpha=ternary_surrogate_alpha)
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepTernaryLIFNode(
            tau=2.0, v_threshold=ternary_threshold, detach_reset=True,
            surrogate_alpha=ternary_surrogate_alpha)

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')
        self.attn_scale = attn_scale
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = q @ k.transpose(-2, -1)
        if self.score_norm == 'softmax':
            attn = torch.softmax(attn * self.score_scale, dim=-1)
        x = (attn @ v) * self.attn_scale

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, ternary_threshold=1.0,
                 ternary_surrogate_alpha=2.0, attn_scale=0.125, ternary_v_mode='ternary',
                 score_norm='none', score_scale=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                                           ternary_threshold=ternary_threshold,
                                           ternary_surrogate_alpha=ternary_surrogate_alpha,
                                           attn_scale=attn_scale, ternary_v_mode=ternary_v_mode,
                                           score_norm=score_norm, score_scale=score_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)

        self.block1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.block2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)

        self.block3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 1)

        self.block4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)

        x = self.block1_lif(x).flatten(0, 1)
        x = self.block1_conv(x)
        x = self.block1_bn(x).reshape(T, B, -1, H, W)

        x = self.block2_lif(x).flatten(0, 1)
        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)

        x = self.block3_lif(x).flatten(0, 1)
        x = self.block3_mp(x)
        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(T, B, -1, int(H / 2), int(W / 2))

        x = self.block4_lif(x).flatten(0, 1)
        x = self.block4_mp(x)
        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ternary_threshold=1.0, ternary_surrogate_alpha=2.0, attn_scale=0.125,
                 ternary_v_mode='ternary', score_norm='none', score_scale=1.0,
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, ternary_threshold=ternary_threshold,
            ternary_surrogate_alpha=ternary_surrogate_alpha, attn_scale=attn_scale,
            ternary_v_mode=ternary_v_mode, score_norm=score_norm, score_scale=score_scale)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    # 修复: timm>=0.9 的 create_model 会注入 pretrained_cfg_overlay 等工厂级参数,
    # vit_snn 不接受, 这里按签名动态过滤, 只透传模型自身参数(对 timm 版本变更稳健)
    # 怎么读: 用 inspect 取 vit_snn 形参表, kwargs 中不在表内的键全部丢弃
    import inspect as _inspect
    _valid = set(_inspect.signature(vit_snn.__init__).parameters.keys())
    _valid.discard('self')
    _dropped = {k: v for k, v in kwargs.items() if k not in _valid}
    if _dropped:
        print(f'[Spikingformer] 忽略 timm 注入的工厂参数: {sorted(_dropped)}')
        kwargs = {k: v for k, v in kwargs.items() if k in _valid}
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(2, 3, 32, 32, device=device)
    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).to(device)

    # print the output
    model.eval()
    y = model(input)
    print(y.shape)
    print('Test Good!')
