import torch
import torch.nn.functional as F
import math

def _fspecial_gauss_1d(size, sigma):
    """生成 1D 高斯窗口"""
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1)  # shape (1, 1, size)

def gaussian_filter1d(input, win):
    """
    对输入进行 1D 卷积（使用高斯窗口），输入形状 [B, C, L]
    win: (1, 1, win_size)
    """
    # 分组卷积，每组通道独立
    B, C, L = input.shape
    # 卷积后保持长度不变，padding = win_size//2
    padding = win.shape[-1] // 2
    out = F.conv1d(input, win, stride=1, padding=padding, groups=C)
    return out

def ssim1d(X, Y, win_size=11, win_sigma=1.5, data_range=1.0, size_average=True):
    """
    计算 1D 信号的 SSIM
    X, Y: [B, C, L]  (C 通常为 1，也可用于多通道)
    """
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError('Inputs must be 3-d tensors [B, C, L]')

    if X.shape != Y.shape:
        raise ValueError('Inputs must have the same shape')

    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 生成窗口并重复到通道数
    win = _fspecial_gauss_1d(win_size, win_sigma).to(X.device)
    win = win.repeat(X.shape[1], 1, 1)  # (C, 1, win_size)

    mu1 = gaussian_filter1d(X, win)
    mu2 = gaussian_filter1d(Y, win)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter1d(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter1d(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter1d(X * Y, win) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * \
               ((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map  # 返回每个样本的 SSIM（可继续处理）