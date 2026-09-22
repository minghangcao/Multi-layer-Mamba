import os
import torch
from data import train_dataloader, vib_train_dataloader, vib_train_dataloader1
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from loss.vgg import PerceptualLoss
from loss.ssim import *
from loss.ssim1d import *
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
from scipy import signal
import torchaudio.functional as AF

# def _train(model, args, logging):
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     logging.info("Total network parameters (excluding idr): %.2fM" % (total_params / 1e6))
#
#     device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     criterion = torch.nn.L1Loss()
#     # vggloss = PerceptualLoss(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
#     dataloader = vib_train_dataloader1(args.data_dir, args.batch_size, args.seq_len, args.num_worker)
#     max_iter = len(dataloader)
#     warmup_epochs = 3
#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
#     scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
#     scheduler.step()
#     epoch = 1
#     if args.resume:
#         state = torch.load(args.resume)
#         epoch = state['epoch']
#         optimizer.load_state_dict(state['optimizer'])
#         model.load_state_dict(state['model'])
#         print('Resume from %d'%epoch)
#         epoch += 1
#
#
#     ############ lossadder ############
#     writer = SummaryWriter()
#     epoch_pixel_adder = Adder()
#     epoch_ssim_adder = Adder()
#     epoch_total_adder = Adder()
#     iter_pixel_adder = Adder()
#     iter_ssim_adder = Adder()
#     iter_total_adder = Adder()
#
#     epoch_timer = Timer('m')
#     iter_timer = Timer('m')
#     best_psnr = 0
#     best_epoch = 0
#     best_ssim = 0
#
#     for epoch_idx in range(epoch, args.num_epoch + 1):
#         logging.info("\n==> Name %s, Epoch %i, previous PSNR = %.4f, SSIM = %.4f in epoch %i" % (args.model_name, epoch_idx, best_psnr, best_ssim, best_epoch))
#         epoch_timer.tic()
#         iter_timer.tic()
#         traintar = tqdm(dataloader, ncols=150)
#         total_loss = 0
#         l1_loss = 0
#         ssim_loss = 0
#
#         for iter_idx, batch_data in enumerate(traintar):
#
#             # input_omega, label_omega = batch_data
#             input_omega, input_omega_peak, label_omega = batch_data
#             input_omega = input_omega.to(device)
#             input_omega_peak = input_omega_peak.to(device)
#             label_omega = label_omega.to(device)
#
#             optimizer.zero_grad()
#             # pred_omega = model(input_omega)
#             pred_omega = model(input_omega, input_omega_peak)
#             l4 = criterion(pred_omega, label_omega)
#             loss_content = l4
#
#             ###########   SSimloss
#             s4 = 1 - torch.mean(ssim1d(pred_omega, label_omega))
#             loss_ssim = s4
#
#             loss = loss_content + 0.1 * loss_ssim
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
#             optimizer.step()
#
#             iter_pixel_adder(loss_content.item())
#             iter_ssim_adder(loss_ssim.item())
#             iter_total_adder(loss.item())
#
#             epoch_pixel_adder(loss_content.item())
#             epoch_ssim_adder(loss_ssim.item())
#             epoch_total_adder(loss.item())
#
#             total_loss += loss.item()
#             l1_loss += loss_content.item()
#             ssim_loss += loss_ssim.item()
#
#             traintar.set_description("Total_Loss: %.4f, L1_Loss: %.4f, SSim_Loss: %.4f, LR: %.6f" %
#                                      (total_loss / (iter_idx + 1), l1_loss / (iter_idx + 1),
#                                       ssim_loss / (iter_idx + 1), scheduler.get_lr()[0]))
#
#             if (iter_idx + 1) % args.print_freq == 0:
#                 writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
#                 iter_timer.tic()
#                 iter_pixel_adder.reset()
#                 iter_ssim_adder.reset()
#                 iter_total_adder.reset()
#
#         overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
#         torch.save({'model': model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'epoch': epoch_idx}, overwrite_name)
#
#         logging.info("Epoch: %i ==> Elapsed time: %.4f Total loss: %.4f L1 Loss: %.4f" % (
#             epoch_idx, epoch_timer.toc(), epoch_total_adder.average(), epoch_pixel_adder.average()))
#
#         epoch_pixel_adder.reset()
#         epoch_ssim_adder.reset()
#         epoch_total_adder.reset()
#         scheduler.step()
#
#         val, ssim_ = _valid(model, args, epoch_idx)
#         logging.info('PSNR: %.4f' % (val))
#         writer.add_scalar('PSNR', val, epoch_idx)
#         if val >= best_psnr:
#             best_psnr = val
#             #best_ssim = ssim_
#             best_ssim = 0
#             best_epoch = epoch_idx
#             torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))

# 用于无监督算法
# ===================== 可微 IIR 滤波器（不变） =====================
def iir_filter(x, b, a):
    """
    批量递归实现 IIR 滤波
    x: [B, L, 1] 输入序列
    b: 分子系数 (nb+1,)
    a: 分母系数 (na+1,)，a[0] = 1
    返回 y: [B, L, 1]
    """
    x = x.permute(0, 2, 1)
    B, L, _ = x.shape
    nb = len(b) - 1
    na = len(a) - 1
    # 确保 a[0] = 1
    assert abs(a[0] - 1.0) < 1e-6, "a[0] must be 1"
    # 扩展系数以便批量处理
    b = torch.tensor(b, dtype=torch.float32, device=x.device).view(1, 1, -1)  # [1,1,nb+1]
    a = torch.tensor(a, dtype=torch.float32, device=x.device).view(1, 1, -1)  # [1,1,na+1]
    # 存储历史输入和输出
    x_history = torch.zeros(B, 1, nb, device=x.device)   # [B,1,nb]
    y_history = torch.zeros(B, 1, na, device=x.device)   # [B,1,na]
    y = torch.zeros_like(x)
    for n in range(L):
        # 当前时刻输入
        x_n = x[:, n, :]  # [B,1]
        # 计算 FIR 部分: b[0]*x[n] + b[1]*x[n-1] + ...
        fir = torch.sum(b * torch.cat([x_n.unsqueeze(-1), x_history], dim=-1), dim=-1, keepdim=True)  # [B,1]
        # 计算 IIR 部分: -a[1]*y[n-1] - a[2]*y[n-2] - ...
        iir = -torch.sum(a[:, :, 1:] * y_history, dim=-1, keepdim=True)  # [B,1]
        y_n = fir + iir
        y[:, n, :] = y_n.squeeze(1)
        # 更新历史
        x_history = torch.cat([x_n.unsqueeze(-1), x_history[:, :, :-1]], dim=-1)  # 移除最旧
        y_history = torch.cat([y_n, y_history[:, :, :-1]], dim=-1)
    return y.permute(0, 2, 1)

def iir_filter_fast(x, b, a):
    """
    x: [B, C, L] 输入（无需 permute）
    b: 分子系数 (nb+1,)
    a: 分母系数 (na+1,)，要求 a[0] = 1
    返回 y: [B, C, L]
    """
    device = x.device
    dtype = x.dtype

    # 归一化，确保 a[0] = 1（torchaudio 要求）
    a0 = a[0]
    b = torch.tensor(b, dtype=dtype, device=device) / a0
    a = torch.tensor(a, dtype=dtype, device=device) / a0

    # torchaudio 的 lfilter 支持 [B, C, L] 或 [B, L] 或 [L]
    # 参数顺序：lfilter(x, a_coeffs, b_coeffs)
    y = AF.lfilter(x, a, b, clamp=False)
    return y

def get_filter_coeffs():
    # 传递函数参数（与损失函数中使用的系数一致）
    tau = 0.5
    a = 1.0073
    b = 0.0004
    c = -6270
    d = 4623
    e = 6084
    f = 12840
    num_acc, den_acc = [tau, 0], [tau, 1]
    num_fog, den_fog = [a], [b, 1]
    num_mhd, den_mhd = [c, d], [1, e, f]
    dt = 1 / 3000  # 注意修改

    sys_acc = signal.cont2discrete((num_acc, den_acc), dt, method='bilinear')
    sys_fog = signal.cont2discrete((num_fog, den_fog), dt, method='bilinear')
    sys_mhd = signal.cont2discrete((num_mhd, den_mhd), dt, method='bilinear')

    b_acc, a_acc = sys_acc[0].flatten(), sys_acc[1].flatten()
    b_fog, a_fog = sys_fog[0].flatten(), sys_fog[1].flatten()
    b_mhd, a_mhd = sys_mhd[0].flatten(), sys_mhd[1].flatten()

    return b_acc, a_acc, b_fog, a_fog, b_mhd, a_mhd

def relative_percentage_error(pred, target, eps=1e-6, reduction='mean'):
    """
    计算预测值相对真实值的百分比误差。
    pred, target: [B, C, L] 或任意形状
    reduction: 'mean' 返回标量, 'none' 返回逐元素误差
    """
    diff = torch.abs(pred - target)
    denom = torch.abs(target) + eps
    rel_err = diff / denom
    if reduction == 'mean':
        return rel_err.mean()
    return rel_err

def _train(model, args, logging):

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info("Total network parameters (excluding idr): %.2fM" % (total_params / 1e6))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.L1Loss()
    # vggloss = PerceptualLoss(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = vib_train_dataloader(args.data_dir, args.batch_size, args.seq_len, args.num_worker)
    max_iter = len(dataloader)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1


    ############ lossadder ############
    writer = SummaryWriter()
    epoch_total_adder = Adder()
    iter_total_adder = Adder()

    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = 0
    best_epoch = 0
    best_ssim = 0
    best_mse = 1000

    for epoch_idx in range(epoch, args.num_epoch + 1):
        logging.info("\n==> Name %s, Epoch %i, previous PSNR = %.4f, SSIM = %.4f in epoch %i" % (args.model_name, epoch_idx, best_psnr, best_ssim, best_epoch))
        epoch_timer.tic()
        iter_timer.tic()
        traintar = tqdm(dataloader, ncols=150)
        total_loss = 0

        for iter_idx, batch_data in enumerate(traintar):
            # 三输入
            input_omega, label_omega = batch_data
            input_omega = input_omega.to(device)
            label_omega = label_omega.to(device)
            optimizer.zero_grad()
            pred_omega = model(input_omega)

            # 三输入+三峰值
            # input_omega, input_omega_peak, label_omega = batch_data
            # input_omega = input_omega.to(device)
            # input_omega_peak = input_omega_peak.to(device)
            # label_omega = label_omega.to(device)
            # optimizer.zero_grad()
            # pred_omega = model(input_omega, input_omega_peak)

            b_acc, a_acc, b_fog, a_fog, b_mhd, a_mhd = get_filter_coeffs()
            # 重构传感器信号
            u1_hat = iir_filter_fast(pred_omega, b_acc, a_acc)
            u2_hat = iir_filter_fast(pred_omega, b_fog, a_fog)
            u3_hat = iir_filter_fast(pred_omega, b_mhd, a_mhd)
            # 重构损失
            loss_recon = (
                    criterion(u1_hat, input_omega[:, 0:1, :]) +
                    criterion(u2_hat, input_omega[:, 1:2, :]) +
                    criterion(u3_hat, input_omega[:, 2:3, :])
            )
            loss = loss_recon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_total_adder(loss.item())
            epoch_total_adder(loss.item())
            total_loss += loss.item()

            # 记录到日志
            traintar.set_description("Total_Loss: %.4f, LR: %.6f" %
                                     (total_loss / (iter_idx + 1), scheduler.get_lr()[0]))

            if (iter_idx + 1) % args.print_freq == 0:
                writer.add_scalar('Total Loss', iter_total_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_total_adder.reset()

        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        logging.info("Epoch: %i ==> Elapsed time: %.4f Total loss: %.4f " % (
            epoch_idx, epoch_timer.toc(), epoch_total_adder.average()))

        epoch_total_adder.reset()
        scheduler.step()

        val, ssim_ = _valid(model, args, epoch_idx)
        logging.info('MSE: %.4f' % (val))
        writer.add_scalar('MSE', val, epoch_idx)
        if val <= best_mse:
            best_mse = val
            best_epoch = epoch_idx
            torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
