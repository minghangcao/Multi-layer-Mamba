import torch
from torchvision.transforms import functional as F
from data import valid_dataloader, vib_valid_dataloader, vib_valid_dataloader1
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f
from pytorch_msssim import ssim
from tqdm import tqdm
from loss.ssim1d import *


# def _valid(model, args, ep):
#     device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
#     its = vib_valid_dataloader1(args.data_dir, batch_size=1, seq_len=256*256, num_workers=0)
#     model.eval()
#     psnr_adder = Adder()
#     ssim_adder = Adder()
#
#     with torch.no_grad():
#
#         for idx, data in enumerate(tqdm(its)):
#             # input_omega, label_omega, _= data
#             input_omega, input_omega_peak, label_omega, _= data
#             input_omega = input_omega.to(device)
#             input_omega_peak = input_omega_peak.to(device)
#             label_omega = label_omega.to(device)
#
#             # pred = model(input_omega)
#             pred = model(input_omega, input_omega_peak)
#
#             pred_clip = torch.clamp(pred, 0, 1)
#             p_numpy = pred_clip.squeeze(0).cpu().numpy()
#             label_numpy = label_omega.squeeze(0).cpu().numpy()
#
#             psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
#             ssim_ = ssim1d(pred_clip, label_omega, data_range=1, size_average=False)
#
#             psnr_adder(psnr)
#             ssim_adder(ssim_)
#
#     model.train()  # 这句话的意义？？
#     return psnr_adder.average(), ssim_adder.average()

def _valid(model, args, ep):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    its = vib_valid_dataloader(args.data_dir, batch_size=1, seq_len=256*256, num_workers=0)
    model.eval()
    # psnr_adder = Adder()
    ssim_adder = Adder()
    mse_adder = Adder()

    with torch.no_grad():

        for idx, data in enumerate(tqdm(its)):
            # 三输入
            input_omega, label_omega, _= data
            input_omega = input_omega.to(device)
            label_omega = label_omega.to(device)
            pred = model(input_omega)

            ## 三输入+三峰值
            # input_omega, input_omega_peak, label_omega, _= data
            # input_omega = input_omega.to(device)
            # input_omega_peak = input_omega_peak.to(device)
            # label_omega = label_omega.to(device)
            # pred = model(input_omega, input_omega_peak)   # 三输入+三峰值

            pred_clip = pred
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_omega.squeeze(0).cpu().numpy()

            # data_range = label_numpy.max() - label_numpy.min() + 1e-6
            # psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=data_range)
            ssim_ = ssim1d(pred_clip, label_omega, data_range=1, size_average=False)

            # ★ 额外计算 MSE
            mse = torch.mean((pred - label_omega) ** 2).item()

            # psnr_adder(psnr)
            ssim_adder(ssim_)
            mse_adder(mse)

    model.train()  # 这句话的意义？？
    return mse_adder.average(), ssim_adder.average(),
