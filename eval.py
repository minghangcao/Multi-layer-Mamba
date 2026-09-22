import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader, vib_test_dataloader, vib_test_dataloader1
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def _eval(model, args):
    state_dict = torch.load(args.test_model, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = vib_test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()


    # 3. 加载训练时保存的标准化统计量（用于反标准化）
    stats_path = os.path.join(args.data_dir, 'stats.npz')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f'统计量文件 {stats_path} 不存在，请先运行训练。')
    stats = np.load(stats_path)
    label_mean = stats['label_mean']
    label_std = stats['label_std']

    # 4. 初始化累加器（用于计算平均指标）
    mse_adder = Adder()
    mae_adder = Adder()
    time_adder = Adder()

    # 5. 创建保存目录
    if args.save_data:
        save_dir = args.save_path if hasattr(args, 'save_path') else './predictions'
        os.makedirs(save_dir, exist_ok=True)


    with torch.no_grad():

        for iter_idx, data in enumerate(dataloader):

            # 三输入
            input_omega, label_omega, name = data
            input_omega = input_omega.to(device)
            label_omega = label_omega.to(device)
            tm = time.time()
            pred = model(input_omega)

            # 三输入+三峰值
            # input_omega, input_omega_peak, label_omega, name = data
            # input_omega = input_omega.to(device)
            # input_omega_peak = input_omega_peak.to(device)  # 三输入+三峰值
            # label_omega = label_omega.to(device)
            # tm = time.time()
            # pred = model(input_omega, input_omega_peak)   # 三输入+三峰值

            elapsed = time.time() - tm
            time_adder(elapsed)

            # 反标准化（恢复到原始量纲）
            pred_np = pred.cpu().numpy()  # [1, L, 1]
            label_np = label_omega.cpu().numpy()  # [1, L, 1]
            pred_orig = pred_np * label_std + label_mean
            label_orig = label_np * label_std + label_mean

            # 计算评价指标
            mse = mean_squared_error(label_orig.flatten(), pred_orig.flatten())
            mae = mean_absolute_error(label_orig.flatten(), pred_orig.flatten())
            mse_adder(mse)
            mae_adder(mae)

            print(f'样本 {iter_idx + 1}: MSE = {mse:.6f}, MAE = {mae:.6f}, 耗时 {elapsed:.4f}s')

            # 保存预测结果
            if args.save_data:
                # 保存为 CSV，包含真实值和预测值
                save_path = os.path.join(save_dir, f'{name}_pred.csv')
                df = pd.DataFrame({
                    'true': label_orig.flatten(),
                    'pred': pred_orig.flatten()
                })
                df.to_csv(save_path, index=False)
                # 也可同时保存为 npy
                # np.save(os.path.join(save_dir, f'{name}_pred.npy'), pred_orig)

            # 输出总体指标
        print('==========================================================')
        print(f'平均 MSE : {mse_adder.average():.6f}')
        print(f'平均 MAE : {mae_adder.average():.6f}')
        print(f'平均耗时 : {time_adder.average():.6f} s')

