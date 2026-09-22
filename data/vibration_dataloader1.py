import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

# ------------------- 自定义 1D 数据增强（支持多输入同步） -------------------

class PairCompose1D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        # args: (x, x_p, y) 或 (x, y)
        for t in self.transforms:
            args = t(*args)
            if not isinstance(args, tuple):
                args = (args,)
        return args


class PairRandomCrop1D:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, *args):
        # args: (x, x_p, y) 或 (x, y)
        total_len = args[0].shape[0]
        if total_len < self.seq_len:
            pad_len = self.seq_len - total_len
            return tuple(np.pad(a, ((0, pad_len), (0, 0)), 'constant') for a in args)
        else:
            start = np.random.randint(0, total_len - self.seq_len + 1)
            return tuple(a[start:start + self.seq_len] for a in args)


class PairTimeReverse1D:
    def __call__(self, *args):
        if random.random() > 0.5:
            return tuple(a[::-1, :] for a in args)
        return args


class PairToTensor1D:
    def __call__(self, *args):
        return tuple(torch.from_numpy(a.astype(np.float32)) for a in args)


# ------------------- 核心数据集类（读取 CSV，全局标准化） -------------------

class VibrationCSVDataset(Dataset):
    def __init__(self, data_dir, seq_len=1024, transform=None, is_test=False, stats=None):
        """
        data_dir   : 包含 .csv 文件的目录
        seq_len    : 固定序列长度
        transform  : 数据增强（如随机翻转、转张量）
        is_test    : 是否为测试集（若 True 则返回文件名）
        stats      : 预计算的均值和标准差字典，若 None 则从数据中计算
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.is_test = is_test
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.file_list.sort()
        if not self.file_list:
            raise RuntimeError(f'No CSV files found in {data_dir}')

        # 1. 读取所有数据并拼接（用于计算全局统计量）
        all_inputs = []       # 原始传感器 (L, 3)
        all_peak_inputs = []  # 掩码后传感器 (L, 3)
        all_labels = []       # 真实值 (L, 1)
        self.samples = []     # 存储标准化后的 (input, peak_input, label)

        for file in self.file_list:
            df = pd.read_csv(os.path.join(data_dir, file))
            # 原始传感器列
            inputs = df[['omega_acc', 'omega_fog', 'omega_mhd']].values.astype(np.float32)
            # 掩码列
            peak_inputs = df[['omega_acc_peak5', 'omega_fog_peak5', 'omega_mhd_peak5']].values.astype(np.float32)
            # 真实值列
            labels = df[['omega_true']].values.astype(np.float32)
            all_inputs.append(inputs)
            all_peak_inputs.append(peak_inputs)
            all_labels.append(labels)

        # 2. 计算全局均值和标准差（仅基于原始传感器，掩码列沿用相同统计量）
        if stats is None:
            concat_inputs = np.vstack(all_inputs)   # (total_L, 3)
            concat_labels = np.vstack(all_labels)   # (total_L, 1)
            self.input_mean = concat_inputs.mean(axis=0, keepdims=True)   # (1, 3)
            self.input_std = concat_inputs.std(axis=0, keepdims=True) + 1e-6
            self.label_mean = concat_labels.mean()
            self.label_std = concat_labels.std() + 1e-6
        else:
            self.input_mean = stats['input_mean']
            self.input_std = stats['input_std']
            self.label_mean = stats['label_mean']
            self.label_std = stats['label_std']

        # 3. 标准化所有样本并存储
        for inp, peak_inp, lab in zip(all_inputs, all_peak_inputs, all_labels):
            inp_norm = (inp - self.input_mean) / self.input_std
            # 掩码列使用相同的均值和标准差（量纲一致）
            peak_norm = (peak_inp - self.input_mean) / self.input_std
            lab_norm = (lab - self.label_mean) / self.label_std
            self.samples.append((inp_norm, peak_norm, lab_norm))

        # 保存统计量以便测试时加载
        self.stats = {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'label_mean': self.label_mean,
            'label_std': self.label_std
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, x_p, y = self.samples[idx]  # 均为 numpy 数组，形状 (L, 3), (L, 3), (L, 1)

        # 应用数据增强（裁剪、翻转、转张量等）
        if self.transform:
            x, x_p, y = self.transform(x, x_p, y)
        else:
            # 若没有 transform，则手动裁剪/填充到固定长度并转 Tensor
            if x.shape[0] < self.seq_len:
                pad_len = self.seq_len - x.shape[0]
                x = np.pad(x, ((0, pad_len), (0, 0)), 'constant')
                x_p = np.pad(x_p, ((0, pad_len), (0, 0)), 'constant')
                y = np.pad(y, ((0, pad_len), (0, 0)), 'constant')
            else:
                x = x[:self.seq_len]
                x_p = x_p[:self.seq_len]
                y = y[:self.seq_len]
            x = torch.from_numpy(x.astype(np.float32))
            x_p = torch.from_numpy(x_p.astype(np.float32))
            y = torch.from_numpy(y.astype(np.float32))

        # 调整维度为 (C, L) 以适配模型（若模型需要 [B, L, C] 则此处可不转置，按需修改）
        x = x.permute(1, 0)      # (3, L)
        x_p = x_p.permute(1, 0)  # (3, L)
        y = y.permute(1, 0)      # (1, L)

        if self.is_test:
            name = self.file_list[idx]
            return x, x_p, y, name
        return x, x_p, y


# ------------------- 工厂函数 -------------------

def vib_train_dataloader1(data_path, batch_size=64, seq_len=256*256, num_workers=0, use_transform=True):
    train_dir = os.path.join(data_path, 'train')
    transform = None
    if use_transform:
        transform = PairCompose1D([
            PairRandomCrop1D(seq_len),      # 随机裁剪
            PairTimeReverse1D(),            # 随机时间反转
            PairToTensor1D()                # 转为 Tensor
        ])
    dataset = VibrationCSVDataset(train_dir, seq_len=seq_len, transform=transform, is_test=False)
    # 保存统计量到文件（以便测试时使用）
    stats = dataset.stats
    np.savez(os.path.join(data_path, 'stats.npz'),
             input_mean=stats['input_mean'],
             input_std=stats['input_std'],
             label_mean=stats['label_mean'],
             label_std=stats['label_std'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    return dataloader


def vib_test_dataloader1(data_path, batch_size=1, seq_len=256*256, num_workers=0):
    test_dir = os.path.join(data_path, 'test')
    # 加载训练时保存的统计量
    stats_path = os.path.join(data_path, 'stats.npz')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f'Please run training first to generate {stats_path}')
    stats = np.load(stats_path)
    stats_dict = {
        'input_mean': stats['input_mean'],
        'input_std': stats['input_std'],
        'label_mean': stats['label_mean'],
        'label_std': stats['label_std']
    }
    # 测试时无增强，但依然需要转 Tensor 和裁剪，由数据集内部处理
    dataset = VibrationCSVDataset(test_dir, seq_len=seq_len,
                                  is_test=True, stats=stats_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return dataloader


def vib_valid_dataloader1(data_path, batch_size=1, seq_len=256*256, num_workers=0):
    return vib_test_dataloader1(data_path, batch_size, seq_len, num_workers)