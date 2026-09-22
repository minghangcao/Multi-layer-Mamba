import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

# ------------------- 自定义 1D 数据增强（与之前一致） -------------------

class PairCompose1D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class PairRandomCrop1D:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, x, y):
        total_len = x.shape[0]
        if total_len < self.seq_len:
            # 长度不足则填充零
            pad_len = self.seq_len - total_len
            x = np.pad(x, ((0, pad_len), (0, 0)), 'constant')
            y = np.pad(y, ((0, pad_len), (0, 0)), 'constant')
            return x, y
        else:
            start = np.random.randint(0, total_len - self.seq_len + 1)
            return x[start:start+self.seq_len], y[start:start+self.seq_len]


class PairTimeReverse1D:
    def __call__(self, x, y):
        if random.random() > 0.5:
            return x[::-1, :], y[::-1, :]
        return x, y


class PairToTensor1D:
    def __call__(self, x, y):
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


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
        all_inputs = []   # 每个样本 (L, 3)
        all_labels = []   # 每个样本 (L, 1)
        self.samples = [] # 存储标准化后的 (input, label) 元组

        for file in self.file_list:
            df = pd.read_csv(os.path.join(data_dir, file))
            # 提取需要的列，忽略时间戳
            inputs = df[['omega_acc', 'omega_fog', 'omega_mhd']].values.astype(np.float32)  # (L, 3)
            labels = df[['omega_true']].values.astype(np.float32)                           # (L, 1)
            all_inputs.append(inputs)
            all_labels.append(labels)

        # 2. 计算全局均值和标准差（按通道）
        if stats is None:
            # 将所有输入拼接成 (total_L, 3)
            concat_inputs = np.vstack(all_inputs)
            concat_labels = np.vstack(all_labels)
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
        for inp, lab in zip(all_inputs, all_labels):
            inp_norm = (inp - self.input_mean) / self.input_std
            lab_norm = (lab - self.label_mean) / self.label_std
            self.samples.append((inp_norm, lab_norm))

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
        x, y = self.samples[idx]  # 已标准化，形状 (L, 3) 和 (L, 1)

        # 应用数据增强（裁剪、翻转、转张量等）
        if self.transform:
            x, y = self.transform(x, y)
        else:
            # 若没有 transform，则直接裁剪到固定长度（测试时使用）
            if x.shape[0] < self.seq_len:
                pad_len = self.seq_len - x.shape[0]
                x = np.pad(x, ((0, pad_len), (0, 0)), 'constant')
                y = np.pad(y, ((0, pad_len), (0, 0)), 'constant')
            else:
                # 测试时取前 seq_len 个点（或可改为随机，但测试应固定）
                x = x[:self.seq_len]
                y = y[:self.seq_len]
            x = torch.from_numpy(x.astype(np.float32))
            y = torch.from_numpy(y.astype(np.float32))
        x = x.permute(1, 0)
        y = y.permute(1, 0)
        if self.is_test:
            name = self.file_list[idx]
            return x, y, name
        return x, y


# ------------------- 工厂函数 -------------------

def vib_train_dataloader(data_path, batch_size=64, seq_len=256*256, num_workers=0, use_transform=True):
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


def vib_test_dataloader(data_path, batch_size=1, seq_len=256*256, num_workers=0):
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
    # 测试时无增强，仅转 Tensor（不裁剪，由数据集内部处理）
    # transform = PairToTensor1D()
    dataset = VibrationCSVDataset(test_dir, seq_len=seq_len,
                                  is_test=True, stats=stats_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return dataloader


def vib_valid_dataloader(data_path, batch_size=1, seq_len=256*256, num_workers=0):
    # 验证集通常与测试集相同逻辑
    return vib_test_dataloader(data_path, batch_size, seq_len, num_workers)