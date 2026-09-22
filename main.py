import os
import argparse
# from train_multi import _train
from train import _train
from eval import _eval
from valid import _valid
import logging
import datetime
logging.getLogger('PIL').setLevel(logging.WARNING)
from models.mymamba import MyMamba
# from models.mymamba1 import MyMamba
import torch
torch.autograd.set_detect_anomaly(True)
import random
import numpy as np


def init_logging(filedir: str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')
    logger = logging.getLogger()
    fh = logging.FileHandler(filename=filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  #!
    torch.backends.cudnn.benchmark = False       #!
    torch.backends.cudnn.enabled = True         #! for accelerating training


def main(args, log=None):

    model = MyMamba()

    if args.mode == 'train':
        _train(model, args, log)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MyMamba', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    # parser.add_argument('--data_dir', type=str, default="/home/students/doctor/2024/tanzj/Dataset/Underwater/UIEB/")
    parser.add_argument('--data_dir', type=str, default="E:/WorkFile/角速度数据分析/py_files/Multi-layer_Mamba/dataset/UIEB/")

    # Train
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=256*256)
    # Test
    parser.add_argument('--test_model', type=str, default="E:/WorkFile/角速度数据分析/py_files/Multi-layer_Mamba/Checkpoints/Mamba/best.pkl")
    parser.add_argument('--save_data', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save_path', type=str, default='result')

    args = parser.parse_args()

    setup_seed(1234)

    if args.mode == 'train':
        args.model_save_dir = os.path.join('Checkpoints/', args.model_name)
        # 打印绝对路径
        abs_path = os.path.abspath(args.model_save_dir)
        print(f"模型将保存至（绝对路径）: {abs_path}")
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        logging = init_logging(args.model_save_dir)
        logging.info(args)
        main(args, logging)
    else:
        main(args)
