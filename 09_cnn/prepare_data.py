"""
准备数据集（卷积神经网络的训练集 / 测试集）

支持两个数据集：

1) MNIST —— 手写数字
   28x28 灰度图，0~9 共 10 类，官方划分 60000 训练 + 10000 测试

2) CIFAR-10 —— 彩色物体图片
   32x32 RGB 三通道图，10 类（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车），
   官方划分 50000 训练 + 10000 测试

输出文件（保存在本文件同级的 data 目录下）：
    MNIST    -> data/train.npz            x: (60000, 28, 28)    y: (60000,)
                data/test.npz             x: (10000, 28, 28)    y: (10000,)
    CIFAR-10 -> data/cifar10_train.npz    x: (50000, 32, 32, 3) y: (50000,)
                data/cifar10_test.npz     x: (10000, 32, 32, 3) y: (10000,)

    x 为 uint8 原始像素（0~255），y 为 uint8 整数标签（0~9）。

注意：这里保存的是原始 0~255 像素，不做归一化，
      归一化（除以 255.0）放到模型训练脚本里做，方便你自由调整预处理方式。

运行方式：
    python prepare_data.py            # 默认准备 MNIST
    python prepare_data.py cifar10    # 准备 CIFAR-10

数据来源：都由 tf.keras.datasets 提供，首次运行会联网下载并缓存到
          ~/.keras/datasets，之后直接读缓存，不再联网。
"""

import argparse
import os

import numpy as np
import tensorflow as tf

# ============================================================
# 路径：统一放在本脚本同级的 data 目录，避免受当前工作目录影响
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================
# 数据集配置表
# ============================================================
# 两个数据集在"加载方式"上是同一套流程，只有下面这些参数不同，
# 所以用一个字典描述差异，避免把同一段代码写两遍。
DATASETS = {
    'mnist': {
        # tf.keras.datasets.mnist.load_data 这个函数本身
        'loader': tf.keras.datasets.mnist.load_data,
        'img_shape': (28, 28),      # 单张图的形状（灰度图没有通道维）
        'channels': 1,              # 通道数：灰度 1
        'classes': list('0123456789'),
        'prefix': '',               # 输出文件名前缀（保持和历史文件名一致）
    },
    'cifar10': {
        'loader': tf.keras.datasets.cifar10.load_data,
        'img_shape': (32, 32, 3),   # 彩色图本身就带通道维
        'channels': 3,              # 通道数：RGB 3
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'],
        'prefix': 'cifar10_',
    },
}


def prepare(name):
    cfg = DATASETS[name]

    os.makedirs(DATA_DIR, exist_ok=True)

    # ========================================================
    # 1. 下载并加载
    # ========================================================
    # 返回值统一是 ((x_train, y_train), (x_test, y_test))
    (x_train, y_train), (x_test, y_test) = cfg['loader']()

    # CIFAR-10 的标签形状是 (N, 1)，多一维没意义，统一拍平成一维 (N,)
    # (MNIST 本来就是 (N,)，reshape 后不变，所以两个数据集可以共用这句)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    print(f'[{name}] 原始训练集：x={x_train.shape}, y={y_train.shape}')
    print(f'[{name}] 原始测试集：x={x_test.shape}, y={y_test.shape}')

    # ========================================================
    # 2. 基本检查
    # ========================================================
    # 图像形状、数据类型、标签范围都要符合预期，否则后面训练会出各种怪问题
    n_classes = len(cfg['classes'])
    assert x_train.shape[1:] == cfg['img_shape'], f'图像形状异常：{x_train.shape}'
    assert x_test.shape[1:] == cfg['img_shape'], f'图像形状异常：{x_test.shape}'
    assert x_train.dtype == np.uint8 and y_train.dtype == np.uint8
    assert y_train.min() == 0 and y_train.max() == n_classes - 1

    # 打印各类别样本数量，检查类别是否均衡
    counts = np.bincount(y_train, minlength=n_classes)
    print(f'[{name}] 各类别样本数（训练集）：')
    for i, cname in enumerate(cfg['classes']):
        print(f'    {i:>2} {cname:<11} {counts[i]}')

    # ========================================================
    # 3. 保存为 npz
    # ========================================================
    # np.savez_compressed：把多个数组打包成一个压缩文件，读取时按 key 取
    train_path = os.path.join(DATA_DIR, f"{cfg['prefix']}train.npz")
    test_path = os.path.join(DATA_DIR, f"{cfg['prefix']}test.npz")

    np.savez_compressed(train_path, x=x_train, y=y_train)
    np.savez_compressed(test_path, x=x_test, y=y_test)

    print(f'[{name}] 已保存：{train_path}')
    print(f'[{name}] 已保存：{test_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备数据集为 npz 格式')
    parser.add_argument('dataset', nargs='?', default='mnist',
                        choices=sorted(DATASETS), help='要准备的数据集（默认 mnist）')
    args = parser.parse_args()
    prepare(args.dataset)
