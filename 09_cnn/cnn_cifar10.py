"""
彩色图片分类：卷积神经网络(CNN) 训练 CIFAR-10

数据集：CIFAR-10，32x32 彩色图(RGB 三通道)，10 类
        airplane(飞机) automobile(汽车) bird(鸟) cat(猫) deer(鹿)
        dog(狗) frog(青蛙) horse(马) ship(船) truck(卡车)
        50000 张训练 + 10000 张测试（由 prepare_data.py cifar10 生成）

--------------------------------------------------------------------------
和 cnn.py(MNIST 灰度图) 相比，流程完全一样，只有 3 处不同：
--------------------------------------------------------------------------
    1) 输入通道数：灰度 1 -> 彩色 3
       体现在 Input(shape=(32,32,3)) 上

    2) 数据要不要手动加通道维
       MNIST 数据是 (N,28,28)，必须自己 [..., np.newaxis] 补成 (N,28,28,1)
       CIFAR-10 数据本身就是 (N,32,32,3)，不用加

    3) 第一层卷积核的"深度"
       灰度：卷积核是 3x3x1，32 个核 = 3*3*1*32+32 = 320 个参数
       彩色：卷积核是 3x3x3，32 个核 = 3*3*3*32+32 = 896 个参数
       注意只有第一层受影响！第二层往后的输入通道数由"上一层的 filters"决定，
       跟原始图片有几个通道没有任何关系，所以代码写法完全一致。

为什么流程能一样？
    Conv2D 内部做的是三维卷积(高、宽、通道)：一个卷积核在 3 个方向上滑动，
    对通道方向是"加权求和"后压成一个数，所以输出永远是一张二维特征图。
    灰度是"1 个通道求和"，彩色是"3 个通道求和"，机制没有区别。

--------------------------------------------------------------------------
层数怎么定？——同样用"尺寸倒推法"
--------------------------------------------------------------------------
每池化一次尺寸减半：32 -> 16 -> 8 -> 4 -> 2
所以 32x32 最多池化 4~5 次，本网络用 3 个卷积块(池化 3 次，尺寸落到 4x4)。

注意 CIFAR-10 比 MNIST 难得多（彩色、目标形态多变、类间相似如猫狗），
准确率 70% 左右就是正常水平，不要拿 MNIST 的 99% 来对比。
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 让 matplotlib 正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据准备
# ============================================================
# 用 __file__ 拼绝对路径，保证在任何工作目录下运行都能找到数据文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_data = np.load(os.path.join(BASE_DIR, 'data', 'cifar10_train.npz'))
test_data = np.load(os.path.join(BASE_DIR, 'data', 'cifar10_test.npz'))

# npz 里存的是原始 uint8 像素(0~255)，形状已经是 (N, 32, 32, 3)
x_train, y_train = train_data['x'], train_data['y']
x_test, y_test = test_data['x'], test_data['y']

# 10 个类别的名称，顺序和标签 0~9 一一对应（可视化时显示用）
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 归一化：0~255 -> 0~1 ---
# 神经网络对输入数值范围敏感。不归一化时输入值很大，
# 会导致梯度尺度混乱，训练又慢又不稳定。
x_train = (x_train / 255.0).astype('float32')
x_test = (x_test / 255.0).astype('float32')

# --- 不需要加通道维 ---
# 彩色图数据本身就是 (N, 32, 32, 3) 四维，符合 Conv2D 要求。
# (MNIST 的 (N,28,28) 才需要 [..., np.newaxis] 补成 1 个通道)

print(f'训练集：x={x_train.shape}, y={y_train.shape}')
print(f'测试集：x={x_test.shape}, y={y_test.shape}')
print(f'像素范围：{x_train.min()} ~ {x_train.max()}')

# ============================================================
# 2. 搭建模型
# ============================================================
# Sequential：按顺序堆叠的线性结构，上一层的输出就是下一层的输入
model = models.Sequential()

# ---- 输入层 ----
# (高, 宽, 通道数)。彩色图通道数 = 3 (R/G/B)。
# 这里的 3 只影响第一层卷积核的深度，后面每一层的通道数都由 filters 决定
model.add(layers.Input(shape=(32, 32, 3)))

# ---- 卷积块 1 ----
# Conv2D 参数含义：
#   filters=32        卷积核的个数 = 输出通道数 = 提取多少种特征
#   kernel_size=(3,3) 卷积核大小，3x3 最常用
#   padding='same'    在图像边缘补 0，让输出尺寸和输入保持一致
#                     (不加的话每层会缩 2 个像素，尺寸变化不好追)
#   activation='relu' 引入非线性
# 输出尺寸：用 same 保持不变，32 -> 32
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))

# MaxPooling2D(2x2)：每 2x2 区域取最大值，宽高各减半 -> 16
model.add(layers.MaxPooling2D((2, 2)))

# ---- 卷积块 2 ----
# 通道数 32 -> 64 翻倍：尺寸缩小了，用更多通道容纳特征
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))  # 16
model.add(layers.MaxPooling2D((2, 2)))                                   # 16 -> 8

# ---- 卷积块 3 ----
# 32x32 的图尺寸宽裕，可以再池化一次，堆到 3 个卷积块
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))  # 8
model.add(layers.MaxPooling2D((2, 2)))                                   # 8 -> 4

# ---- 分类头 ----
# Flatten：把 (4,4,64) 三维特征图拉平成 4*4*64 = 1024 维一维向量
#   Dense 只接受一维输入，所以必须有这一层做桥接
model.add(layers.Flatten())

# Dropout(0.5)：训练时随机"关掉"一半神经元，防止过拟合。
#   CIFAR-10 比 MNIST 更容易过拟合(图像更复杂、训练样本相对更少)，所以很必要。
model.add(layers.Dropout(0.5))

# 全连接层：1024 -> 128，把卷积提取的特征组合起来
model.add(layers.Dense(128, activation='relu'))

# 输出层：128 -> 10
#   10 = 类别数；softmax 转成 10 个类的概率分布
model.add(layers.Dense(10, activation='softmax'))

# 打印每一层的输出形状和参数量
model.summary()

# ============================================================
# 3. 编译：指定损失函数、优化器、评估指标
# ============================================================
model.compile(
    # 优化器：Adam 自适应学习率，收敛快，一般默认就用它
    optimizer='adam',

    # 损失函数：多分类 + 标签是整数(0~9) -> sparse_categorical_crossentropy
    #   "sparse(稀疏)" 指标签以单个整数给出，而不是 one-hot 向量
    loss='sparse_categorical_crossentropy',

    # 训练过程中额外打印准确率，方便观察
    metrics=['accuracy']
)

# ============================================================
# 4. 训练
# ============================================================
# CIFAR-10 比 MNIST 难得多，轮数要给够。
# 实测 10 轮时损失还在稳定下降(0.83 -> 0.77)，说明远没收敛，所以这里用 30 轮。
history = model.fit(
    x_train, y_train,
    batch_size=64,          # 每批 64 个样本(彩色图更占内存，比 MNIST 调大一点)
    epochs=30,              # 迭代轮数
    validation_split=0.1,   # 从训练集切 10% 做验证集，用于监控过拟合
                            # 注意：不动 test 测试集，测试集只在最后评估时用一次
    verbose=1
)

# ============================================================
# 5. 评估与预测
# ============================================================
# evaluate：在测试集上计算损失和准确率(不做参数更新)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\n测试集结果：loss={test_loss:.4f}，accuracy={test_acc:.4f}')

# predict：前向传播得到每个样本属于 10 个类的概率，形状 (N, 10)
probs = model.predict(x_test, verbose=0)
# argmax(axis=1)：取概率最大的那个类的下标 = 预测的类别
y_pred = probs.argmax(axis=1)

# 逐类看一下准确率，能发现"哪类最容易认错"(比如猫狗常常互相认错)
print('\n各类别测试准确率：')
for i, name in enumerate(class_names):
    mask = y_test == i                     # 选出真实标签为 i 的样本
    acc = (y_pred[mask] == y_test[mask]).mean()
    print(f'  {name:<11}：{acc:.4f}（共 {mask.sum()} 张）')

# ============================================================
# 6. 可视化
# ============================================================
# 图 1：训练曲线
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='训练 loss')
plt.plot(history.history['val_loss'], label='验证 loss')
plt.plot(history.history['accuracy'], label='训练 accuracy')
plt.plot(history.history['val_accuracy'], label='验证 accuracy')
plt.xlabel('epoch')
plt.ylabel('value')
plt.title('训练曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 图 2：随机抽 10 张测试图，显示预测结果
idx = np.random.choice(len(x_test), 10, replace=False)

plt.figure(figsize=(14, 6))
for i, k in enumerate(idx):
    plt.subplot(2, 5, i + 1)
    # 彩色图直接用 imshow，不要传 cmap='gray'，也不用 reshape
    # (MNIST 才需要 reshape(28,28) + cmap='gray')
    plt.imshow(x_test[k])
    # 预测对用绿色标题，预测错用红色标题，一目了然
    correct = y_pred[k] == y_test[k]
    plt.title(f'预测:{class_names[y_pred[k]]}\n真实:{class_names[y_test[k]]}',
              color='green' if correct else 'red')
    plt.axis('off')

plt.tight_layout()
plt.show()
