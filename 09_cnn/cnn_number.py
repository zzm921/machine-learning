"""
手写数字识别：卷积神经网络(CNN)

数据集：MNIST，0~9 共 10 类手写数字，28x28 灰度图
        60000 张训练 + 10000 张测试（由 prepare_data.py 生成到 data/train.npz、data/test.npz）

--------------------------------------------------------------------------
一、为什么图片任务要用 CNN，而不是 07/08 章的全连接网络(Dense)？
--------------------------------------------------------------------------
全连接网络必须先把图片 Flatten 成一维向量(28x28=784)再送进去，
这一步会把"上下左右相邻像素"的空间关系彻底丢掉——
而笔画、边缘、拐角这些图案，恰恰是靠空间位置关系才能识别的。

卷积层(Conv2D)用一个小卷积核(例如 3x3)在整张图上滑动，两个关键好处：
    1) 局部连接：每个输出只依赖一小块区域，天然适合捕捉局部笔画/边缘
    2) 权值共享：同一个卷积核扫遍全图，参数量远小于全连接
       对比：3x3x1 的卷积核提出 32 个特征图 = 3*3*1*32+32 = 320 个参数
             784 维输入接 32 个神经元 = 784*32+32  = 25088 个参数

--------------------------------------------------------------------------
二、层数怎么定？——用"尺寸倒推法"，不是拍脑袋
--------------------------------------------------------------------------
每做一次 MaxPooling2D(2x2)，特征图的宽高就减半：
    28 -> 14 -> 7 -> 3
所以 28x28 的图最多池化 3 次；再池化就变成 1x1，空间信息全没了，卷积失去意义。
=> 卷积块数量上限是 3，本网络用 3 个卷积块 + 1 个分类头。

通道数的规律：尺寸减半时通道翻倍(32 -> 64)，保持信息量平衡。
浅层学局部细节(边缘/笔画)，深层学更抽象的组合特征。

(作为对比：VGG16 能吃 224x224 的大图，可以池化 5~6 次，所以能堆到 16 层。
 你拿 28x28 的 MNIST 去堆 50 层，池化 5 次就到 1x1 了，纯属浪费。)
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

train_data = np.load(os.path.join(BASE_DIR, 'data', 'train.npz'))
test_data = np.load(os.path.join(BASE_DIR, 'data', 'test.npz'))

# npz 里存的是原始 uint8 像素(0~255)，形状 (N, 28, 28)
x_train, y_train = train_data['x'], train_data['y']
x_test, y_test = test_data['x'], test_data['y']

# 类别名称，用于可视化时显示标签
class_names = [str(i) for i in range(10)]

# --- 归一化：0~255 -> 0~1 ---
# 神经网络对输入数值范围敏感。不归一化时输入值很大，
# 会导致梯度尺度混乱，训练又慢又不稳定。
x_train = (x_train / 255.0).astype('float32')
x_test = (x_test / 255.0).astype('float32')

# --- 加通道维 ---
# Conv2D 要求输入是 (N, 高, 宽, 通道数) 四维张量。
# 灰度图通道数 = 1，彩色图(RGB)才是 3。
# [..., np.newaxis] 把 (N,28,28) 变成 (N,28,28,1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print(f'训练集：x={x_train.shape}, y={y_train.shape}')
print(f'测试集：x={x_test.shape}, y={y_test.shape}')
print(f'像素范围：{x_train.min()} ~ {x_train.max()}')

# ============================================================
# 2. 搭建模型
# ============================================================
# Sequential：按顺序堆叠的线性结构，上一层的输出就是下一层的输入
model = models.Sequential()

# ---- 输入层 ----
# 显式声明输入形状 (高, 宽, 通道数)，只有第一层需要写。
# Keras 3 推荐用独立的 Input 层，而不是把 input_shape 塞给 Conv2D
# (后者虽然也能用，但会打 UserWarning)。
model.add(layers.Input(shape=(28, 28, 1)))

# ---- 卷积块 1 ----
# Conv2D 参数含义：
#   filters=32        卷积核的个数 = 输出通道数 = 提取多少种特征
#   kernel_size=(3,3) 卷积核大小。3x3 最常用：感受野够用、参数还少
#                     (5x5 的参数是 3x3 的约 2.8 倍，一般用两个 3x3 替代)
#   activation='relu' 引入非线性，否则多层卷积叠加等价于一层线性变换
# 输出尺寸：28-3+1 = 26  （不带 padding 的卷积会让尺寸缩小 kernel_size-1）
model.add(layers.Conv2D(32, (3, 3), activation='relu'))

# MaxPooling2D(2x2)：在每 2x2 的区域里取最大值，宽高各减半
#   作用：1) 降采样，减少后面的计算量和参数量
#         2) 提供一定的平移不变性(数字稍微偏一点也能认出来)
# 输出尺寸：26 -> 13
model.add(layers.MaxPooling2D((2, 2)))

# ---- 卷积块 2 ----
# 通道数 32 -> 64 翻倍：此时尺寸又缩小了，用更多通道来容纳特征
model.add(layers.Conv2D(64, (3, 3), activation='relu'))   # 13-3+1 = 11
model.add(layers.MaxPooling2D((2, 2)))                     # 11 -> 5

# ---- 卷积块 3 ----
# 这一块后面故意不再接池化，尺寸停在 3x3。
# 它的作用不只是"加深"：注意当前特征图是 (5,5,64)=1600 维，
# 如果不加这一层直接 Flatten，后面的 Dense 就要吃 1600 个输入，
# 参数量会翻倍。先用卷积压到 (3,3,64)=576 维，全连接层能省一大半参数。
model.add(layers.Conv2D(64, (3, 3), activation='relu'))   # 5-3+1 = 3

# ---- 分类头 ----
# Flatten：把 (3,3,64) 的三维特征图拉平成 3*3*64 = 576 维一维向量。
#   Dense 只接受一维输入，所以卷积部分和全连接部分之间必须有这一层做桥接。
model.add(layers.Flatten())

# Dropout(0.5)：训练时每一批随机"关掉"50% 的神经元。
#   作用是防止过拟合(网络死记硬背训练样本，导致验证集表现差)。
#   只在训练时生效，预测/评估时自动关闭。放在全连接层前最常见。
model.add(layers.Dropout(0.5))

# 全连接层：576 -> 64，把卷积提取的特征组合起来
model.add(layers.Dense(64, activation='relu'))

# 输出层：64 -> 10
#   10 = 类别数(数字 0~9)
#   softmax 把 10 个原始分数转成概率分布(每项在 0~1、总和为 1)，
#   取最大值对应的下标就是预测出的数字
model.add(layers.Dense(10, activation='softmax'))

# 打印每一层的输出形状和参数量
model.summary()

# ============================================================
# 3. 编译：指定损失函数、优化器、评估指标
# ============================================================
model.compile(
    # 优化器：Adam 自适应学习率，比原始梯度下降收敛快，一般默认就用它
    optimizer='adam',

    # 损失函数：多分类 + 标签是整数(0~9) -> sparse_categorical_crossentropy
    #   "sparse(稀疏)" 指的是标签以单个整数的形式给出，而不是 one-hot 向量。
    #   如果标签已经 one-hot 编码过，才要用 categorical_crossentropy。
    #   配 softmax 输出层使用。
    loss='sparse_categorical_crossentropy',

    # 训练过程中额外打印准确率，方便观察
    metrics=['accuracy']
)

# ============================================================
# 4. 训练
# ============================================================
# fit 会循环 epochs 次，每次做"前向传播 -> 算损失 -> 反向传播 -> 更新权重"
# history 保存了每一轮的 loss / accuracy / val_loss / val_accuracy，用来画曲线
history = model.fit(
    x_train, y_train,
    batch_size=32,          # 每批 32 个样本，权重每处理完一批更新一次
    epochs=5,               # 迭代轮数。MNIST 比较简单，5 轮就能到 99% 左右
    validation_split=0.1,   # 从训练集里切 10% 出来做验证集，用于监控过拟合
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
# argmax(axis=1)：取概率最大的那个类的下标 = 预测的数字
y_pred = probs.argmax(axis=1)

# 逐类看一下准确率，能发现"哪个数字最容易认错"
print('\n各类别测试准确率：')
for i, name in enumerate(class_names):
    mask = y_test == i                     # 选出真实标签为 i 的样本
    acc = (y_pred[mask] == y_test[mask]).mean()
    print(f'  数字 {name}：{acc:.4f}（共 {mask.sum()} 张）')

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
    plt.imshow(x_test[k].reshape(28, 28), cmap='gray')
    # 预测对用绿色标题，预测错用红色标题，一目了然
    correct = y_pred[k] == y_test[k]
    plt.title(f'预测:{class_names[y_pred[k]]} 真实:{class_names[y_test[k]]}',
              color='green' if correct else 'red')
    plt.axis('off')

plt.tight_layout()
plt.show()
