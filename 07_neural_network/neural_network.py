"""
激活函数
平均方差(mse)：  y可正可负，一般用于回归任务
sigmoid: 1/(1+exp(-z))   二分类使用，输出 (0,1) 之间的概率
ReLU: max(0,z)   比sigmoid 更简单，计算速度快，但是会使一部分神经元的输出为0，导致“死亡”

神经网络训练流程：
数据准备 -> 搭建模型 -> 编译(损失/优化器/指标) -> 训练 -> 评估与预测 -> 可视化

本例任务：XOR(异或) 问题
两个特征相同 -> 标签 0，两个特征不同 -> 标签 1
异或问题不是线性可分的，单个逻辑回归无法拟合，必须依靠隐藏层做特征变换
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# 1. 数据准备
# ============================================================
# 在 [0,1]x[0,1] 的正方形区域内均匀采样，按异或规则打标签
# x1、x2 是否都 > 0.5，相同则为 0，不同则为 1
np.random.seed(1)

n_samples = 1000

# 特征矩阵：形状 (n_samples, 2)，每一行是一个样本的两个特征
X = np.random.uniform(0, 1, size=(n_samples, 2))

# 标签：异或规则，形状 (n_samples, 1)，二分类用 0/1 表示
y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.float32)
y = y.reshape(-1, 1)

# 异或的 4 个角点，训练完成后用来直观检查预测结果
xor_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# ============================================================
# 2. 搭建模型
# ============================================================
# Sequential：按顺序堆叠的线性结构，前一层的输出就是后一层的输入
# Dense：全连接层，内部计算 z = W·a_in + b，再交给激活函数
model = tf.keras.Sequential([
    # 隐藏层：3 个神经元，sigmoid 激活
    #   input_shape=(2,) 表示输入是 2 维特征(由第一层负责声明输入维度)
    #   异或问题至少需要 2 个隐藏神经元才能把它们强行"撑开"成线性可分
    tf.keras.layers.Dense(3, activation='sigmoid', input_shape=(2,)),

    # 输出层：1 个神经元，sigmoid 激活
    #   二分类只需要 1 个输出，输出值就是"标签为 1 的概率"
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 打印网络结构(每层参数量、输出形状)
model.summary()

# 如果想画网络结构图，需要额外安装 graphviz/pydot，这里先注释掉：
# tf.keras.utils.plot_model(model, show_shapes=True, to_file='neural_network.png')

# ============================================================
# 3. 编译：指定损失函数、优化器、评估指标
# ============================================================
model.compile(
    # 损失函数：二分类交叉熵(输出层是 sigmoid 概率，配 binary_crossentropy)
    #   回归任务才用 mse(平均方差)
    loss='binary_crossentropy',

    # 优化器：Adam 自适应学习率，比原始梯度下降收敛更快；lr 太大可能震荡，太小收敛慢
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),

    # 评估指标：训练过程中额外打印准确率，方便观察拟合情况
    metrics=['accuracy']
)

# ============================================================
# 4. 训练
# ============================================================
# fit 会执行"前向传播 -> 算损失 -> 反向传播 -> 更新 W 和 b"的循环 epochs 次
# history 里保存了每一轮的 loss 和 accuracy，用于后面画学习曲线
#
# ---- GPU 加速说明（NVIDIA 显卡）----
# 1) Keras 会自动把计算放到可用的 GPU 上，下面这段训练代码不需要任何改动
# 2) 检查是否识别到 GPU（返回 [] 说明当前装的是 CPU 版 TensorFlow）：
#        print(tf.config.list_physical_devices('GPU'))
# 3) 想显式指定在 GPU 上训练：
#        with tf.device('/GPU:0'):
#            history = model.fit(X, y, epochs=1000, batch_size=32)
# 4) 原生 Windows 上 TensorFlow >= 2.11 不支持 GPU（即使装了 CUDA 也不会启用），
#    只支持到 tensorflow 2.10 + CUDA 11.2 + cuDNN 8.1，且需要 Python 3.9/3.10；
#    版本不匹配时能看到 'GPU support is not available on native Windows' 警告。
#    想在高版本 TensorFlow 上用 GPU 需要改用 WSL2：
#        pip install "tensorflow[and-cuda]"
# 5) 本例只有 1000 个样本、13 个可训练参数，计算量太小，
#    GPU 的核函数启动和数据拷贝开销大于计算本身，通常不会比 CPU 快、甚至更慢；
#    样本量上万、batch_size 调大、或换成卷积网络后 GPU 才有明显优势
history = model.fit(
    X, y,
    epochs=1000,      # 迭代轮数：太少欠拟合，太多容易过拟合(本例数据简单，多一些无妨)
    batch_size=32,    # 每批样本数：权重每处理一个 batch 更新一次
    verbose=0         # 0 不打印过程，避免 1000 轮刷屏
)

print(f"训练结束：loss={history.history['loss'][-1]:.4f}，"
      f"accuracy={history.history['accuracy'][-1]:.4f}")

# ============================================================
# 5. 评估与预测
# ============================================================
# evaluate：在数据上计算编译时指定的损失和指标(不做参数更新)
test_loss, test_acc = model.evaluate(X, y, verbose=0)
print(f"评估结果：loss={test_loss:.4f}，accuracy={test_acc:.4f}")

# predict：前向传播得到预测概率，形状 (n, 1)
probs = model.predict(xor_points, verbose=0)

print("XOR 四个角点的预测：")
for point, prob in zip(xor_points, probs.ravel()):
    # 概率 >= 0.5 判为正类 1，否则为 0
    label = 1 if prob >= 0.5 else 0
    print(f"  {point.astype(int)} -> 概率 {prob:.4f}，预测 {label}，"
          f"真实 {int(point[0] > 0.5) ^ int(point[1] > 0.5)}")

# ============================================================
# 6. 可视化
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：学习曲线，观察损失是否收敛
ax1.plot(history.history['loss'], label='loss')
ax1.plot(history.history['accuracy'], label='accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('value')
ax1.set_title('Training curve')
ax1.legend()
ax1.grid(alpha=0.3)

# 右图：决策边界
# 在平面上铺一张网格，逐点预测概率，用颜色深浅表示"标签为 1 的概率"
xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, 200), np.linspace(-0.1, 1.1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]          # 拼成 (40000, 2) 的输入
zz = model.predict(grid, verbose=0).reshape(xx.shape)

ax2.contourf(xx, yy, zz, levels=20, cmap='RdBu_r', alpha=0.8)
ax2.scatter(X[y.ravel() == 0][:, 0], X[y.ravel() == 0][:, 1],
            c='blue', s=10, label='y=0')
ax2.scatter(X[y.ravel() == 1][:, 0], X[y.ravel() == 1][:, 1],
            c='red', s=10, label='y=1')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Decision boundary')
ax2.legend()

plt.tight_layout()
plt.show()