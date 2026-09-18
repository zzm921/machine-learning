"""
多标签分类
与多分类不同，多标签分类每个样本可以有多个标签。
例如，一个图片可以有多个物体，每个物体就是一个标签。
与判断单个物体不同，最后一个sigmoid函数的神经元输出一个概率
多标签则在输出层使用多个sigmoid函数的神经元，每个神经元输出一个标签的概率。
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 配置 matplotlib 支持中文显示（避免 CJK 字形缺失警告）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False   # 修复负号显示为方块的问题

"""
备注：数据集构造思路
直接用本目录的 ex3data1.mat（吴恩达 ex3 的手写数字数据）：
    输入 X：形状 (5000, 400)，5000 张 20*20 的灰度图，每行是一张图展平后的 400 个像素
    原始 y：形状 (5000, 1)，取值 1~10，其中 10 表示数字 0

它不是现成的多标签数据，标签只有一个"这张图是几"，所以需要从数字本身派生出多个标签。
这里的做法是：把"识别是几"换成一串互不排斥的判断题，每张图同时回答 4 个问题：
    label1 是不是"大数字"   -> 数字 >= 5            即 5,6,7,8,9
    label2 是不是偶数       -> 数字 % 2 == 0        即 0,2,4,6,8
    label3 是不是质数       -> 数字属于 {2,3,5,7}
    label4 有没有闭合的环   -> 数字属于 {0,6,8,9}（8 有两个环）

举例：数字 8 -> [1,1,0,1]，数字 3 -> [0,0,1,0]，数字 5 -> [1,0,1,0]
标签之间可以同时为 1（如 6 是大数字、偶数、含环，但不是质数），这就是多标签。

标签矩阵 Y 的形状是 (m, 4)，第 i 列对应第 i 个标签。
这与多分类的区别：
    多分类：输出层用 softmax，各类别互斥，输出概率之和为 1
    多标签：输出层用多个 sigmoid，每个标签独立输出一个概率，可以同时为 1
"""

# 读取手写数字数据
# 以脚本所在目录为基准拼出数据文件的绝对路径，避免在别的目录下运行时报找不到文件
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ex3data1.mat')
data = loadmat(data_path)
X = data['X']            # (5000, 400)
y = data['y'].flatten()  # (5000,)  取值 1~10

# 把原始标签 1~10 映射成实际数字 0~9（原始数据里 10 代表 0）
digit = y % 10           # (5000,)  取值 0~9

# 逐个标签独立判断，每个都是形状 (m,) 的 0/1 向量
label_big = (digit >= 5).astype(int)                 # 是否为大数字
label_even = (digit % 2 == 0).astype(int)            # 是否为偶数
label_prime = np.isin(digit, [2, 3, 5, 7]).astype(int)   # 是否为质数
label_loop = np.isin(digit, [0, 6, 8, 9]).astype(int)    # 是否含闭合环

# 按列拼接成 (m, 4) 的标签矩阵，每列对应一个标签
Y = np.column_stack([label_big, label_even, label_prime, label_loop])

# 划分训练集 / 验证集 / 测试集：70% / 15% / 15%，按数字类别分层抽样，保证各类比例一致
# 注意：stratify 只能传 1 维标签，多标签任务的 Y 是二维 (m, 4)，所以用 digit 来分层
X_train_val, X_test, Y_train_val, Y_test, digit_train_val, _ = train_test_split(
    X, Y, digit, test_size=0.15, random_state=42, stratify=digit)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.15 / 0.85, random_state=42, stratify=digit_train_val)
print(f"训练集: {X_train.shape}  验证集: {X_val.shape}  测试集: {X_test.shape}")    
# 标签顺序，后面打印和画图都用它
label_names = ['大数字', '偶数', '质数', '含闭环']

print(f"X shape: {X.shape}, Y shape: {Y.shape}")
for i, name in enumerate(label_names):
    print(f"标签{i + 1}（{name}）为正的样本数: {Y[:, i].sum()}")

# 看看 4 个标签一共出现了哪几种组合，各自多少样本
combos, counts = np.unique(Y, axis=0, return_counts=True)
print("\n标签组合统计:")
for combo, count in sorted(zip(combos, counts), key=lambda t: -t[1]):
    print(f"  {combo} -> {count} 个样本")

# 可视化：每个数字抽一个样本，显示图像并标出它的 4 个标签
plt.figure(figsize=(12, 6))
for digit_value in range(10):
    idx = np.where(digit == digit_value)[0][0]   # 取该数字的第一个样本
    # .mat 里的图像是按列优先展平的，reshape 后要转置回来才是正的
    img = X[idx].reshape(20, 20).T
    plt.subplot(2, 5, digit_value + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"数字 {digit_value}\n[{','.join(map(str, Y[idx]))}]", fontsize=10)
    plt.axis('off')

plt.suptitle(f"多标签分类数据集：手写数字，标签顺序 = {label_names}")
plt.tight_layout()
plt.show()


#定义神经网络层
model=Sequential([
    Dense(units=25, activation='relu', input_shape=(X.shape[1],)),
    Dense(units=15, activation='relu'),
    # 输出层用 linear，配合 loss 里的 from_logits=True：把 sigmoid 折进损失函数内部做，
    # 与多分类任务的输出层处理方式一致，比"先 sigmoid 再算 BCE"数值上更稳
    Dense(units=4, activation='linear')
])


model.compile(loss=BinaryCrossentropy(from_logits=True),
              # 多标签不能用 metrics=['accuracy']：Keras 会把它当成多分类，改用 argmax 比较，
              # 算出来的比例和标签对错无关（实测在 logits 上会退化成 ~0.5 的随机数）
              # BinaryAccuracy 才是逐个标签判断对错
              metrics=[BinaryAccuracy(name='elem_acc')],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 早停：验证集损失连续 10 轮不再下降就停下，并回滚到最好的那一轮权重
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=100,
                    callbacks=[early_stop])

# 画 loss 和逐元素准确率曲线（训练集 vs 验证集）
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='train loss')
axes[0].plot(history.history['val_loss'], label='val loss')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['elem_acc'], label='train elem_acc')
axes[1].plot(history.history['val_elem_acc'], label='val elem_acc')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('elem_acc')
axes[1].set_title('逐元素准确率（每个标签单独判断对错）')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 评估模型在测试集上的性能
test_loss, test_elem_acc = model.evaluate(X_test, Y_test, verbose=0)
# 逐元素准确率必须和"全零预测"对照：4 个标签的正样本比例约 0.5/0.5/0.4/0.4，
# 什么都不学、全部预测 0，也能拿到约 0.55 的逐元素准确率
all_zero_acc = 1 - Y_test.mean()
print(f"测试集损失: {test_loss:.4f}, 逐元素准确率: {test_elem_acc:.4f}")
print(f"全零预测的逐元素准确率基线: {all_zero_acc:.4f}")

# 每个样本"命中几个标签"：预测 vs 真实
# 输出层是 linear，先 sigmoid 转成概率，再按 0.5 判定每个标签是否为 1
test_probs = tf.sigmoid(model.predict(X_test, verbose=0)).numpy()
pred_counts = (test_probs >= 0.5).sum(axis=1)   # 预测命中的标签个数
true_counts = Y_test.sum(axis=1).astype(int)    # 真实命中的标签个数

print(f"\n平均命中标签数: 真实 {true_counts.mean():.2f} 个, 预测 {pred_counts.mean():.2f} 个")

plt.figure(figsize=(6, 4))
ks = np.arange(5)
width = 0.35
plt.bar(ks - width / 2, [(true_counts == k).sum() for k in ks], width, label='真实')
plt.bar(ks + width / 2, [(pred_counts == k).sum() for k in ks], width, label='预测')
plt.xlabel('一个样本命中的标签个数')
plt.ylabel('样本数')
plt.title('命中标签数分布（平均: 真实 %.2f / 预测 %.2f）' % (true_counts.mean(), pred_counts.mean()))
plt.xticks(ks)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

