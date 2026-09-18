"""
多分类神经网络
激活函数 
    1) 输出层：softmax 激活
    2) 隐藏层：ReLU 激活

softmax 激活函数，分独立步骤进行计算的时候，会有计算误差。
例如，一个 10 分类问题，每个样本的输出是一个 10 维向量，
每个元素代表该样本属于该类的概率。如果直接对这个向量应用 softmax 函数，
会得到一个归一化的概率分布。

但是，当我们将这个向量分独立步骤进行计算时，例如先计算每个元素的指数，
再对所有指数求和，最后将每个指数除以总和，就会得到不同的结果。
这是因为在计算指数时，会引入一些舍入误差，导致最终的结果与直接应用 softmax 函数不同。

解决方法
tensorflow 最后一层使用线性函数，在计算loss中传入 from_logits=True
"""
import os 
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ex3data1.mat')
data = loadmat(data_path)
X = data['X']            # (5000, 400)
y = data['y'].flatten()  # (5000,)  取值 1~10
# 把原始标签 1~10 映射成实际数字 0~9（原始数据里 10 代表 0）
digit = y % 10           # (5000,)  取值 0~9
#画图展示数据
plt.figure(figsize=(12, 6))
for digit_value in range(10):
    idx = np.where(digit == digit_value)[0][0]   # 取该数字的第一个样本
    # .mat 里的图像是按列优先展平的，reshape 后要转置回来才是正的
    img = X[idx].reshape(20, 20).T
    plt.subplot(2, 5, digit_value + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Digit: {digit_value}")
    plt.axis('off')

plt.show()

# 原始，直接使用softmax激活函数。
# model = Sequential([
#     Dense(units=25, activation='relu', input_shape=(X.shape[1],)),
#     Dense(units=15, activation='relu'),
#     Dense(units=10, activation='softmax')
# ])

# model.compile(loss=SparseCategoricalCrossentropy())


# 划分训练集 / 验证集 / 测试集：70% / 15% / 15%，按数字类别分层抽样，保证各类比例一致
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, digit, test_size=0.15, random_state=42, stratify=digit)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15 / 0.85, random_state=42, stratify=y_train_val)
print(f"训练集: {X_train.shape}  验证集: {X_val.shape}  测试集: {X_test.shape}")

model = Sequential([
    Dense(units=25, activation='relu', input_shape=(X.shape[1],)),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100)

# 画 loss 和 accuracy 曲线（训练集 vs 验证集）
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='train loss')
axes[0].plot(history.history['val_loss'], label='val loss')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='train accuracy')
axes[1].plot(history.history['val_accuracy'], label='val accuracy')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('accuracy')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 在测试集上评估（只看结果，不再更新参数）
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集 loss: {test_loss:.4f}  accuracy: {test_acc:.4f}")

# 测试案例：从测试集随机取 1 个样本做预测演示
sample_idx = np.random.randint(len(X_test))
sample = X_test[sample_idx]
true_label = int(y_test[sample_idx])

# 输出层是 linear，拿到的是 logits，需要手动 softmax 转成概率
logits = model.predict(sample.reshape(1, -1), verbose=0)[0]
probs = tf.nn.softmax(logits).numpy()
pred_label = int(np.argmax(probs))

plt.figure(figsize=(4, 4))
plt.imshow(sample.reshape(20, 20).T, cmap='gray')
plt.title(f"true: {true_label}   predict: {pred_label}")
plt.axis('off')
plt.show()

print(f"测试样本索引: {sample_idx}  真实标签: {true_label}  预测标签: {pred_label}")
print("10 个类别的预测概率:")
for d in range(10):
    print(f"  数字 {d}: {probs[d]:.4f}")