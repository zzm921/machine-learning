
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
from itertools import combinations_with_replacement

# 配置 matplotlib 支持中文显示（避免 CJK 字形缺失警告）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False   # 修复负号显示为方块的问题

# ===== 数据集说明 =====
# ex2data2.txt（2 个原始特征 + 目标，3 列）
#   第0列：microchip test 1（微芯片测试 1 得分）
#   第1列：microchip test 2（微芯片测试 2 得分）
#   第2列：accepted（芯片是否合格，1=合格，0=不合格）
# 该数据线性不可分，需要构造多项式特征 + 正则化来控制过拟合
# =====================
DATA_FILE = 'ex2data2.txt'

# 读取数据
df_data = pd.read_csv(DATA_FILE, header=None)

print("===== 查看数据前五行 =====")
print(df_data.head())

print("===== 查看数据统计信息 =====")
print(df_data.describe())

# 取出正负样本
positive = df_data[df_data.iloc[:, -1] == 1]
negative = df_data[df_data.iloc[:, -1] == 0]

# 图 1：原始数据分布（线性不可分，肉眼可见无法用一条直线划分）
plt.scatter(positive.iloc[:, 0], positive.iloc[:, 1], marker='o', label='Accepted')
plt.scatter(negative.iloc[:, 0], negative.iloc[:, 1], marker='x', label='Rejected')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.title('原始数据分布（线性不可分）')
plt.show()

# ===== 构造多项式特征 =====
def map_feature(X, degree=3):
    # 把 n 个原始特征映射为 degree 次以内的所有多项式特征（第 1 列是偏置项全 1）
    # 通用支持任意数量特征：n 个特征、degree 次，特征总数 = C(degree+n, n)
    # 2 个特征、degree=3 时 = C(5,2) = 10 个
    m = X.shape[0]
    out = [np.ones(m)]   # 偏置列（全 1）
    for d in range(1, degree + 1):
        # 总次数为 d 的单项式：从 n 个特征中"有放回"取 d 个相乘，覆盖所有指数组合
        # 例如 n=2, d=2 时依次为 x0²、x0·x1、x1²
        for exps in combinations_with_replacement(range(X.shape[1]), d):
            feat = np.ones(m)
            for j in exps:
                feat *= X[:, j]
            out.append(feat)
    return np.column_stack(out)


# 提取原始特征与目标
X_raw = df_data.iloc[:, :-1].values
y = df_data.iloc[:, -1].values

# 多项式映射（含偏置列）
X_poly = map_feature(X_raw)
print('多项式特征维度:', X_poly.shape)   # (118, 10)

# 特征缩放：只缩放特征列（第 1 列起），偏置列保持全 1，帮助梯度下降更快收敛
mu = np.mean(X_poly[:, 1:], axis=0)
sigma = np.std(X_poly[:, 1:], axis=0)
X = np.column_stack((np.ones(len(X_poly)), (X_poly[:, 1:] - mu) / sigma))

# ===== 正则化逻辑回归 =====
"""
带正则化的代价函数：
J(θ) = (1/m)·Σ[-y·log(h) - (1-y)·log(1-h)] + (λ/(2m))·Σ_{j=1..n} θ_j²

- 第一项与普通逻辑回归相同（h = sigmoid(X·θ)）
- 第二项是正则项：对所有特征 θ_j（j≥1）施加惩罚，θ₀（偏置）不参与惩罚
- λ 控制惩罚强度：λ 越大，θ 被压得越小，模型越简单，越不容易过拟合

梯度更新：
θ₀ = θ₀ - α·(1/m)·X₀ᵀ(h-y)                  （θ₀ 无惩罚）
θ_j = θ_j - α·[(1/m)·X_jᵀ(h-y) + (λ/m)·θ_j]   （j≥1，比普通梯度多一项 (λ/m)·θ_j）
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function_reg(X, y, theta, lam):
    m = len(y)
    # 计算预测值（np.clip 防止 sigmoid 出现 0/1 导致 log(0)=inf/NaN）
    predictions = np.clip(sigmoid(X.dot(theta)), 1e-10, 1 - 1e-10)
    # 基础代价（与普通逻辑回归相同）
    cost = (1 / m) * np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    # 正则项：θ₀（theta[0]）不参与惩罚
    reg = (lam / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg


def gradient_descent_reg(X, y, theta, alpha, lam):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    # 普通梯度：(1/m)·Xᵀ(h-y)
    d_theta = (1 / m) * X.T.dot(predictions - y)
    # 正则化梯度：(λ/m)·θ，θ₀ 置零表示偏置不参与惩罚 
    reg_grad = (lam / m) * theta
    # theta 0 是偏置项，不参与正则化
    reg_grad[0] = 0
    # 更新参数
    theta = theta - alpha * (d_theta + reg_grad)
    return theta


def batch_gradient_descent_reg(X, y, theta, alpha=0.1, epoch=5000, lam=0):
    cost_data = [cost_function_reg(X, y, theta, lam)]   # 初始代价（列表，方便 .append()）
    _theta = np.copy(theta)
    for i in range(epoch):
        _theta = gradient_descent_reg(X, y, _theta, alpha, lam)
        cost_data.append(cost_function_reg(X, y, _theta, lam))
    return _theta, cost_data


# ===== 训练：对比不同 λ =====
alpha = 0.1     # 学习率
epoch = 5000    # 迭代次数
lambdas = [0, 1, 100]   # λ=0 不惩罚（过拟合），λ=1 适中，λ=100 惩罚过强（欠拟合）

models = {}
for lam in lambdas:
    theta_l, cost_l = batch_gradient_descent_reg(X, y, np.zeros(X.shape[1]), alpha, epoch, lam)
    models[lam] = (theta_l, cost_l)
    # 训练集准确率
    pred = (sigmoid(X.dot(theta_l)) >= 0.5).astype(int)
    acc = np.mean(pred == y)
    print(f'λ={lam:>3}: 最终代价 = {cost_l[-1]:.4f}, 训练集准确率 = {acc:.1%}')


# ===== 图 2：三个 λ 的决策边界（等高线）=====
# 在原始特征空间生成网格 → 多项式映射 → 缩放 → 预测概率 → 画 sigmoid=0.5 的等高线
plt.figure(figsize=(8, 6))
plt.scatter(positive.iloc[:, 0], positive.iloc[:, 1], marker='o', label='Accepted')
plt.scatter(negative.iloc[:, 0], negative.iloc[:, 1], marker='x', label='Rejected')

grid_x1, grid_x2 = np.meshgrid(np.linspace(-1.1, 1.2, 300), np.linspace(-1.1, 1.2, 300))
grid_pts = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
grid_poly = map_feature(grid_pts)
grid_X = np.column_stack((np.ones(len(grid_poly)), (grid_poly[:, 1:] - mu) / sigma))

colors = {0: 'red', 1: 'green', 100: 'blue'}
for lam in lambdas:
    theta_l = models[lam][0]
    prob = sigmoid(grid_X.dot(theta_l)).reshape(grid_x1.shape)
    # 概率 = 0.5 的等高线即决策边界
    plt.contour(grid_x1, grid_x2, prob, levels=[0.5], linewidths=2, colors=colors[lam])

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('不同 λ 的决策边界（λ=0 过拟合 / λ=1 适中 / λ=100 欠拟合）')
handles = [
    mlines.Line2D([], [], color='red', label='λ=0'),
    mlines.Line2D([], [], color='green', label='λ=1'),
    mlines.Line2D([], [], color='blue', label='λ=100'),
]
plt.legend(handles=handles)
plt.show()


# 图 3：收敛曲线
plt.figure(figsize=(8, 5))
for lam in lambdas:
    plt.plot(models[lam][1], label=f'λ={lam}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend()
plt.title('不同 λ 的收敛曲线')
plt.show()


# （可选）预测：新样本需走 多项式映射 → 缩放 → sigmoid 全流程
new_x = np.array([0.2, -0.3])   # 例：test1=0.2，test2=-0.3
new_poly = map_feature(new_x.reshape(1, -1))   # reshape 成 (1, n) 单样本矩阵
new_X = np.concatenate(([1], (new_poly[0, 1:] - mu) / sigma))
pred = sigmoid(new_X.dot(models[1][0]))
print(f'新样本 {new_x}: λ=1 模型预测合格概率 = {pred:.4f}')
