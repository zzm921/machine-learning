
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path    

# ===== 数据集说明 =====
# data1 = ex1data1.txt（单特征，2 列）
#   第0列：人口 population
#   第1列：利润 profit（目标）
# data2 = ex1data2.txt（多特征，3 列）
#   第0列：房屋面积 area
#   第1列：卧室数 bedrooms
#   第2列：房价 price（目标）
# =====================
DATA_FILE = 'ex1data1.txt'   # 改为 'ex1data2.txt' 即切换为多特征数据

# 读取数据
df_data = pd.read_csv(Path(__file__).parent / f'./{DATA_FILE}', header=None)

# 查看数据前五行
print("===== 查看数据前五行 =====")
print(df_data.head())

print("===== 查看数据信息 =====")
print(df_data.info())


print("===== 查看数据统计信息 =====")
print(df_data.describe())



# 画图，查看原始点的分布（y 轴始终取最后一列目标）
plt.scatter(df_data.iloc[:, 0], df_data.iloc[:, -1], marker='o')
plt.xlabel('特征 0（第 0 列）')
plt.ylabel('目标（最后一列）')
plt.show()


"""
特征缩放
"""
# 特征缩放
def get_X(df_data):
    # 提取特征列：除最后一列外的所有列（最后一列为目标 y）
    X = df_data.iloc[:, :-1].values
    # 特征缩放：每个特征分别减去均值、除以标准差（按列 axis=0）
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # 添加偏置项（全 1 列），对应 theta[0]——即原来的偏置 b，b 已并入 theta 向量，不再单独使用
    X = np.column_stack((np.ones(len(X)), X))
    return X


def get_y(df_data):
    # 提取目标列（最后一列）
    y = df_data.iloc[:, -1].values
    return y


# 定义代价函数
"""
代价函数（Cost Function）是用于衡量模型预测值与实际值之间差异的函数。
在线性回归中，代价函数用于评估模型的拟合程度。
x 是输入变量（特征），y 是实际输出变量（目标），θ 是模型的参数（权重向量，已包含偏置项）。
线性回归的代价函数 J(θ)=(1/2m)*sum(F(x_i)-y_i)^2  
其中，F(X_i) 是模型的预测值，m 是样本数量。

模型定义为
F(X)=X·θ    （θ 的第一个元素对应偏置项，X 的第一列为全 1）

"""
def cost_function(X, y, theta):
    # 计算样本数量
    m = len(y)
    # 计算预测值
    predictions = X.dot(theta)
    # 计算代价
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))

    return cost

"""
· 梯度下降算法（Gradient Descent）是一种优化算法，用于最小化代价函数。
· 它通过迭代更新模型参数，以逐渐减小代价函数的值。
· 梯度下降算法的更新规则为：
    θ = θ - α * ∂J(θ)/∂θ
· 其中，α 是学习率（learning rate），用于控制每次迭代的步长。
· 梯度下降算法的目标是找到使得代价函数 J(θ) 最小的参数 θ。
"""
def gradient_descent(X, y, theta, alpha):
    # 计算样本数量
    m=len(y)
    # 计算预测值
    predictions = X.dot(theta)
    # 计算代价函数的导数（向量化：X.T.dot(误差) 一次算出所有特征（含偏置）的梯度）
    d_theta=(1/m)*X.T.dot(predictions-y)

    # 更新参数
    theta=theta-alpha*d_theta
    return theta


def batch_gradient_descent(X, y, theta, alpha=0.01, epoch=1000):
    # 迭代更新参数
    cost_data=[cost_function(X, y, theta)]  # 初始代价（列表，方便 .append()）
    _theta=np.copy(theta)
    for i in range(epoch):
       _theta=gradient_descent(X, y, _theta, alpha)
       cost_data.append(cost_function(X, y, _theta))
    return _theta,cost_data


"""
===== 接下来如何训练（备注）=====

【训练整体思路】
初始化参数 → 反复"算代价 → 算梯度 → 更新参数"迭代 N 次
→ 用代价曲线确认收敛 → 用拟合直线确认效果

【第 1 步】准备数据
    X = get_X(df_data)   # 特征矩阵（已缩放、已含偏置列）
    y = get_y(df_data)   # 目标
    ⚠️ 偏置 b 已并入 X 的第一列全 1，因此训练结果中 theta[0] 就是原来的 b

【第 2 步】初始化参数与超参
    theta = np.zeros(X.shape[1])   # X 已含偏置列，θ 为 n+1 维向量（theta[0] 即偏置 b）
    alpha = 0.01     # 学习率
    epoch = 1500     # 迭代次数

【第 3 步】运行训练
    theta, cost_data = batch_gradient_descent(X, y, theta, alpha, epoch)

【第 4 步】可视化验证（必做）
    图 A 拟合效果：散点图(原始数据) + 直线 y = theta[0] + theta[1]*x
        ⚠️ 特征已缩放，画图时 x 需用缩放后的值（或用均值和标准差反变换回原始尺度）
    图 B 收敛曲线：plt.plot(cost_data)，应单调下降并趋于水平
    若曲线震荡或上升 → 学习率太大，需调小 alpha

【第 5 步】(可选) 学习率对比
    用 alpha = 0.01 / 0.03 / 0.1 各训练一次，把三条 cost 曲线画在一起对比收敛速度

【第 6 步】(可选) 预测
    new_x 先按训练时的均值和标准差做特征缩放（按列），再拼接偏置列
    new_X = np.concatenate(([1], new_x_scaled))
    pred = new_X.dot(theta)
"""

X=get_X(df_data)
y=get_y(df_data)

# 初始化参数与超参
theta = np.zeros(X.shape[1])   # X 已含偏置列，θ 为 n+1 维向量（theta[0] 即偏置 b）
alpha = 0.01     # 学习率
epoch = 1500     # 迭代次数

# 运行训练
theta, cost_data = batch_gradient_descent(X, y, theta, alpha, epoch)

# 可视化验证
# 图 A 拟合效果
# 单特征：原始数据散点 + 拟合直线；多特征：预测值 vs 实际值（越贴近 y=x 越好）
if X.shape[1] == 2:
    plt.scatter(df_data.iloc[:, 0], y, marker='o')
    plt.plot(df_data.iloc[:, 0], X.dot(theta), color='red')
    plt.xlabel('特征 0（第 0 列）')
    plt.ylabel('目标（最后一列）')
else:
    plt.scatter(y, X.dot(theta))
    lo, hi = float(y.min()), float(y.max())
    plt.plot([lo, hi], [lo, hi], color='red', ls='--')
    plt.xlabel('实际值 y')
    plt.ylabel('预测值')
plt.show()

# 图 B 收敛曲线
plt.plot(cost_data)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()

# （可选）预测
# 新样本需使用与训练时相同的特征缩放（按列 axis=0）
if DATA_FILE == 'ex1data1.txt':
    new_x = np.array([8.0])         # data1：人口 8 万
else:
    new_x = np.array([2104, 3])     # data2：面积 2104、卧室 3

X_raw = df_data.iloc[:, :-1].values
mu = np.mean(X_raw, axis=0)         # 按列均值
sigma = np.std(X_raw, axis=0)       # 按列标准差
new_x_scaled = (new_x - mu) / sigma
new_X = np.concatenate(([1], new_x_scaled))  # 拼接偏置列
pred = new_X.dot(theta)
print(f'预测结果: {pred:.4f}')




   


