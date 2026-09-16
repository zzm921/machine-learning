
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path    
# 配置 matplotlib 支持中文显示（避免 CJK 字形缺失警告）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False   # 修复负号显示为方块的问题
# ===== 数据集说明 =====
# ex2data1.txt（2 个特征 + 目标，3 列）
#   第0列：exam1（第一次考试成绩）
#   第1列：exam2（第二次考试成绩）
#   第2列：admitted（是否被录取，1=录取，0=未录取）
# =====================
DATA_FILE = 'ex2data1.txt'   # 数据集文件名

# 读取数据
df_data = pd.read_csv(Path(__file__).parent / f'./{DATA_FILE}', header=None)

# 查看数据前五行
print("===== 查看数据前五行 =====")
print(df_data.head())

print("===== 查看数据信息 =====")
print(df_data.info())


print("===== 查看数据统计信息 =====")
print(df_data.describe())


#取出正负样本
positive=df_data[df_data.iloc[:, -1] == 1]
negative=df_data[df_data.iloc[:, -1] == 0]
# 画出离散分布点，x1为特征1，y2为特征2，根据目标值区分颜色
plt.scatter(positive.iloc[:, 0], positive.iloc[:, 1], marker='o', label='Admitted')
plt.scatter(negative.iloc[:, 0], negative.iloc[:, 1], marker='x', label='Not Admitted')
plt.xlabel('特征 1（第 1 列）')
plt.ylabel('特征 2（第 2 列）')
plt.legend()
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
逻辑回归中，代价函数用于评估模型的分类效果。
x 是输入变量（特征），y 是实际输出变量（目标），θ 是模型的参数（权重向量，已包含偏置项）。
逻辑回归的代价函数 J(θ)=(1/m)*sum(-y_i*log(F(x_i))-(1-y_i)*log(1-F(x_i)))  
其中，F(X_i) 是模型的预测值，m 是样本数量。

模型定义为
F(X)=sigmoid(g(X))    （sigmoid 函数将任意值映射到 (0, 1) 区间）
g(X)=X·θ
sigmoid 函数定义为：
    sigmoid(z) = 1 / (1 + exp(-z))

"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    # 计算样本数量
    m = len(y)
    # 计算预测值（np.clip 防止 sigmoid 出现 0/1 导致 log(0)=inf/NaN）
    predictions = np.clip(sigmoid(X.dot(theta)), 1e-10, 1 - 1e-10)
    # 计算代价
    cost = (1 / m) * np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))

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
    predictions = sigmoid(X.dot(theta))
    # 代价函数是 J(θ)=(1/m)*sum(-y_i*log(F(x_i))-(1-y_i)*log(1-F(x_i)))  
    # 其中，F(x_i) 是模型的预测值，m 是样本数量
   # 对每个特征 j，计算梯度 ∂J(θ)/∂θ_j=(1/m)*sum((F(x_i)-y_i)*x_i_j)
   # 其中，x_i_j 是样本 i 第 j 个特征值
    d_theta = (1 / m) * X.T.dot(predictions - y)

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



X=get_X(df_data)
y=get_y(df_data)

# 初始化参数与超参
theta = np.zeros(X.shape[1])   # X 已含偏置列，θ 为 n+1 维向量（theta[0] 即偏置 b）
alpha = 0.01     # 学习率
epoch = 1500     # 迭代次数

# 运行训练
theta, cost_data = batch_gradient_descent(X, y, theta, alpha, epoch)

# 逻辑回归
# 先画出所有样本（原始空间），x轴是特征1，y轴是特征2
plt.scatter(positive.iloc[:, 0], positive.iloc[:, 1], marker='o', label='Admitted')
plt.scatter(negative.iloc[:, 0], negative.iloc[:, 1], marker='x', label='Not Admitted')
plt.xlabel('特征 1（第 1 列）')
plt.ylabel('特征 2（第 2 列）')
plt.legend()

# 画出逻辑回归边界  g(X)=X·θ=0，即 θ[0]+θ[1]*x1+θ[2]*x2=0
# θ 是在缩放后的特征空间求得的，需把边界线变换回原始特征空间再绘制
X_raw = df_data.iloc[:, :-1].values
mu = np.mean(X_raw, axis=0)         # 按列均值
sigma = np.std(X_raw, axis=0)       # 按列标准差

# 在原始空间取特征 1 的绘图范围，再缩放到训练时的特征空间
x1_raw = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 100)
x1_scaled = (x1_raw - mu[0]) / sigma[0]
# 缩放空间内边界：θ[0] + θ[1]*x1_scaled + θ[2]*x2_scaled = 0
x2_scaled = -(theta[1] * x1_scaled + theta[0]) / theta[2]
# 变换回原始空间
x2_raw = x2_scaled * sigma[1] + mu[1]
plt.plot(x1_raw, x2_raw, color='red', label='Decision Boundary')
# 限制坐标轴范围在数据范围内，避免边界线延伸过远导致散点图被压缩成一团
x_margin = 0.05 * (X_raw[:, 0].max() - X_raw[:, 0].min())
y_margin = 0.05 * (X_raw[:, 1].max() - X_raw[:, 1].min())
plt.xlim(X_raw[:, 0].min() - x_margin, X_raw[:, 0].max() + x_margin)
plt.ylim(X_raw[:, 1].min() - y_margin, X_raw[:, 1].max() + y_margin)
plt.legend()
plt.show()

# 收敛曲线：代价随迭代次数逐渐下降
plt.plot(cost_data)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()

# （可选）预测
# 新样本需使用与训练时相同的特征缩放（按列 axis=0，mu/sigma 已在边界绘制处求得）
new_x = np.array([45, 85])   # 例：exam1=45，exam2=85
new_x_scaled = (new_x - mu) / sigma
new_X = np.concatenate(([1], new_x_scaled))  # 拼接偏置列
pred = sigmoid(new_X.dot(theta))
print(f'预测结果: {pred:.4f}')




   


