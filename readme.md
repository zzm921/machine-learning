# Machine Learning 学习仓库

吴恩达机器学习课程实践代码，从基础库到线性回归、逻辑回归的完整学习路径。

## 目录结构

| 目录 | 内容 | 说明 |
|---|---|---|
| `01_numpy` | numpy_use.py | NumPy 基础用法 |
| `02_pandas` | pandas_use.py + data.csv | Pandas 数据读取与分析 |
| `03_matplotlib` | matplotlib_use.py | Matplotlib 绘图基础 |
| `04_linear_regression` | linear_regression.py / .ipynb | 线性回归（梯度下降、多特征、特征缩放） |
| `05_logistic_regression` | logistic_regression.py / .ipynb | 逻辑回归（二分类、决策边界） |
| | regularized_logistic_regression.py / .ipynb | 逻辑回归 + 多项式特征 + L2 正则化 |

## 环境依赖

- Python 3.x
- numpy、pandas、matplotlib
- jupyter / nbconvert（运行 notebook）

## 模块说明

### 04 线性回归

- 数据：`ex1data1.txt`（单特征：人口→利润）、`ex1data2.txt`（多特征：面积/卧室→房价）
- 实现：特征标准化、代价函数、批量梯度下降、收敛曲线、学习率对比、预测

### 05 逻辑回归

#### logistic_regression（线性边界）

- 数据：`ex2data1.txt`，两门考试成绩预测录取，正负样本各半
- 模型：`h(x) = sigmoid(θ₀ + θ₁·x₁ + θ₂·x₂)`
- 实现：特征标准化、代价函数、梯度下降、决策边界绘制（缩放空间 → 原始空间反变换）、学习率对比实验
- 学习率实验结论：

| α | 最终代价 | 表现 |
|---|---|---|
| 0.001 | 0.53 | 收敛过慢 |
| 0.01 | 0.28 | 较慢 |
| 0.1 | 0.21 | 快 |
| 0.3 | 0.20 | 最快，接近最优 |

#### regularized_logistic_regression（非线性边界 + 正则化）

- 数据：`ex2data2.txt`，微芯片两项测试得分预测质量，线性不可分
- 特征映射：把 n 个原始特征扩展为 degree 次以内全部多项式特征（通用支持任意特征数）
  - 特征总数 = C(degree + n, n)，默认 degree=3，2 特征时为 10 个
  - degree 越大模型表达力越强，但也越容易过拟合
- 正则化：代价函数对 θ（θ₀ 除外）加 L2 惩罚项 λ/2m·Σθ²，梯度更新对应做权重衰减
- λ 对比实验结论：

| λ | 最终代价 | 训练准确率 | 边界表现 |
|---|---|---|---|
| 0 | 0.345 | 84.7% | 过拟合：边界弯曲缠绕样本 |
| 1 | 0.419 | 82.2% | 适中：平滑 |
| 100 | 0.662 | 72.0% | 欠拟合：接近直线 |

> degree 与 λ 的配合：degree 决定模型复杂度（上限），λ 决定对复杂度的惩罚（约束）。两者搭配可有效管控过拟合。

## 特征设计速查（如何取值、如何判断）

- **从低到高尝试**：degree = 1 → 3 → 6 → 10，够用即可，越简单越不易过拟合
- **判断信号**：训练损失低、验证损失高 → 过拟合（降 degree 或增 λ）；两者都高 → 欠拟合（反向调整）
- **辅助手段**：决策边界可视化、收敛曲线
- **注意**：实际工程需拆分训练/验证/测试三份数据评估，仅看训练集准确率无法判断泛化能力

## 运行方式

```bash
# 运行脚本（自动弹出图形窗口）
python 05_logistic_regression/logistic_regression.py

# 运行 notebook
jupyter notebook 05_logistic_regression/regularized_logistic_regression.ipynb
```
