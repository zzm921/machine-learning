# Machine Learning 学习仓库

吴恩达机器学习课程实践代码：从 NumPy / Pandas / Matplotlib 基础，到线性回归、逻辑回归、神经网络、多分类与多标签分类的完整学习路径。每个模块都保留「数据 → 模型 → 训练 → 可视化」的完整链路，并附课程核心概念速查。

## 目录结构

| 目录 | 内容 | 说明 |
|---|---|---|
| `01_numpy` | numpy_use.py | NumPy 基础用法、ndarray 与向量化计算 |
| `02_pandas` | pandas_use.py + data.csv / data2.csv | Pandas 数据读取、筛选、合并 |
| `03_matplotlib` | matplotlib_use.py | Matplotlib 绘图基础 |
| `04_linear_regression` | linear_regression.py / .ipynb | 线性回归（单特征、多特征、特征缩放、梯度下降） |
| `05_logistic_regression` | logistic_regression.py / .ipynb | 逻辑回归（二分类、决策边界） |
| | regularized_logistic_regression.py / .ipynb | 逻辑回归 + 多项式特征 + L2 正则化 |
| `06_forward_propagation` | forward_propagation.py / .ipynb | 前向传播：单层 dense 的循环实现 vs 矩阵实现 |
| `07_neural_network` | neural_network.py / .ipynb | 神经网络解决 XOR 非线性可分问题 |
| `08_mutil_class` | mutil_class.py | 多分类：手写数字 0~9（softmax / logits 数值稳定性） |
| | multi_label_classification.py | 多标签：一张图同时判断 4 个属性（多个 sigmoid） |
| | multi_class_vs_multi_label.ipynb | 多分类与多标签的对照笔记 |
| | ex3data1.mat | 手写数字数据集，5000 张 20×20 灰度图 |

## 环境依赖

- Python 3.x
- numpy、pandas、matplotlib
- scipy（读取 .mat 数据）、scikit-learn（数据划分）
- tensorflow / keras（神经网络部分）
- jupyter / nbconvert（运行 notebook）

```bash
pip install -r requirements.txt
```

## 模块说明

### 04 线性回归

- 数据：`ex1data1.txt`（单特征：人口 → 利润）、`ex1data2.txt`（多特征：面积 / 卧室数 → 房价）
- 实现：特征标准化、代价函数、批量梯度下降、收敛曲线、学习率对比、预测
- 要点：多特征量纲差异大时必须做特征缩放，否则等高线细长、梯度下降震荡难收敛

### 05 逻辑回归

#### logistic_regression（线性边界）

- 数据：`ex2data1.txt`，两门考试成绩预测录取，正负样本各半
- 模型：`h(x) = sigmoid(θ₀ + θ₁·x₁ + θ₂·x₂)`
- 实现：特征标准化、代价函数、梯度下降、决策边界绘制（缩放空间 → 原始空间反变换）、学习率对比

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

| λ | 最终代价 | 训练准确率 | 边界表现 |
|---|---|---|---|
| 0 | 0.345 | 84.7% | 过拟合：边界弯曲缠绕样本 |
| 1 | 0.419 | 82.2% | 适中：平滑 |
| 100 | 0.662 | 72.0% | 欠拟合：接近直线 |

> degree 决定模型复杂度上限，λ 决定对复杂度的惩罚力度，两者配合才能有效管控过拟合。

### 06 前向传播

- 目标：手写单层全连接（dense）的计算过程，理解「神经元 = 加权和 + 激活」
- 同一件事的两种写法：
  - 循环版：逐个神经元算 `z = w·a_in + b`，再 sigmoid，概念直观但慢
  - 矩阵版：`a_out = sigmoid(a_in · W + b)`，一次矩阵乘法算完整层，是实际框架的做法
- 权重矩阵 W 的每一**列**是一个神经元的权重；输入是行向量，输出也是行向量
- 多层堆叠：把上一层的输出当作下一层的输入，串起来就是完整网络（sequential）

### 07 神经网络（XOR）

- 数据：在 [0,1]² 内均匀采样 1000 个点，两特征是否同侧（异或规则）打标签
- 网络：`2 → 3(sigmoid) → 1(sigmoid)`，二分类输出 1 个神经元，值即「为正类的概率」
- 损失：`binary_crossentropy`（配 sigmoid 输出）；优化器 Adam（自适应学习率，比原始梯度下降快）
- 可视化：左图学习曲线（loss + accuracy），右图决策边界（网格逐点预测概率 + 等值填色）
- 结论：异或线性不可分，单个逻辑回归无法拟合；隐藏层先做特征变换，把原空间「撑开」成线性可分，输出层再做线性划分

### 08 多分类（手写数字 0~9）

- 数据：5000 张 20×20 灰度图展平成 400 维；原始标签 1~10，其中 10 代表数字 0，需 `% 10` 映射为 0~9
- 网络：`400 → 25(relu) → 15(relu) → 10`
- 关键处理：输出层用 **linear**，损失内部 `from_logits=True`；把 softmax 折进损失里做，避免「先 softmax 再算交叉熵」的数值误差
- 数据划分：70% 训练 / 15% 验证 / 15% 测试，按数字类别分层抽样保证各类比例一致
- 可视化：训练 vs 验证的 loss / accuracy 曲线；随机抽一个测试样本展示图像、真实标签、预测标签与 10 类概率

### 08 多标签（一张图 4 个属性）

- 思路：把「识别是几」换成 4 个互不排斥的判断题，从数字本身派生标签
  - 大数字（≥5）、偶数、质数、含闭合环（0/6/8/9）
  - 例：8 → `[1,1,0,1]`，3 → `[0,0,1,0]`，6 → `[1,1,0,1]`
- 网络：`400 → 25(relu) → 15(relu) → 4`，输出层 linear，损失 `BinaryCrossentropy(from_logits=True)`
- 关键区别：
  - 多分类用 **softmax**，各类互斥，概率和为 1
  - 多标签用**多个 sigmoid**，每个标签独立输出概率，可同时为 1
- 指标坑：多标签**不能**直接用 `metrics=['accuracy']`（Keras 会按多分类逻辑用 argmax 比较，结果失真），要用 `BinaryAccuracy` 逐标签判断对错
- 评估补充：逐元素准确率必须和「全零预测」基线对照（4 个标签正例比例约 0.5/0.5/0.4/0.4，全预测 0 也能拿到约 0.55）；再看「每个样本命中几个标签」的分布是否与真实一致
- 正则化手段：EarlyStopping（验证损失连续若干轮不降就停，并回滚到最优权重）

## 吴恩达课程核心概念速查

### 学习范式

- **监督学习**：数据带标签，学从 x 到 y 的映射；房价预测（回归）、肿瘤判定（分类）
- **无监督学习**：数据无标签，找内在结构；聚类、降维
- **回归 vs 分类**：输出连续值 → 回归；输出离散类别 → 分类

### 三大件：假设、代价、优化

```
假设函数   h(x) = 模型对单个样本的预测
代价函数   J(θ) = 所有样本损失的平均   # 衡量「预测得有多差」
优化目标   min J(θ)                    # 找一组让代价最小的参数

梯度下降   θ := θ - α * ∂J(θ)/∂θ       # 沿最陡下坡方向走一小步
           α 太小 → 收敛慢；α 太大 → 震荡甚至发散
           批量梯度下降：每步用全部样本算梯度，稳定但慢
```

- 代价函数必须是凸的才好梯度下降；线性回归用平方误差，逻辑回归用交叉熵（而不是平方误差，否则非凸）

### 特征处理

- **特征缩放 / 标准化**：把各特征拉到相近量纲（如均值 0、方差 1），让代价函数等高线更接近圆形，梯度下降更快更稳
- **多项式特征映射**：原始线性不可分时，用高次项构造新特征，在更高维空间里变线性可分；代价是特征数爆炸、易过拟合

### 过拟合与正则化

| 现象 | 训练损失 | 验证损失 | 对策 |
|---|---|---|---|
| 欠拟合 | 高 | 高 | 加特征 / 加模型复杂度 / 减小 λ |
| 刚好 | 低 | 低 | —— |
| 过拟合 | 低 | 高 | 加数据 / 减特征 / 增大 λ / 早停 |

```
L2 正则化   J(θ) = 原代价 + λ/(2m) * Σ θ_j²   # 不惩罚 θ₀（偏置）
权重衰减    梯度更新时 θ_j 额外乘一个略小于 1 的系数，参数被持续压小
λ 的作用    λ 越大约束越强；λ 过大会让参数趋近 0，退化为欠拟合
```

- **数据划分**：训练集调参、验证集选模型 / 早停、测试集只看最终结果；只看训练集准确率无法判断泛化能力

### 逻辑回归

```
h(x) = sigmoid(θᵀx) = 1 / (1 + e^(-θᵀx))     # 输出「为正类的概率」
判定   h(x) ≥ 0.5 → 预测 1，否则预测 0       # 即 θᵀx ≥ 0 的边界
代价   J(θ) = -(1/m) Σ [ y·log(h) + (1-y)·log(1-h) ]
```

- **决策边界**：θᵀx = 0 的曲面，是模型的属性而非数据的属性；特征映射后边界可以是曲线
- sigmoid 导数 `σ'(z) = σ(z)(1-σ(z))`，这是梯度能写成漂亮形式的来源

### 神经网络

```
单神经元   a = g(w·a_in + b)         # g 是激活函数
单层       a_out = g(a_in · W + b)   # W 每列对应一个神经元
多层堆叠   前向传播 = 逐层做上面的运算，前一层的输出是后一层的输入
训练循环   前向传播 → 算损失 → 反向传播求梯度 → 更新 W 和 b
```

- **隐藏层的意义**：自动学习新的特征表示，替代手工多项式特征映射；层数 / 神经元越多表达力越强
- **激活函数选择**：

| 激活 | 表达式 | 用在哪 | 注意 |
|---|---|---|---|
| linear | z | 回归输出层 / logits 输出层 | 不引入非线性 |
| sigmoid | 1/(1+e^-z) | 二分类输出层、小网络的隐藏层 | 两端梯度趋 0，深网络中易梯度消失 |
| ReLU | max(0, z) | 隐藏层默认选择 | 计算快、缓解梯度消失；可能出现「死亡」神经元 |

- **为什么需要非线性激活**：全是线性层叠加等价于一个线性层，加深网络没有任何意义，非线性才能拟合复杂边界

### 多分类与多标签

```
多分类（互斥）  输出层 softmax：p_k = e^(z_k) / Σ e^(z_j)，概率和为 1，取 argmax
               损失用交叉熵；类别为整数标签时用 sparse 版本
多标签（可共存）输出层 n 个 sigmoid，每个标签独立给概率，可同时 ≥ 0.5
               损失用 BinaryCrossentropy，逐标签算
数值稳定性      输出层用 linear，把 softmax / sigmoid 折进损失函数内部（from_logits=True）
               避免「先激活再算损失」的中间舍入误差
```

### 损失 / 指标速查

| 任务 | 输出层激活 | 损失函数 | 指标 |
|---|---|---|---|
| 回归 | linear | MSE | MAE / MSE |
| 二分类 | linear（logits） | BinaryCrossentropy(from_logits=True) | BinaryAccuracy |
| 多分类（互斥） | linear（logits） | SparseCategoricalCrossentropy(from_logits=True) | Accuracy |
| 多标签（共存） | linear（logits） | BinaryCrossentropy(from_logits=True) | BinaryAccuracy（逐元素） |

> 指标只影响打印 / 监控，不参与训练；损失才决定梯度往哪走。激活函数与损失函数必须配套，错配会导致梯度错误、loss 变 nan 或指标失真。

### 常用训练技巧

- **固定随机种子**：采样、划分、初始化都可复现
- **标准化输入**：神经网络同样受益于特征缩放
- **学习曲线**：训练 vs 验证两条线，判断欠拟合 / 过拟合 / 该早停
- **早停 EarlyStopping**：验证损失不再下降就停，并回滚到最优权重，是最省事的防过拟合手段
- **分层抽样划分**：类别不均衡时保证各子集类别比例一致

## 运行方式

```bash
# 运行脚本（自动弹出图形窗口）
python 04_linear_regression/linear_regression.py
python 05_logistic_regression/logistic_regression.py
python 06_forward_propagation/forward_propagation.py
python 07_neural_network/neural_network.py
python 08_mutil_class/mutil_class.py
python 08_mutil_class/multi_label_classification.py

# 运行 notebook
jupyter notebook 05_logistic_regression/regularized_logistic_regression.ipynb
```

> 08 目录需要 `scipy` 与 `scikit-learn`；07、08 需要 `tensorflow`。脚本内的 `plt.show()` 会依次弹出多个图形窗口，逐个关闭即可。