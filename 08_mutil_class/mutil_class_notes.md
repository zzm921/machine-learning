# 08 · 多分类与多标签 知识卡片

> **本章定位**：把 07 的「二选一」扩展成两种更贴近现实的任务：**多分类**（互斥、选一个）与**多标签**（可共存、选多个）。解决的核心问题是——**输出层与损失函数怎么配，才不会算错**。
>
> **承上启下**：承接 07 的二分类；向后 09 把这些网络用于真实图像。

---

## 知识地图

```
多分类（互斥）                        多标签（可共存）
├─ softmax：概率和=1，取 argmax      ├─ 输出层：多个独立 sigmoid
├─ 标签：整数(sparse)/one-hot         ├─ 损失：BinaryCrossentropy
├─ 输出层 linear + from_logits       ├─ 指标：BinaryAccuracy（逐标签）
├─ 数据：5000×400，标签 1~10 →0~9     ├─ 标签从数字派生，Y 形 (m,4)
├─ 划分 70/15/15 + 分层抽样            ├─ 基线：全零预测的逐元素准确率
├─ 网络 400→25(relu)→15(relu)→10     ├─ 命中标签数分布对比
└─ 单样本预测：logits→softmax→argmax └─ EarlyStopping 防过拟合
```

---

## 核心知识点卡片

### 1. 多分类 vs 多标签（本章总纲）

```
多分类：类别互斥，一个样本只属于一类 → softmax，各类概率之和 = 1
多标签：标签共存，一个样本可同时命中多个 → 多个 sigmoid，每个独立给概率
```

### 2. softmax（多分类输出层）

```
p_k = e^(z_k) / Σ_j e^(z_j)     所有 p_k ∈ (0,1) 且 Σp_k = 1
预测类 = argmax(p)
```

- **作用**：把 10 个原始分数转成互斥的概率分布。

### 3. 数值稳定性：`from_logits=True`（本章重点）

- **问题**：手工「先算指数 → 求和 → 相除」分步做 softmax，会引入舍入误差，与整体计算不一致。
- **做法**：输出层用 **linear**（输出原始分数 logits），把 softmax **折进损失函数内部**做。

```
输出层：Dense(10, activation='linear')
损失  ：SparseCategoricalCrossentropy(from_logits=True)
预测时：需自己对 logits 手动 softmax 才能拿到概率
```

### 4. 标签形式：sparse vs one-hot

```
标签是单个整数(0~9)      → SparseCategoricalCrossentropy（sparse 指“稀疏、非 one-hot”）
标签已 one-hot 编码      → CategoricalCrossentropy
```

### 5. 数据准备（手写数字 0~9）

```
来源：ex3data1.mat，X 形 (5000, 400)，每行是一张 20×20 灰度图展平
标签：原始 1~10，其中 10 代表 0 → 用 % 10 映射成 0~9
图像还原：.mat 里的图按列优先展平，reshape(20,20) 后要转置才正
```

### 6. 数据划分与分层抽样

```
训练 / 验证 / 测试 = 70% / 15% / 15%
stratify=标签   → 分层抽样，保证各子集里每个数字的比例一致
验证集用于监控过拟合；测试集只在最后评估用一次
```

### 7. 网络与训练

```
结构：400 → 25(relu) → 15(relu) → 10(linear)
损失：SparseCategoricalCrossentropy(from_logits=True)
曲线：训练 vs 验证的 loss / accuracy，判断欠拟合/过拟合
```

### 8. 单样本预测全流程

```
新样本 → 前向得到 logits（10 个数）→ 手动 softmax 得概率 → argmax 得预测类
同时展示真实标签，直观核对
```

---

## 多标签部分

### 9. 从数字派生多标签（构造多标签数据）

```
label1 大数字   数字 ≥ 5
label2 偶数     数字 % 2 == 0
label3 质数     数字 ∈ {2,3,5,7}
label4 含闭合环 数字 ∈ {0,6,8,9}
例：8 → [1,1,0,1]；3 → [0,0,1,0]；5 → [1,0,1,0]
标签矩阵 Y 形 (m, 4)，可同时为 1
```

### 10. 多标签的输出层与损失

```
输出层：Dense(4, activation='linear')     # 4 个标签
损失  ：BinaryCrossentropy(from_logits=True)  # 逐标签二分类
预测  ：先 sigmoid 得概率，再按 0.5 判定每个标签
```

### 11. 指标坑（本章最易错）

- **不能用** `metrics=['accuracy']`：Keras 会按**多分类**逻辑用 argmax 比较，结果与标签对错无关（实测退化成约 0.5 的随机数）。
- **要用** `BinaryAccuracy`：**逐个标签**判断对错。

### 12. 逐元素准确率必须对照基线

```
4 个标签正例比例约 0.5 / 0.5 / 0.4 / 0.4
什么都不学、全预测 0，也能拿到约 0.55 的逐元素准确率
→ 所以必须把模型准确率和「全零基线」对比，才知道是否真的学到东西
```

### 13. 命中标签数分布

```
对比「每个样本命中几个标签」的真实分布 vs 预测分布
平均值接近 + 分布形状接近，才说明模型抓住了标签结构
```

### 14. EarlyStopping（防过拟合）

```
监控 val_loss，连续 patience 轮不降就停，并回滚到最优权重
是最省事、最常用的防过拟合手段
```

### 15. 多标签划分的分层细节

- `stratify` 只能传**一维**标签；多标签的 `Y` 是二维 (m,4)，所以改用**数字本身**（digit）来分层。

---

## 易错点 / 常见坑

| 坑 | 说明 |
|---|---|
| 多标签用 `accuracy` | Keras 当多分类处理，结果失真，要用 `BinaryAccuracy` |
| softmax 分步计算 | 有舍入误差；用 `from_logits=True` 折进损失 |
| 逐元素准确率不设基线 | 不与「全零预测」对比就看不出真实效果 |
| softmax + sparse 错配 | 整数标签配 sparse；one-hot 配 categorical |
| `.mat` 图像方向反了 | 列优先展平，reshape 后要转置 |
| stratify 传二维标签 | 会报错；多标签改造用一维 digit 分层 |
| 输出层直接 softmax/sigmoid | 与 `from_logits=True` 联用时应把输出层设为 linear |

---

## 与相邻章节的联系

- **向前**：07 的「sigmoid + 二分类交叉熵」在本章扩展为 softmax（多类）与多 sigmoid（多标签）。
- **向后**：09 的 CNN 分类头就是「卷积提特征 + 这里的 softmax 多分类」，`sparse_categorical_crossentropy` 用法完全一致。
- **统一规律**：**输出层激活**（linear/logits）与**损失函数**（sparse / binary / categorical）必须成套匹配——这条线索从 07 贯穿到 09。
