# 02 · Pandas 知识卡片

> **本章定位**：把「数组」升级成「**带标签的表格**」。本章解决的核心问题是——**像操作 Excel 一样读、看、选、改、算、并一张表**。
>
> **承上启下**：底层依赖 01 的 ndarray；04 起所有数据读取（`read_csv` / `read_csv(header=None)`）都靠本章。

---

## 知识地图

```
两种数据结构
├─ Series     带标签的一维数组（一列）
└─ DataFrame  带行列标签的二维表（整张表）
操作
├─ 读写      read_csv / read_excel / to_csv
├─ 查看      head / tail / len / （04 起还有 info / describe）
├─ 列操作    选列 / 选多列 / 改列名 columns
├─ 行操作    iloc 选行 / 改行标签 index
├─ 单元格    iloc[i, j] 读与写
├─ 筛选      布尔条件过滤
├─ 索引      iloc(位置) / loc(标签) / set_index
├─ 排序      sort_index / sort_values
├─ 映射      apply
├─ 分组      groupby + 聚合
├─ 变形      unstack / pivot（长→宽）
├─ 合并      merge
└─ 画图      plot
```

---

## 核心知识点卡片

### 1. 两种数据结构

- **Series**：一维，**带标签**（索引）的数组——相当于「一列数据 + 每行的名字」。
- **DataFrame**：二维表格，有**行标签**和**列标签**——可以理解成「多个 Series 拼成的字典」。
- **联系**：取 DataFrame 的**一列**得到 Series；取**多列**仍得到 DataFrame。

```
df['字段']      → Series（一列）
df[['A','B']]   → DataFrame（多列）
```

### 2. 读取与保存

```
read_csv(路径)               读 csv → DataFrame
read_csv(路径, header=None)  数据没有表头时用（04 起的 txt 数据都这么读）
read_excel(路径)             读 excel
to_csv(路径, index=False)    写出 csv（index=False 不把行号写进去）
```

### 3. 快速查看数据

```
head(n)     看前 n 行（默认 5）
tail(n)     看后 n 行（默认 5）
len(df)     总行数
```

> 04 起还会用到 `info()`（看每列类型与非空数量）和 `describe()`（看数值列的统计摘要），它们是「上来先摸清数据」的标配动作。

### 4. 行 / 列 / 单元格的取法

```
选列       df['列名'] / df[['列1','列2']]
选行       df.iloc[0]        按位置取第一行
单元格     df.iloc[0, 1]     第 0 行第 1 列；df.iloc[[0,1],[1,2]] 取多行多列
改写单元格 df.iloc[0,1] = 新值
```

### 5. iloc vs loc（**本章最重要的区分**）

| | 含义 | 举例 |
|---|---|---|
| `iloc[]` | 按**整数位置** | `df.iloc[0, 1]` 第 0 行第 1 列 |
| `loc[]` | 按**标签** | `df.loc['a', 'Name']` 行标签 a、列名 Name |

> 位置与标签在「改了行/列名之后」会分道扬镳：`iloc` 永远按第几个算，`loc` 按名字找。

### 6. 修改行列标签

```
df.columns = ['ID','Name','Age','Gender','Score']   # 整体重命名列
df.index   = ['a','b','c','d','e']                   # 重命名行标签
```

> 改完列名后，**后续一律要用新名字访问**，沿用旧名会 `KeyError`。

### 7. 布尔筛选

```
df[df['Age'] > 30]        # 选出 Age 大于 30 的行
```

原理：`df['Age'] > 30` 先产生一列 True/False，再用它当「行掩码」过滤。

### 8. set_index（把某列设为索引）

```
df.set_index('Age', inplace=True, drop=False)
```

- `inplace=True`：**原地修改**，不返回新表。
- `drop=False`：保留该列（否则被移进索引后，列里就没有它了）。

### 9. 排序

```
sort_index()                              按索引排序
sort_values(by='Score', ascending=False)  按某列的值排序，ascending=False 降序
```

### 10. apply（按单元格套自定义函数）

- **是什么**：把函数**逐元素**作用到某列上，返回新的一列。
- **怎么用**

```
df['Grade'] = df['Score'].apply(score_to_grade)   # 把分数映射成等级 A/B/C...
```

### 11. groupby（分组聚合）

- **是什么**：先按某列**分组**，再对每组做聚合（求和/均值/计数…）。
- **注意**：`groupby` 本身返回的是**分组对象**（还没算出结果），必须再挂一个聚合函数才有数值。

```
df.groupby('Gender')                  # 只是分组对象
df.groupby('Gender')['Score'].mean()  # 每组 Score 的均值（这才是结果）
```

### 12. 变形：长格式 → 宽格式

```
unstack()  把层级索引里的一层“摊开”成列
pivot(index, columns, values)  指定行/列/值，把长表转成透视表
```

### 13. merge（合并两张表）

```
pd.merge(左表, 右表, left_on='ID', right_on='ID')   # 按公共键拼接
```

### 14. 快速画图

```
df.plot(x='ID', y='Score', kind='bar')   # 直接用 DataFrame 出图（底层是 matplotlib）
```

---

## 易错点 / 常见坑

| 坑 | 说明 |
|---|---|
| 改了列名还用旧名 | `columns` 重命名后必须用新名访问，否则 `KeyError` |
| 单列 vs 多列返回类型 | `df['c']` 是 Series，`df[['c']]` 才是 DataFrame |
| `iloc` / `loc` 混淆 | 前者按位置、后者按标签；改过标签后两者结果会不同 |
| `groupby` 忘了聚合 | 只 `groupby` 得到的是分组对象，必须接 `.mean()` 之类才是结果 |
| `set_index` 丢了列 | 不写 `drop=False` 时该列会从列中消失 |
| 忘了 `inplace` | 很多方法默认返回新表，不改原表；需要原地改要 `inplace=True` |
| 无表头数据直接读 | 04 起的 txt 数据没有表头，必须 `header=None`，否则第一行被当成列名 |

---

## 与相邻章节的联系

- **向前**：数据结构建立在 01 的 ndarray 之上（`.values` 就能取出裸数组）。
- **向后**：04～09 统一用 `read_csv(..., header=None)` 读数据，用 `iloc[:, :-1]` 取特征、`iloc[:, -1]` 取目标——这是本章「取列」知识的标准用法。
- **向后**：04 的 `head() / info() / describe()`「三步摸底」，是本章「查看数据」在建模流程里的固定环节。
