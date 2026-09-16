"""
01_numpy.numpyUse 的 Docstring
numpy 是一个开源的 Python 库，主要用于进行数值计算。它提供了一个高性能的多维数组对象，以及用于操作这些数组的工具。numpy 是许多其他科学计算库的基础，如 pandas、matplotlib、scikit-learn 等。
numpy 软件包核心是ndarray，它是一个多维数组对象，提供了对数组的高效操作。
ndarray 是 numpy 中最基本的数据结构，它在内存中以连续的块存储，因此可以利用 CPU 的向量化操作进行快速计算。


为什么 NumPy 很快？
向量化描述了代码中没有任何显式循环、索引等——这些事情当然是在“幕后”以经过优化的、预编译的 C 代码中进行的。向量化代码有许多优点，其中包括：
向量化代码更简洁，更易于阅读
更少的代码行通常意味着更少的错误
代码更接近标准的 数学符号（这通常使编写数学构造更容易且更准确）
向量化产生了更“Pythonic”的代码。如果没有向量化，我们的代码将充斥着低效且难以阅读的 for 循环。

"""



# 导入 numpy 库
import numpy as np

# 创建一个数组
arr = np.array([1, 2, 3])
print('创建一个数组arr :',arr)

# 将序列转化为高维数组
arr_2d = np.array([(1, 2, 3), (4, 5, 6)])
print('将序列转化为高维数组arr_2d :',arr_2d)

# 指定类型创建数组
arr_float = np.array([1.0, 2.0, 3.0],dtype=np.float32)
print('指定类型创建数组arr_float :',arr_float)

## 创建全0 数组
zero_arr = np.zeros((2, 3))
print(zero_arr)


## 创建全部是1的数组
one_arr = np.ones((2, 3))
print('创建全部是1的数组one_arr :',one_arr)

# 创建空数组
empty_arr = np.empty((2, 3))
print('创建空数组empty_arr :',empty_arr)    

# 创建指定范围的数组
range_arr = np.arange(4, 10)
print('创建指定范围的数组range_arr :',range_arr)

# 创建等间隔数组
linspace_arr = np.linspace(0, 1, 5)
print('创建等间隔数组linspace_arr :',linspace_arr)




# 访问数组

# indexing 索引
print('索引arr[0] :',arr[0])
print('索引arr[1:3] :',arr[1:3])

# slicing 切片
print('切片arr[1:] :',arr[1:])
print('切片arr[:2] :',arr[:2])
print('切片arr[1:3] :',arr[1:3])

# interation 迭代
# 遍历数组的每个元素
for i in arr:
    print(i)
# 遍历数组的每个元素的索引
for index, value in np.ndenumerate(arr):
    print(index, value)


# 数组属性
#数组维度
print('数组维度arr.ndim :',arr.ndim)

# 数组的形状 数组的形状是一个元组，元组的每个元素表示数组在该维度上的元素数量。
print('数组的形状arr.shape :',arr.shape)

#数组总元素数量
print('数组总元素数量arr.size :',arr.size)

# 数组类型
print('数组类型arr.dtype :',arr.dtype)

# 基本操作 
# ndarray 和一个数组运算 
# + - * / :将没一个数组和数字进行运算
# ** n :将数组的每个元素进行n次方
arr_add = arr + 10
print('ndarray 和一个数组运算arr_add :',arr_add)

arr_sub = arr - 10
print('ndarray 和一个数组运算arr_sub :',arr_sub)

arr_mul = arr * 10
print('ndarray 和一个数组运算arr_mul :',arr_mul)

arr_div = arr / 10
print('ndarray 和一个数组运算arr_div :',arr_div)

arr_pow = arr ** 2
print('ndarray 和一个数组运算arr_pow :',arr_pow)

# 两个 dbarray 运算
# *:按照元素位置进行乘法运算
# @:矩阵乘法 同np.dot() ,求向量点积
arr_1 = np.array([[1, 2], [3, 4]])
arr_2 = np.array([[5, 6], [7, 8]])
arr_mul_2 = arr_1 * arr_2
print('两个 dbarray 运算arr_mul_2 :',arr_mul_2)

arr_dot = arr_1 @ arr_2
print('两个 dbarray 运算arr_dot :',arr_dot)


# 统计函数 
#  sum():计算数组所有元素的总和
print('统计函数 sum() :',arr.sum())

# mean():计算数组所有元素的平均值
print('统计函数 mean() :',arr.mean())

# max():计算数组所有元素的最大值
print('统计函数 max() :',arr.max())

# min():计算数组所有元素的最小值
print('统计函数 min() :',arr.min())

# std():计算数组所有元素的标准差
print('统计函数 std() :',arr.std())

# var():计算数组所有元素的方差
print('统计函数 var() :',arr.var())

# 全局函数 unviveral function
# 广播机制
# 广播机制是 numpy 中一个非常重要的概念，它允许我们在不同形状的数组之间进行操作。
# 当我们对两个数组进行操作时，numpy 会自动将它们的形状调整为相同的形状，然后进行操作。
# 这就好像是 numpy 自动将较小的数组复制了多次，以匹配较大的数组的形状。
# 例如，当我们将一个标量和一个数组进行加法运算时，numpy 会自动将标量复制成和数组形状相同的数组，然后进行加法运算。
# 这就是广播机制的工作原理。
sin_arr = np.sin(arr)
print('全局函数 unviveral function sin_arr :',sin_arr)

# exp():计算数组所有元素的指数
exp_arr = np.exp(arr)
print('全局函数 unviveral function exp_arr :',exp_arr)



# 操作形状
# reshape():改变数组的形状
arr_reshaped = arr.reshape((3, 1))
print('操作形状 reshape() :',arr_reshaped)

# flatten():将数组转换为一维数组
arr_flatten = arr.flatten()
print('操作形状 flatten() :',arr_flatten)


# resize():改变数组的形状，填充缺失元素
resized_arr = np.resize(arr, (2, 2))
print('操作形状 resize() :',resized_arr)

# transpose():转置数组
arr_transposed = arr_2d.transpose()
print('操作形状 transpose() :',arr_transposed)


# 拼接数组
# vstack():垂直拼接数组
arr_vstack = np.vstack((arr, arr))
print('拼接数组 vstack() :',arr_vstack)

# hstack():水平拼接数组
arr_hstack = np.hstack((arr, arr))
print('拼接数组 hstack() :',arr_hstack)

# 拆分数组
# vsplit():垂直拆分数组
arr_vsplit = np.vsplit(arr_vstack, 2)
print('拆分数组 vsplit() :',arr_vsplit)

# hsplit():水平拆分数组
arr_hsplit = np.hsplit(arr_hstack, 2)
print('拆分数组 hsplit() :',arr_hsplit)

# 拷贝和视图
# 拷贝:创建一个新的数组，新数组的元素和原数组的元素相同，但是新数组和原数组的内存地址不同。
# 视图:创建一个新的数组，新数组的元素和原数组的元素相同，但是新数组和原数组的内存地址相同。
# 拷贝数组
arr_copy = arr.copy()
print('拷贝数组 copy() :',arr_copy)

# 视图数组
arr_view = arr.view()
print('视图数组 view() :',arr_view)



# 线性代数
# numpy可以实现大量矩阵操作
#transpose():转置数组
arr_transposed = arr_2d.transpose()
print('操作形状 transpose() :',arr_transposed)

# dot():矩阵乘法 同np.dot() ,求向量点积
arr_dot = arr_2d.dot(arr_transposed)
print('线性代数 dot() :',arr_dot)

#eye():创建一个单位矩阵
eye_arr = np.eye(3)
print('线性代数 eye() :',eye_arr)

# trace():计算矩阵的迹
print('线性代数 trace() :',arr_2d.trace())


# 直方图
# histogram():计算数组的直方图
hist, bins = np.histogram(arr, bins=5)
print('直方图 histogram() :',hist)
print('直方图 bins :',bins)