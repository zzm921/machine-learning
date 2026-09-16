from matplotlib import pyplot
from pathlib import Path

import pandas as pd

"""
pandas 是一个强大的数据分析工具，提供了高效的数据结构和数据分析功能。
数据类型 data type
- 序列(Series) 是一种一维数组，用于存储一维数据，每个元素都有各自的标签。可以当做一个带标签的元素组成的numpy数组。标签可以是字符或者数字
- 数据框(DataFrame) 是一种二维的，表格型的数据结构。pandas 的dataframe可以存储许多不同类型的数据，并且每个周都有标签，可以把它当做一个series字典。
"""

# 文件操作
# 从文件中导入数据
# read_csv():从csv文件中导入数据，读取csv文件为dataframe
df_csv = pd.read_csv(Path(__file__).parent / './data.csv')
print('从csv文件中导入数据 read_csv() :',df_csv)

# 读取另一个数据集，用于合并
df_csv2 = pd.read_csv(Path(__file__).parent / './data2.csv')
print('从csv文件中导入数据 read_csv() :',df_csv2)

# read_excel():从excel文件中导入数据，读取excel文件为dataframe
# df_excel = pd.read_excel(Path(__file__).parent / './data.xlsx')
# print('从excel文件中导入数据 read_excel() :',df_excel)

#head():查看数据框的前n行，默认前5行
print('查看数据框的前n行 head() :',df_csv.head(3))

# tail():查看数据框的后n行，默认后5行
print('查看数据框的后n行 tail() :',df_csv.tail(3))


#保存数据
#to_csv():将数据框保存为csv文件
# df_csv.to_csv(Path(__file__).parent / './datasave.csv', index=False)


# 数据操作
# 列操作 column operations
# 获取第一列 返回的是一个series
print('获取第一列 :',df_csv['id'])

# 获取多列 
print('获取多列 :',df_csv[['id','name']])


#改变列标签
df_csv.columns = ['ID','Name','Age','Gender',"Score"]
print('改变列标签 :',df_csv)

# 行操作 row operations
# 获取第一行 返回的是一个series
print('获取第一行 :',df_csv.iloc[0])

# 获取多行
print('获取多行 :',df_csv.iloc[[0,1]])

# 改变行标签
df_csv.index = ['a','b','c','d','e']
print('改变行标签 :',df_csv)

# 总行数
print('总行数 :',len(df_csv))


# 单元格操作 cell operations
# 获取单个单元格 
print('获取单个单元格 :',df_csv.iloc[0,1])

# 获取多行多列
print('获取多行多列 :',df_csv.iloc[[0,1],[1,2]])

# 改变单个单元格的值
df_csv.iloc[0,1] = 'Alice'
print('改变单个单元格的值 :',df_csv)


# 数据筛选 filter operations
# 筛选出年龄大于30的行
print('筛选出年龄大于30的行 :',df_csv[df_csv['Age']>30])


# 索引 index
# df.iloc[] 基于整数位置的索引
# df.loc[] 基于标签的索引
print('基于整数位置的索引 iloc[] :',df_csv.iloc[0,1])
print('基于标签的索引 loc[] :',df_csv.loc['a','Name'])

# 设置新的索引，drop=False 保留 Age 列以便后续统计
df_csv.set_index('Age',inplace=True,drop=False)
print('设置新的索引 :',df_csv.head())



# 排序
# sort_index():根据索引排序
print('根据索引排序 sort_index() :',df_csv.sort_index())

# sort_values():根据值排序
print('根据值排序 sort_values() :',df_csv.sort_values(by='Score',ascending=False))

# 对数据集应用函数 apply()
# 对Score列应用函数，将分数转换为等级
def score_to_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df_csv['Grade'] = df_csv['Score'].apply(score_to_grade)
print('对Score列应用函数，将分数转换为等级 :',df_csv)


# 操作数据集的结构
# groupby():根据一个或多个列对数据进行分组
print('根据性别分组 groupby() :',df_csv.groupby('Gender'))

# 对分组数据进行聚合操作
print('对分组数据进行聚合操作 :',df_csv.groupby('Gender')['Score'].mean())

# max min  mean 等统计函数
print('最大年龄 max() :',df_csv['Age'].max())
print('最小年龄 min() :',df_csv['Age'].min())
print('平均年龄 mean() :',df_csv['Age'].mean())
# unstack():将数据框从长格式转换为宽格式
print('将数据框从长格式转换为宽格式 unstack() :',df_csv.unstack())

# pivot():旋转数据框，将长格式数据转换为宽格式
print('将数据框从长格式转换为宽格式 pivot() :',df_csv.pivot(index='ID',columns='Name',values='Score'))

# 合并数据
# merge():合并数据框，默认按公共列合并
print('合并数据框 merge() :',pd.merge(df_csv,df_csv2,left_on='ID',right_on='ID'))


# 快速画图 plot()
df_csv.plot(x='ID',y='Score',kind='bar')
pyplot.show()