import matplotlib.pyplot as plt
import numpy as np
# 生成0到2之间的100个均匀分布的点
x = np.linspace(0, 2, 100)

# 绘制折线图 plot 参数详情
# plot(x, y, color='red', linestyle='--', linewidth=2, marker='o', markersize=8, label='linear')
# x:  x 轴上的点
# y:  y 轴上的点
# color:  线的颜色，默认值为 'b'（蓝色）
# linestyle:  线的样式，默认值为 '-'（实线）
# linewidth:  线的宽度，默认值为 1.0
# marker:  数据点的标记样式，默认值为 'None'（无标记）
# markersize:  数据点的标记大小，默认值为 6.0
# label:  线的标签，用于图例
plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

# 添加标签
plt.xlabel('x label')
plt.ylabel('y label')


# 添加标题
plt.title("Simple Plot")

# 添加图例
plt.legend()
# 显示图片
plt.show()

