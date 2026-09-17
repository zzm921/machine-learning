
"""
前向传播  实现 
forward_propagation   
dense layer 实现 三个神经元的神经网络层计算
设置 w1 = [1,2], w2 = [-3,4], w3 = [5,-6]   
b1=1 , b2=1, b3=3

需要计算
a1= sigmoid(w1*a_in+b1)
a2= sigmoid(w2*a_in+b2)
a3= sigmoid(w3*a_in+b3)

特征转换成矩阵 2*3
 W向量
[
[1,-3,5],
[2,4,-6]
]
a_in 是一个1*2 矩阵
计算 a_in*W  根据矩阵乘法  得到一个[a_in*w1,a_in*w2,a_in*w3]的矩阵，再加上 偏置B
a_out=sigmoid(np.matmul(a_in,W)+B) 
即
a_out=[a_out[0],a_out[1],a_out[2]]
"""

import numpy as np

W=np.array([
[1,-3,5],
[2,4,-6]
])

# 输入 
a_in=np.array([1,2])

# 偏置
b=np.array([1,1,3])

def sigmoid(z):
    return 1/(1+np.exp(-z))


# 循环计算单个神经元再拼接
def denseLoop(a_in , W , b):
    a_out=np.zeros(W.shape[1])
    for i in range(W.shape[1]):
        z=np.dot(a_in,W[:,i])+b[i]
        a_out[i]=sigmoid(z)
    return a_out

#向量计算 极大提高了计算效率
def dense(a_in , W , B):
    z = np.matmul(a_in, W) + B     
    a_out=sigmoid(z)
    return a_out



"""
神经网络实现
如果有多层神经网络，那么就需要将每一层的输出作为下一层的输入
def sequential(x ):
   a1=dense(x,W1,b)
   a2=dense(a1,W2,b2)
   a3=dense(a2,W3,b3)
   fx=a3
   return fx
"""

