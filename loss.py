import numpy as np

def mean_squared_error(y, t):
 """均方误差"""
 return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
 """交叉熵误差"""
 delta = 1e-7
 return -np.sum(t * np.log(y + delta))

# 设“2”为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 例1：“2”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
MSE =mean_squared_error(np.array(y), np.array(t))
print("均方差异：",MSE)

CEE  = cross_entropy_error(np.array(y), np.array(t))
print("交叉熵误差：",CEE)
# 例2：“7”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
MSE = mean_squared_error(np.array(y), np.array(t))
print("均方差异：",MSE)

CEE  = cross_entropy_error(np.array(y), np.array(t))
print("交叉熵误差：",CEE)