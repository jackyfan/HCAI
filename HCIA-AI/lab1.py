"""
使用scikit-learn 实现线性回归算法
"""
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import  numpy as np

#构建房价数据集并可视化
# x是房屋面积，作为特征
x = np.array([121,125,131,141,152,161]).reshape(-1,1)
# y是房屋价格，作为标签
y = np.array([300,350,425,405,496,517])

plt.scatter(x,y)

plt.xlabel("area")
plt.ylabel("price")

#创建线性回归模型
lr = LinearRegression()
#模型在数据集上训练
lr.fit(x,y)
#存储模型的斜率
w = lr.coef_
#存储模型的截距
b = lr.intercept_
print("斜率:",w,"截距:",b)
plt.plot([x[0],x[-1]],[w*x[0]+b,w*x[-1]+b])
#plt.show()

test_x = np.array([[130]])
price = lr.predict(test_x)
print("预测的房价:",price)
