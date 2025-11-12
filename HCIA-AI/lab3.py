from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# X：每一项表示租金和面积
# y：表示是否租赁该房间（0：不租，1：租）
X=[[2200,15],[2750,20],[5000,40],[4000,20],[3300,20],[2000,10],[2500,12],[12000,80],
   [2880,10],[2300,15],[1500,10],[3000,8],[2000,14],[2000,10],[2150,8],[3400,20],
   [5000,20],[4000,10],[3300,15],[2000,12],[2500,14],[10000,100],[3150,10],
   [2950,15],[1500,5],[3000,18],[8000,12],[2220,14],[6000,100],[3050,10]
  ]

y=[1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,1,0]

scaler = StandardScaler()
#标准化数据，保证每个维度的特征数据方差为1，均值为0，这样预测结果不会被某些维度过大的特征值主导
X_train = scaler.fit_transform(X)
lr = LogisticRegression()
lr.fit(X_train,y)
testX = [[2000,8]]
X_test = scaler.transform(testX)
print('待预测的值：',X_test)
#预测结果
result = lr.predict(X_test)
print('预测结果：',result)
#预测概率
prob = lr.predict_proba(X_test)
print('预测概率：',prob)