# _*_ codig utf8 _*_

import numpy as np
import pandas as pd
import matplotlib as mpl
import  matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
mpl.rcParams['font.sans-serif'] = [u'simHei']

#1. Load data
data= pd.read_csv('../datas/household_power_consumption_1000.txt',sep=';')
#data.info()
#print(data.head())

#2 Get property attribute and target antribute x, y
x = data.iloc[:,2:4].astype(np.float)
y= data.iloc[:,5].astype(np.float)
#print(y[:5])
#scalar = StandardScaler()
#print(x)
#3 split test dataset and training  dataset
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=22)
#train_x = scalar.fit_transform(train_x,train_y)
#print(train_x)
#print(train_x.head())
#print(train_y.shape)
#4. build the model
x1=np.mat(train_x)
y1=np.mat(train_y).reshape(-1,1)
#print(y1.shape)
#4 Calulate the Theta values
#theta = (x1.T*x1).I*x1.T*y1
#print(theta)

algo = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('lasso',Lasso(fit_intercept=True,alpha=2))
])
algo.fit(x1,y1)

#print(algo.coef_)
# 8. 模型效果评估
# a. 查看模型训练好的相关参数
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['lasso'].coef_))
print("截距项值:{}".format(algo.get_params()['lasso'].intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(train_x, train_y)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
#test_x = scalar.transform(test_x)
print("模型在测试数据上的效果(R2)：{}".format(algo.score(test_x, test_y)))

#5 predict the test data
#predict = np.mat(test_x)*theta
predict = algo.predict(test_x)
#print(predict)
#6 figures
t = np.arange(len(test_x))
plt.figure(facecolor='w')
plt.plot(t, test_y, 'r-', label=u'真实值')
plt.plot(t, predict, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.show()
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn.metrics import r2_score
prec = auc(list(test_y),list(predict))
print(prec)