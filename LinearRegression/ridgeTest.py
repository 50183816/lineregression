# _*_ codig utf8 _*_

import numpy as np
import pandas as pd
import matplotlib as mpl
import  matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import Ridge
import sklearn.linear_model as lm
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
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
    ('lasso',lm.Ridge(fit_intercept=True,alpha=1))
   # ('lr',lm.LinearRegression(fit_intercept=True,normalize=True))
])
algo.fit(x1,y1)

algo2 = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('lasso',lm.RidgeCV(fit_intercept=True,alphas=np.logspace(-6, 6, 13),cv=2))
])
algo2.fit(x1,y1)

algo3 = Pipeline(steps=[
    ('scaler',PolynomialFeatures(degree=2)),
   # ('en',lm.ElasticNet(fit_intercept=True,alpha=0.1,l1_ratio=0.5,random_state=23))
    ('sgd',lm.SGDRegressor())
])
algo3.fit(x1,y1)

#print(algo.coef_)
# 8. 模型效果评估
# a. 查看模型训练好的相关参数
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.steps[-1][-1].coef_))
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo2.steps[-1][-1].coef_))
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo3.steps[-1][-1].coef_))
print("截距项值:{}".format(algo.steps[-1][-1].intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(train_x, train_y)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
#test_x = scalar.transform(test_x)
print("模型在测试数据上的效果(R2)：{}".format(algo.score(test_x, test_y)))

#5 predict the test data
#predict = np.mat(test_x)*theta
predict = algo.predict(test_x)
predict2 = algo2.predict(test_x)
predict3 = algo3.predict(test_x)
#print(predict)
#6 figures
t = np.arange(len(test_x))
fig,axes = plt.subplots(3,1)
#plt.figure(facecolor='w')
axes[0].plot(t, test_y, 'r-', label=u'真实值')
axes[0].plot(t, predict, 'b-', label=u'预测值')
axes[0].legend(loc='lower right')
axes[0].set_title('Ridge回归预测,score=%f'%(algo.score(test_x,test_y)))

print('R2:%f'%r2_score(test_y,predict))
print('R2:%f'%r2_score(test_y,predict2))
print('R2:%f'%r2_score(test_y,predict3))
axes[1].legend(loc='lower right')
axes[1].set_title('RidgeCV回归预测,score=%f'%algo2.score(test_x,test_y))
axes[1].plot(t, test_y, 'r-', label=u'真实值')
axes[1].plot(t, predict2, 'g-', label=u'RidgeCV')


axes[2].legend(loc='lower right')
axes[2].set_title('ElasticNet回归预测,score=%f'%algo3.score(test_x,test_y))
axes[2].plot(t, test_y, 'r-', label=u'真实值')
axes[2].plot(t, predict3, 'g-', label=u'ElasticNet')
# plt.legend(loc='lower right')
# plt.title('Ridge回归预测')
#plt.show()
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import auc
# from sklearn.metrics import r2_score
# prec = auc(list(test_y),list(predict))
# print(prec)