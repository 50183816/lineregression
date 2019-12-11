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
import random
import math
mpl.rcParams['font.sans-serif'] = [u'simHei']
#mpl.rcParams[]
#1. Load data
data= pd.read_csv('../datas/household_power_consumption_1000.txt',sep=';')
#data.info()
#print(data.head())

#2 Get property attribute and target antribute x, y
x = data.iloc[:,2:4].astype(np.float)
y= data.iloc[:,5].astype(np.float)
matx = np.mat(x)
maty = np.mat(y)
print(matx.shape)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=22)

def predict(theta,x,y,intercept=0):
    '''
    :param theta:
    :param x:
    :param y:
    :param intercept:
    :return:
    '''
    sum = 0
    n = len(x)
    for i in range(n):
        sum += theta[i]*x[i]
    sum += intercept
    return sum

def fit(X,Y,alpha=0.0001,fit_intercept=False,tol=1e-6,total_iter=2000):
    m,n = X.shape
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)
    theta = np.zeros(n)
    intercept = 0
    jtheta = 30000
    jtheta_change = jtheta
    iter = 0
    diff = np.zeros(shape=[m])
    while jtheta_change > tol and iter < total_iter:
        #1.计算梯度
        gd = 0
        #计算预测值和实际值之间的差值
        for i in range(m):
            y_predict = predict(theta,X[i],Y[i],intercept)
            y_true = Y[i]
            diff[i] = y_true - y_predict
        #计算GD
        for j in range(n):
            for k in range(m):
                gd += X[k][j] * diff[k]
        #2.计算下一个theta
        theta = theta + alpha * gd
        if fit_intercept:
            intercept = intercept + alpha * np.sum(diff)
        #3 利用新的theta计算新的损失函数值
        tmp = 0
        for i in range(m):
            y_true = Y[i]
            y_predict = predict(theta,X[i],Y[i],intercept)
            tmp +=(y_true - y_predict) ** 2
        tmp /= 2.0

        #计算两次损失函数的值
        jtheta_change = np.abs(tmp - jtheta)
        #print(jtheta_change)
        iter += 1
        jtheta = tmp

    return theta,intercept,iter

def predic_X(X,theta,intercept):
    pre_y = []
    X = np.asarray(X)
    for x in X:
        result = predict(theta,x,None,intercept)
        pre_y.append(result)
    return pre_y

if __name__ == '__main__':
    #pass
    theta,intercept,iter = fit(train_x,train_y,alpha=0.0001,fit_intercept=True,total_iter=1000)
    print("参数为{}".format(theta))
    print('截距项为{}'.format(intercept))
    print('供迭代{} 次'.format(iter))
    prey = predic_X(test_x,theta,intercept)
    print('R2 score 为{}'.format(r2_score(test_y,prey)))

    algo3 = Pipeline(steps=[
        #('scaler', PolynomialFeatures(degree=2)),
        # ('en',lm.ElasticNet(fit_intercept=True,alpha=0.1,l1_ratio=0.5,random_state=23))
        ('sgd', lm.SGDRegressor())
    ])
    algo3.fit(train_x, train_y)
    prey2 = algo3.predict(test_x)
    print('SGD R2 score 为{}'.format(r2_score(test_y, prey2)))
    #print(theta)
    t = np.arange(len(test_x))
    plt.figure(facecolor='white')

    plt.plot(t,test_y,color='red',lw=5,label='实际值')
    plt.plot(t, prey2, color='#FF00db',lw=3,label='SGD模型')
    plt.plot(t,prey,color='#00F0db',label='自定义回归模型')
    plt.legend(loc='upper right')
    plt.show()
