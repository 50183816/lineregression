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
def JTheta(theta,x,y,intercept=0):
    sum = 0
    n = x.shape[0]
    for i in range(n):
        sum += (theta * x[i].T - y[i]+intercept).T *(theta * x[i].T - y[i]+intercept)
    return sum/2

def JTheta2(theta,x,y,intercept=0):
    sum = 0
    n = x.shape[0]
    for i in range(n):
        sum += predict(theta,x[i],y[i],intercept) ** 2
    return sum/2

def predict(theta,x,y,intercept=0):
    '''

    :param theta:
    :param x:
    :param y:
    :param intercept:
    :return:
    '''
    # = np.asarray(theta)
    #print(x)
   # x = np.asarray(x)
    sum = 0
    #print('x的shape：{}'.format(type(x)))
    n = len(x)
    for i in range(n):
        # print(theta[i].shape)
        # print(x[i])
        sum += theta[i]*x[i]
    sum += intercept
    return sum


def dh(theta,x,y):
    sum = 0
    i = random.randint(0,x.shape[0]-1)
    print('原Theta:')
    print(theta)
    print('样本:')
    print(x[i])
    print(y[i])
    r = (y[i] - theta * x[i].T) * x[i]
    print('梯度：')
    print(r)
    return r


def dhb(theta,x,y):
    sum = 0
    n = x.shape[0]
    for i in range(n):
        sum += (y[i] - theta * x[i].T) * x[i]/n

    return sum

def dh1(theta,x,y):
    # print(theta.shape)
    # print(x.shape)
    # print(y.shape)
    r = (y - theta * x.T) * x
    # print('梯度：')
    # print(r)
    return r

def fit(X,Y,alpha=0.0001,fit_intercept=False,tol=1e-6,total_iter=2000):
    m,n = X.shape
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)

    theta = np.zeros(n)
    intercept = 30000
    jtheta = intercept
    jtheta_change = jtheta
    iter = 0
    diff = np.zeros(shape=[m])
    while jtheta_change > tol and iter < total_iter:
        #1.计算梯度
        #gd = dhb(theta,matx,maty.T)
        gd = 0
        #计算预测值和实际值之间的差值
        for i in range(m):
            y_predict = predict(theta,X[i],Y[i],intercept)
            y_true = Y[i]
            #print((y_predict))
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
    theta,intercept,iter = fit(train_x,train_y,alpha=0.01,fit_intercept=True,total_iter=200)
    print("参数为{}".format(theta))
    print('截距项为{}'.format(intercept))
    print('供迭代{} 次'.format(iter))
    prey = predic_X(test_x,theta,intercept)
    print('R2 score 为{}'.format(r2_score(test_y,prey)))
    #print(theta)
