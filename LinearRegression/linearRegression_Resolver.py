# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#线性回归，解析式求解
class DirectResolverLR():
    def __init__(self):
        self.coef_=[]
        self.intercept_=[]
    #训练模型基于公式： θ = (X.T*X + λI)*X.T*Y
    def fit(self,x,y,lbd):
            m,_ = np.shape(x)
            intercept = np.ones((m,1))
            x = np.concatenate((x,intercept),axis=1)
            _,n = np.shape(x)
            addition = np.identity(n) * lbd
            tmp = np.linalg.inv((np.dot(x.T,x)+addition))
            theta = np.dot(np.dot(tmp,x.T),y)
            self.coef_ = theta
            return theta

    def predict(self,x):
        m,_ = np.shape(x)
        intercept = np.ones((m, 1))
        x = np.concatenate((x, intercept), axis=1)
        y = np.dot(x,self.coef_)
        return y

    def score(self,x,y):
        predy =self.predict(x)
        result = r2_score(y,predy)
        return result

if __name__ == '__main__':
    # 1. Load data
    data = pd.read_csv('../datas/household_power_consumption_1000.txt', sep=';')
    # data.info()
    # print(data.head())

    # 2 Get property attribute and target antribute x, y
    x = data.iloc[:, 2:4].astype(np.float)
    y = data.iloc[:, 5].astype(np.float)
    # scalar = StandardScaler()
    # x = scalar.fit_transform(x, y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=2)
    lr = DirectResolverLR()
    theta = lr.fit(train_x,train_y,0)
    print(theta)
    print('训练集score=%.2f'%lr.score(train_x,train_y))
    print('测试集score=%.2f' % lr.score(test_x, test_y))

    t = np.arange(len(test_x))
    plt.figure(facecolor='w')
    plt.plot(t, test_y, 'r-', label=u'真实值')
    plt.plot(t, lr.predict(test_x), 'b-', label=u'预测值')
    plt.legend(loc='lower right')
    plt.show()