# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'simHei']

warnings.filterwarnings('ignore')


def calcuateJ(x, theta):
    n = np.shape(theta)
    tmp = 0
    for i in np.arange(n[0]):
        tmp += x[i] * theta[i]

    return tmp


# 梯度下降法训练模型
def fit(x, y, threshold=0e-5, max_iter=100, alpha=0.01):
    x = np.array(x)
    y = np.array(y)
    m, n = np.shape(x)
    theta = np.ones(n)
    # theta[-1]=0.6
    iter = 0
    h = np.dot(x, theta.T)

    hy_diff = y
    jtheta = 200
    jtheta_change = 1000
    h_points = []
    while jtheta_change > threshold and iter < max_iter:
        print('{}轮的theta是{}\n,h是{}'.format(iter, theta, h[:5]))
        random_index = np.arange(m)  # np.random.permutation(m)

        temp_h = 0
        for i in np.arange(m):
            th = calcuateJ(x[i, :], theta.T)
            tmp = 0
            y_diff = (y[i] - th)
            temp_h += y_diff ** 2
            for j in np.arange(n):
                tmp += y_diff * x[i, j]
                theta[j] += alpha * tmp
            # if i>0 and i%100 == 0:
            #     h_points.append(temp_h/100)
            #     temp_h = 0
        h_points.append(temp_h / m)
        tmp = 0
        h = np.dot(x, theta.T)
        for i in np.arange(m):
            tmp += (h[i] - y[i]) ** 2
        tmp /= m
        print('jt={}, tmp={}'.format(jtheta, tmp))
        jtheta_change = np.abs(jtheta - tmp)
        jtheta = tmp
        iter += 1
    return theta,h_points


# 1. Load data
data = pd.read_csv('../datas/household_power_consumption_1000.txt', sep=';')
# data.info()
# print(data.head())

# 2 Get property attribute and target antribute x, y
x = data.iloc[:, 2:4].astype(np.float)
y = data.iloc[:, 5].astype(np.float)
scalar = StandardScaler()
x = scalar.fit_transform(x, y)
x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=2)
theta,hpoint = fit(train_x, train_y, max_iter=500, alpha=0.00009)
print('训练出来的theta是：{}'.format(theta))

y_pred = np.dot(test_x, theta.T)
r2 = r2_score(test_y, y_pred)
print('测试集上的Score=%.3f' % r2)

t = np.arange(len(test_x))
plt.figure(facecolor='w')
plt.subplot(221)
plt.plot(t, test_y, 'r-', label=u'真实值')
plt.plot(t, y_pred, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.title('Random GD score=%.3f' % r2)

plt.subplot(222)
plt.plot(np.arange(len(hpoint)),hpoint,'r-',label='损失函数收敛情况')
plt.title('损失函数收敛情况')
plt.show()
