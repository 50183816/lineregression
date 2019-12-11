# _*_ codig utf8 _*_
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def getsplit(x,y,d):
    #问题：固定了划分，但是实际上是要看具体值去确定正负例在左右的分布
    min_error_rate = 1
    min_error_rate_index = 0
    thresholdvalue = x[0]
    predict = []
    for i in np.arange(len(x)-1):
       pos_split = list(map(lambda yv: 1 if yv<0 else 0, y[0:i+1]))
       neg_split = list(map(lambda yv: 1 if yv>0 else 0, y[i+1:]))
       combined = np.concatenate((pos_split,neg_split))
       error_rate = np.sum(combined * d)
       if(error_rate < min_error_rate):
           min_error_rate = error_rate
           min_error_rate_index = i


    return min_error_rate,min_error_rate_index

if __name__ == '__main__':

    #1. 准备样本数据
    X= np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
    Y = np.array([1,1,1,-1,-1,-1,1,1,1,-1]).reshape(-1,1)

    #2. 设置初始样本权重D
    d0 = np.ones(len(X))
    d0 = d0 / len(d0)#初始权重，设为均值

    #3 训练第一个决策树
    # tree = DecisionTreeClassifier()
    # tree.fit(X,Y,sample_weight=d0)
    # predicted = tree.predict(X)
    # print(predicted)
    #3 划分样本
    for r in np.arange(5):
        _,splitindex=getsplit(X,Y,d0)
        print('第{}轮划分的属性为{}'.format(r,splitindex))
        # print(X[:,0]<=splitindex)
        predict1 = np.array(Y.ravel()) #[1 if  ]  np.array([1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])
        predict1[X[:,0]<=splitindex] = 1
        predict1[X[:,0]>splitindex] = -1
        # print(predict1)
        pred_true_values = zip(Y.ravel(),predict1,d0)
        #4.计算新的样本权重D1，epsilon 和alpha值
        sum = 0
        for y,p,d in pred_true_values:
            sum += (0 if y==p else 1)*d
            # print((y,p,d))
        # print('epsilon_1 = {}'.format(sum))

        alpha1 = 0.5 * np.log1p((1-sum)/sum)
        print('alpha{} = {}'.format(r+1,alpha1))
        sum = 0
        d1=[]
        pred_true_values = zip(Y.ravel(),predict1,d0)
        for y,p,d in pred_true_values:
            # print((y,p,d))
            d1.append(d*np.exp(-1*y*p*alpha1))
            sum += d*np.exp(-1*y*p*alpha1)
        # print(d1)
        # print(sum)
        d0 = np.array(d1)/sum
        # print(d0)
        # min_erro_rate,min_erro_index = getsplit(X,Y,d1)
        # print((min_erro_rate,min_erro_index))
        #5. 训练下一个模型

