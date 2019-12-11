# _*_ codig utf8 _*_
import numpy as np
'''
ＥＭ算法计算过程：
１.　对模型参数θ给定初始值。 这里θ就是（p1，p2）的概率
2.   跌代计算新的(p1,p2)。 先利用上一轮的值计算出本轮中样本的概率P(z|x),根据概率找出每个样本对应的Z值，然后再使用MLE计算出
     新的(p1,p2). 直到收敛。
3.   另外一种方式，在计算出概率P(z|X)之后，对每一个Z值，对全样本计算白黑球分布总体期望，再基于期望利用MLE计算P值，直到收敛。
'''
data = [[3, 2], [2, 3], [1, 4], [3, 2], [2, 3]]

#给定初始值 theta
p1 = 0.1
p2 = 0.9
#开始迭代
for i in np.arange(1, 10):
    d1 = []
    d2 = []
    p = []
    #计算每个样本的概率
    tw1,tb1,tw2,tb2 = 0,0,0,0
    for w, b in data:
        tp1 = np.power(p1, w) * np.power(1 - p1, b)
        tp2 = np.power(p2, w) * np.power(1 - p2, b)
        # tmp = [tp1 / (tp1 + tp2), tp2 / (tp1 + tp2)]
        tw1 += w*(tp1/(tp1+tp2))
        tb1 += b*(tp1/(tp1+tp2))
        tw2 += w*(tp2/(tp1+tp2))
        tb2 += b*(tp2/(tp1+tp2))


        p.append([tp1/(tp1+tp2),tp2/(tp1+tp2)])
        if tp1 > tp2 :
            d1.append([w,b])
        else:
            d2.append([w,b])
    # print(p)
    # pp = np.array(p)
    # d1 = np.array(d1)
    # d2 = np.array(d2)
    # data = np.array(data)
    # tmp = np.dot(p.T, data)
    p1 = tw1/(tw1+tb1)
    p2=tw2/(tw2+tb2)

    # p1 = np.sum(d1[:,0])/np.sum(d1)
    # p2=np.sum(d2[:,0])/np.sum(d2)
    print(p1,p2)
    # print([p1, p2])
    # print('概率为：\n{}'.format(p))