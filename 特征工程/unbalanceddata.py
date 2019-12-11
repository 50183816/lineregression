# _*_ codig utf8 _*_
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
if __name__ == '__main__':
    N1 = 10000
    N2=500
    data1,label1 = datasets.make_blobs(N1,2,centers=[(0,0)],cluster_std=2)
    data2, label2 = datasets.make_blobs(N2, 2, centers=[(-2, -2.5)], cluster_std=[(2.0,1.0)])
    label1 = label1.reshape((-1,1))
    label2 = label2.reshape((-1, 1))
    label2[label2==0] = 1
    data1_ = np.concatenate((data1,label1),axis=1)
    data1_ = data1_[np.random.permutation(N1)[:N2]] #下采样
    data2_ = np.concatenate((data2, label2), axis=1)
    print(data1.shape)
    data = np.concatenate((data1_,data2_),axis=0)
    print(data)
    X = data[:,0:2]
    Y = data[:,-1]
    lr = LogisticRegression()#class_weight={0:0.1,1:3} 类别权重，关注的一类提高权重
    lr.fit(X,Y)
    predict = lr.predict(X)
    w1,w2 = lr.coef_[0]
    b = lr.intercept_
    report = classification_report(Y,predict)
    print(report)
    plt.plot(data1[:,0],data1[:,1],'bo',markersize=1)
    plt.plot(data2[:, 0], data2[:, 1], 'ro',markersize=1)
    plt.plot([-8,8],[8*w1/w2-b/w2,-8*w1/w2-b/w2],'r-')
    plt.show()
