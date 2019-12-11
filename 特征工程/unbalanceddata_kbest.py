# _*_ codig utf8 _*_
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
if __name__ == '__main__':
    N1 = 10000
    N2=500
    np.random.seed(22)
    data1,label1 = datasets.make_blobs(N1,2,centers=[(0,0)],cluster_std=2)
    data2, label2 = datasets.make_blobs(N2, 2, centers=[(-2, -2.5)], cluster_std=[(2.0,1.0)])
    label1 = label1.reshape((-1,1))
    label2 = label2.reshape((-1, 1))
    label2[label2==0] = 1
    data1_ = np.concatenate((data1,label1),axis=1)
    data2_ = np.concatenate((data2, label2), axis=1)
    data = np.concatenate((data1_,data2_),axis=0)
    # KNeighbores
    # print(data1_[0, :].reshape(-1, 3))
    neighbors = 10
    nb = NearestNeighbors(n_neighbors=neighbors)
    # print(data[:,:-1].shape)
    nb.fit(data[:,:-1])
    index_to_delete=[]
    # print(data1_[0])
    # print(data1_[0][:-1])
    for d in np.arange(N1):
        _, index = nb.kneighbors(data1_[d][:-1].reshape(-1, 2))
        # print(np.shape(index))
        index = index.flatten()
        issameclass=True
        # print('K邻居的数据：{}'.format(data[index]))
        for i in np.arange(neighbors):
            if data[index[i],-1] != data1_[d,-1]:
                issameclass = False
                break
        if not issameclass:
            index_to_delete.append(d)

    print(index_to_delete)
    data1_ = np.delete(data1_,index_to_delete,axis=0)
    data = np.concatenate((data1_, data2_), axis=0)
    print('最後剩餘數據：{}'.format(np.shape(data1_)))
    X = data[:,0:2]
    Y = data[:,-1]
    lr = LogisticRegression()#class_weight={0:0.1,1:3} 类别权重，关注的一类提高权重
    lr.fit(X,Y)
    predict = lr.predict(X)
    w1,w2 = lr.coef_[0]
    b = lr.intercept_
    report = classification_report(Y,predict)
    print(report)
    plt.plot(data1_[:,0],data1_[:,1],'bo',markersize=1)
    plt.plot(data2[:, 0], data2[:, 1], 'ro',markersize=1)
    plt.plot([-8,8],[8*w1/w2-b/w2,-8*w1/w2-b/w2],'r-')
    plt.show()
