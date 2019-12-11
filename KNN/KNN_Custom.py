# _*_ codig utf8 _*_
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,auc,roc_curve
from matplotlib import pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier

class CustomKNN():
    def __init__(self,k=5):
        self.X=[]
        self.Y=[]
        self.K = k

    def CalculateDistince(self,x1,x2):
        '''
        计算两点之间的欧几里得距离
        :param x1:
        :param x2:
        :return: 返回距离
        '''
        tmp = np.array(x1)-np.array(x2)
        return np.sqrt(np.sum(list(map(lambda x:np.square(x), tmp))))

    def GetKNeighbors(self,x,k=5):
        '''
        查询K个邻居
        :param x: 样本点
        :param k: 邻居个数
        :return: 邻居列表，nX2数组：[邻居点索引，和样本点的距离]
        '''
        neighbors = []
        dist = 1000
        max_dist_idx=-1
        for i in np.arange(np.shape(self.X)[0]):
            tmp_dist = self.CalculateDistince(x,self.X[i])
            if len(neighbors) < k:
                neighbors.append([i,tmp_dist])
            elif dist>tmp_dist:
                neighbors[max_dist_idx] = [i,tmp_dist]
            max_idx = np.argmax(np.array(neighbors)[:,-1], axis=0)
            dist = neighbors[max_idx][-1]
            max_dist_idx = max_idx
        return np.array(neighbors)

    def fit(self,x,y):
         self.X = x
         self.Y = y

    def VoteForClass(self,idx):
        '''
        投票决定类别
        :param idx: 参与投票的样本点索引
        :return: 类别，idx中的多数类别
        '''
        classes = self.Y[idx.astype(np.int)]
        print(classes)
        tmp = np.bincount(classes)
        print('tmp ={}'.format(tmp))
        idx = np.argmax(tmp)
        return idx

    def predict(self,x ):
        '''
        预测
        :param x:
        :return:
        '''
        m,_ = np.shape(x)
        predict = []
        nbs=[]
        for i in np.arange(m):
            neighbors = self.GetKNeighbors(x[i],self.K)
            print(neighbors[:,0])
            cls = self.VoteForClass(neighbors[:,0])
            predict.append(cls)
            nbs.append(list(neighbors[:,0].astype(np.int)))
        return predict,nbs

if __name__ == '__main__':
    x = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y = np.array([0, 1, 0, 1, 1, 0])
    x,y = load_iris(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    knn = CustomKNN(k=5)
    knn.fit(x_train,y_train)
    result,nbs = knn.predict(x_test)
    report = classification_report(y_test,result)
    print(report)
    x_test_len = np.arange(len(x_test))
    sklkb = KNeighborsClassifier()
    sklkb.fit(x_train,y_train)
    pred_test = sklkb.predict(x_test)
    report = classification_report(y_test,pred_test)
    print('SklearnKNN Report:\n{}'.format(report))
    plt.plot(x_test_len, y_test, 'ro', markersize=7, label='真实值')
    plt.plot(x_test_len, result, 'bo', markersize=5, label='KNN预测值')
    # plt.plot(x_test_len, predit2, 'ko', markersize=3, label='Logistics预测值')
    # Logistics预测值plt.set_title('鸢尾花分类预测，准确度：KNN=%f Logis=%f' % (r1, r2))
    plt.show()
    # roc_curve(y_test,result)
    # print('predicted class is {}, neighbor is {}'.format(result,nbs))