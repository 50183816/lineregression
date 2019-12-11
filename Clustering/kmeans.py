# _*_ codig utf8 _*_
from  sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 建立测试数据
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))

# x,y = make_blobs(n_samples=500,n_features=2,random_state=22,centers=4,cluster_std=[0.2,0.4,0.8,1.6])
# x,y = make_circles(n_samples=500,shuffle=True,noise=0.1)
x,y = make_moons(n_samples=500,shuffle=True,noise=0.01)
X,x_test,ytrain,y_test = train_test_split(x,y,test_size=0.4)


plt.subplot(221)
plt.scatter(x_test[:,0],x_test[:,1])
plt.title('原始数据')
#plt1.show()
kmean = KMeans(n_clusters=4,init='k-means++')
kmean.fit(X)
labels = kmean.predict(x_test)
#labels = kmean.labels_
score = silhouette_score(x_test,labels)
print('预测score：{}'.format(score))

plt.subplot(222)
plt.scatter(x_test[:,0],x_test[:,1],c=labels.astype(np.float),edgecolors='k')
plt.title('KMeans划分结果')
minibatch = MiniBatchKMeans(n_clusters=4)
minibatch.fit(X)
labels2  = minibatch.predict(x_test)
# labels2 = minibatch.labels_

plt.subplot(223)
plt.scatter(x_test[:,0],x_test[:,1],c=labels2.astype(np.float),edgecolors='k')
plt.title('MiniBatch划分结果')
score = silhouette_score(x_test,labels2)
print('MiniBatch预测score：{}'.format(score))
plt.show()

