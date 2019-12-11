# _*_ codig utf8 _*_
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 建立测试数据
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))
centers = [(-1, -1), (0, 0), (-1, 1), (1, 1)]
k = len(centers)
x, y = make_blobs(n_samples=5000, n_features=2, random_state=22, centers=centers, cluster_std=0.5)  # [0.2,0.4,0.8,1.6]
# x,y = make_circles(n_samples=500,shuffle=True,noise=0.1)
# x,y = make_moons(n_samples=500,shuffle=True,noise=0.01)
X, x_test, ytrain, y_test = train_test_split(x, y, test_size=0.4)

t0 = time.time()
kmean = KMeans(n_clusters=k, init='random')
kmean.fit(X)
km_timecost = time.time() - t0
labels = kmean.predict(x_test)
# labels = kmean.labels_
score = silhouette_score(x_test, labels)
print('预测score：{}'.format(score))

minibatch = MiniBatchKMeans(n_clusters=k, batch_size=50,n_init=3)
t0 = time.time()
minibatch.fit(X)
mbkm_timecost = time.time() - t0
labels2 = minibatch.predict(x_test)

# 打印中心点
km_centers = kmean.cluster_centers_
minibatch_center = minibatch.cluster_centers_
print('km预测值{}\n中心点为：\n{}'.format(labels, km_centers))
print('minibatchkm预测值{}中心点为：\n{}'.format(labels2, minibatch_center))
index = pairwise_distances_argmin(km_centers, minibatch_center)
print('km中中心点在minibatch中心点中的index:{}'.format(index))

# 画图，测试集
plt.figure(figsize=(12, 6), facecolor='w')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.subplot(221)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=6, cmap=cm, edgecolors='none')
#plt.xticks(())
# plt.xlim(-4,4)
plt.yticks(np.arange(-2,2,0.5))
plt.grid(False)
plt.title('原始数据')

plt.subplot(222)
plt.scatter(x_test[:, 0], x_test[:, 1], c=labels.astype(np.float),cmap=cm,s=6, edgecolors='none')
plt.scatter(km_centers[:,0],km_centers[:,1],c=range(k),cmap=cm2,s=60,edgecolors='none')
plt.xticks(())
plt.yticks(())
plt.grid(True)
plt.title('KMeans划分结果,用时%.2fs' % km_timecost)

plt.subplot(223)
plt.scatter(x_test[:, 0], x_test[:, 1], c=labels2.astype(np.float),s=6, edgecolors='k')
plt.scatter(minibatch_center[:,0],minibatch_center[:,1],c=range(k),cmap=cm2,s=60,edgecolors='none')
plt.xticks(())
plt.yticks(())
plt.grid(True)
plt.title('MiniBatch划分结果,用时 %.2fs' % mbkm_timecost)
score = silhouette_score(x_test, labels2)
print('MiniBatch预测score：{}'.format(score))
plt.show()
