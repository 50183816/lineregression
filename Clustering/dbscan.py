# _*_ codig utf8 _*_
from sklearn.cluster import DBSCAN
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

'''
dbscan 算法： 密度聚类算法，以样本点x的eps邻域范围内的样本数作为密度N，将满足N(x)>M的x做为核心点，所有和核心点密度
相连的点归于核心点的簇。
'''
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 建立测试数据
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))
centers = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
k = 4  # len(centers)
N = 2000
# X, Y = make_blobs(n_samples=N, n_features=2, random_state=22, centers=centers, cluster_std=[1,0.75, 0.5,0.25])  # [0.2,0.4,0.8,1.6]
X,Y = make_circles(n_samples=N,shuffle=False,noise=0.01)
# X,Y = make_moons(n_samples=N,noise=0.05)
k = 20
# X, x_test, ytrain, y_test = train_test_split(x, y, test_size=0.4)

# 添加噪音
usenoise = False
if usenoise:
    n_noise = int(0.1 * N)
    r = np.random.rand(n_noise, 2)
    min1, min2 = np.min(X, axis=0)
    max1, max2 = np.max(X, axis=0)
    r[:, 0] = r[:, 0] * (max1 - min1) + min1
    r[:, 1] = r[:, 1] * (max2 - min2) + min2
    X = np.concatenate((X, r), axis=0)
    Y = np.concatenate((Y, [3] * n_noise))

# agens基于WL
linkage = ['ward', 'single', 'complete', 'average']
i = 2
# 画图，测试集
plt.figure(figsize=(12, 6), facecolor='w')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
cm = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#ffd966', '#5c5a5a'])
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.subplot(321)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=6, cmap=cm, edgecolors='none')
plt.xticks(())
# plt.xlim(-4,4)
plt.yticks(np.arange(-2, 2, 0.5))
plt.grid(False)
plt.title('原始数据')
radius = [0.1, 0.2, 0.3, 0.4,0.5]
params=[(0.05, 2), (0.1, 10), (0.2, 15), (0.3, 5), (0.3, 10)]
for r,m in params:
    # 可以不指定簇的数目，如果指定，会按照凝聚聚类的思想合并到指定簇。一般自动划分的簇多于要求的簇
    print(r)
    print(m)
    birch = DBSCAN(eps=r, min_samples=m, algorithm='auto')
    t0 = time.time()
    birch.fit(X)
    ward_timecost = time.time() - t0
    labels1 = birch.labels_
    k = len(np.unique(labels1)) - (1 if -1 in labels1 else 0)
    # print(labels1)
    score = 0
    if len(np.unique(labels1)) >= 2:
        score = silhouette_score(X, labels=labels1)
    else:
        score = 0
    plt.subplot(320 + i)
    plt.scatter(X[:, 0], X[:, 1], c=labels1.astype(np.float), cmap=cm, s=6, edgecolors='none')
    birch_center = birch.core_sample_indices_
    print(birch_center.shape)
    # plt.scatter(X[birch_center, 0], X[birch_center, 1], c='k', cmap=cm2, s=10, edgecolors='none')
    plt.xticks(())
    plt.yticks(())
    plt.grid(True)
    # print(birch.subcluster_centers_[:,0])
    plt.title('eps=%s划分结果,用时%.2fs,轮廓系数为%.2f,有%d个簇' % (r, ward_timecost, score,k))
    i += 1
    # print('{}预测值{}'.format(lkg,labels1))

plt.show()
