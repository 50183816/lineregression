# _*_ codig utf8 _*_
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# http://scipy.github.io/devdocs/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal

# 产生模拟数据
np.random.seed(22)
N1 = 400
N2 = 100
# 类别1的数据
mean = (0, 0, 0)
conv1 = np.diag([1, 2, 3])
# print(conv1)
data1 = np.random.multivariate_normal(mean=mean, cov=conv1, size=N1)
# TODO：2D图形显示
# 类别2的数据
mean2 = (5, 6, 7)
conv2 = np.array([[3, 1, 0], [1, 1, 0], [0, 0, 5]])
data2 = np.random.multivariate_normal(mean=mean2, cov=conv2, size=N2)

# 合并两个数据
data = np.vstack((data1, data2))

# 构建模型
max_iter = 100

m, d = data.shape

print('样本数：{}'.format((m, d)))

# 给定初值
mu1 = data.min(axis=0)
mu2 = data.max(axis=0)
sigma = np.identity(d)
sigma2 = np.identity(d)
print('初始均值：{}，{}'.format(mu1, mu2))
pi = 0.5  # pi2 = 1-pi

# 实现EM算法
for i in np.arange(max_iter):
    # E Step:计算当前模型参数下各个模型样本的条件概率
    norm1 = multivariate_normal(mu1, sigma)  # 概率密度函数
    norm2 = multivariate_normal(mu2, sigma2)
    # 获取样本概率密度值
    pdf1 = pi * norm1.pdf(data)
    pdf2 = (1 - pi) * norm2.pdf(data)
    # 归一化操作，计算w
    w1 = pdf1 / (pdf1 + pdf2)
    w2 = 1 - w1

    # MStep，根据样本条件概率更新模型参数
    # 均值更新
    mu1 = np.dot(w1, data) / np.sum(w1)
    mu2 = np.dot(w2, data) / np.sum(w2)
    sigma = np.dot(w1 * (data - mu1).T, data - mu1) / np.sum(w1)
    sigma2 = np.dot(w2 * (data - mu2).T, data - mu2) / np.sum(w2)
    pi = np.sum(w1) / m

print('最终的均值：{}，{}，pi:{},sigma1:{},sigma2:{}'.format(mu1, mu2, pi, sigma, sigma2))

x_test = [[0, 0, 0], [5, 6, 7], [2.5, 1.5, 2.5], [6, 8, 4]]
norm1 = multivariate_normal(mu1, sigma)  # 概率密度函数
norm2 = multivariate_normal(mu2, sigma2)
# 获取样本概率密度值
pdf1 = pi * norm1.pdf(data)
pdf2 = (1 - pi) * norm2.pdf(data)
# 归一化操作，计算w
w1 = pdf1 / (pdf1 + pdf2)
w2 = 1 - w1

print('预测为w1的概率{}，w2的概率为{}'.format(w1, w2))

w = np.vstack((w1, w2))
y_hat = np.argmax(w, axis=0)
# print('预测的值为{}'.format(y_hat.shape))


fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
y1 = np.array([True] * N1 + [False] * N2)
y2 = ~y1
ax.scatter(data[y1, 0], data[y1, 1], data[y1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[y2, 0], data[y2, 1], data[y2, 2], c='b', s=30, marker='o', depthshade=True)

ax = fig.add_subplot(122, projection='3d')
y1 = np.array([True if y == 0 else False for y in y_hat ])
y2 = ~y1
ax.scatter(data[y1, 0], data[y1, 1], data[y1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[y2, 0], data[y2, 1], data[y2, 2], c='b', s=30, marker='o', depthshade=True)

# 设置总标题
# plt.suptitle(u'EM算法的实现,准备率：%.2f%%' % (acc * 100), fontsize=20)
plt.subplots_adjust(top=0.90)
plt.tight_layout()
plt.show()
