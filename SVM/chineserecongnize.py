# _*_ codig utf8 _*_
from PIL import Image
import numpy as np
import os
from sklearn.svm import SVC,NuSVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import time
from sklearn.preprocessing import StandardScaler
# 手写中文识别  汉字 一

dir = 'F:\\AI学习资料\\[20181201]_SVM、多分类及多标签分类算法\\05_随堂代码\\05_随堂代码\作业数据\\中文字符识别\\训练数据\\0000'

X = []
Y = []
category = 0
folders = np.arange(0, 10)
for f in folders:
    category = f
    for root, _, files in os.walk('%s%d' % (dir, f)):
        for file in files:
            filepath = os.path.join(root, file)
            # print(filepath)
            img = Image.open(filepath)
            img2 = img.resize((32, 32))
            bytes = np.asarray(img2).ravel()
            # print(bytes.shape)
            X.append(bytes)
            Y.append(category)

print(len(Y))
print(Y)
X = np.reshape(X, (-1, 3072))
ss = StandardScaler()
ss.fit_transform(X,Y)
svc = SVC(kernel='poly', degree=6,gamma=0.001)
start = time.time()
svc.fit(X, Y)
end = time.time()
print('用時:{}s'.format(end-start))

joblib.dump(svc, './svc.alg')
py = svc.predict(X)
rpt = classification_report(Y, py)
print(rpt)
score = svc.score(X, Y)
print(score)
d1 = Image.open('y2.png').resize((32, 32))
data = np.asarray(d1).ravel()
d1y = svc.predict([data])
print('预测值为:{}'.format(d1y))
