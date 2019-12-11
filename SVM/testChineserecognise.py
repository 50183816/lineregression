# _*_ codig utf8 _*_
# _*_ codig utf8 _*_
from PIL import  Image
import numpy as np
import os
from sklearn.svm import SVC,NuSVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report


svc = joblib.load('./svc.alg')

dir = 'F:\\AI学习资料\\[20181201]_SVM、多分类及多标签分类算法\\05_随堂代码\\05_随堂代码\作业数据\\中文字符识别\\验证数据\\0000'

X = []
Y = []
category = 0
folders = np.arange(0,10)
for f in folders:
    category = f
    for root , _ , files in os.walk('%s%d'%(dir,f)):
        for file in files:
            filepath = os.path.join(root,file)
            # print(filepath)
            img = Image.open(filepath)
            img2 = img.resize((32,32))
            bytes = np.asarray(img2).ravel()
            # print(bytes.shape)
            X.append(bytes)
            Y.append(category)

X = np.reshape(X,(-1,3072))
print(X.shape)
# d1 = Image.open('y1.png').resize((32,32))
# d1.show()
# data = np.asarray(d1).ravel()
# d1y = svc.predict([data])
# print('预测值为:{}'.format(d1y))

predicted = svc.predict(X)
report = classification_report(Y,predicted)
print(report)
print(predicted)