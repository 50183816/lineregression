# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
import sklearn.linear_model as lmd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.externals import joblib

# 1 加载数据
datafile_path = '../datas/boston_housing.data'
# print(help(pd.read_csv))
# return_X_Y：分开返回特征属性和目标属性
X, Y = load_boston(return_X_y=True)  # pd.read_csv(datafile_path,sep='\\s+')
# print()

# 2 数据清洗

# 3 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=22, shuffle=True)

# 4 特征工程

# 5 s算法模型构建
algo = Pipeline(steps=[('scaler', StandardScaler()),
                       #     ('poly',PolynomialFeatures(degree=3)),
                       ('lr', lmd.Lasso(alpha=0.01, fit_intercept=True, max_iter=2000, random_state=22))])

# 6 训练模型
algo.fit(x_train, y_train)
trainscore = algo.score(x_train, y_train)

y_predict = algo.predict(x_test)
testscore = algo.score(x_test, y_test)
# 7 检测模型效果
print('模型训练数据集R2得分:{}'.format(trainscore))
print('模型测试数据集R2得分:{}'.format(testscore))
# print('模型测试数据集R2得分:{}'.format(r2_score(y_test,y_predict)))
print('模型参数:{}'.format(algo.steps[-1][-1].coef_[:10]))
print('模型截距项:{}'.format(algo.steps[-1][-1].intercept_))

# 保存模型
joblib.dump(algo, './02_lasso.algo')

#Lasso选取非零特征后再用Ridge回归
print('*'*100)
#获取参数为0的特征
model = algo.steps[-1][-1]
threshold = 0.004
x_train = x_train[:, np.abs(model.coef_) > threshold]
x_test = x_test[:, np.abs(model.coef_) > threshold]
# print(np.abs(model.coef_) > threshold)
algo = Pipeline(steps=[('scaler',StandardScaler()),
                       ('poly',PolynomialFeatures(degree=2)),
                       ('lr',lmd.Ridge(alpha=0.5,fit_intercept=True,max_iter=2000,random_state=2))])

#6 训练模型
algo.fit(x_train,y_train)
trainscore = algo.score(x_train,y_train)

y_predict = algo.predict(x_test)
testscore = algo.score(x_test,y_test)
#7 检测模型效果
print('模型训练数据集R2得分:{}'.format(trainscore))
print('模型测试数据集R2得分:{}'.format(testscore))
# print('模型测试数据集R2得分:{}'.format(r2_score(y_test,y_predict)))
print('模型参数:{}'.format(algo.steps[-1][-1].coef_[:10]))
print('模型截距项:{}'.format(algo.steps[-1][-1].intercept_))
