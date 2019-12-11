# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split,GridSearchCV
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7, shuffle=True)

# 4 特征工程

# 5 s算法模型构建
lr1 = LinearRegression(fit_intercept=True)
lr2 = Ridge(alpha=0.5, fit_intercept=True)
lr3=RidgeCV(alphas=[0.01,0.05,0.1,0.5,1],fit_intercept=True,cv=5)
lr4=Lasso(alpha=0.05,fit_intercept=True,max_iter=1000)
lr5=DecisionTreeRegressor(max_depth=3)
lr6=ExtraTreeRegressor(max_depth=10,max_features=None)
pipe = Pipeline(steps=[('scaler', StandardScaler()),
                       # ('poly',PolynomialFeatures(degree=2)),
                       ('lr', AdaBoostRegressor(random_state=5, n_estimators=3000,base_estimator=lr1,learning_rate=0.05))])

parameters={'lr__n_estimators':[10,100,50],'lr__base_estimator':[lr1,lr5,lr6],'lr__learning_rate':[0.8,1,0.1,0.5,0.01,0.05,2,3,4]}
algo=GridSearchCV(estimator=pipe,param_grid=parameters,cv=3,iid=False)
# 6 训练模型
algo.fit(x_train, y_train)
trainscore = algo.score(x_train, y_train)

y_predict = algo.predict(x_test)
testscore = algo.score(x_test, y_test)
# 7 检测模型效果
print('模型训练数据集R2得分:{}'.format(trainscore))
print('模型测试数据集R2得分:{}'.format(testscore))
print('模型参数:{}'.format(algo.best_params_))
# print('模型测试数据集R2得分:{}'.format(r2_score(y_test,y_predict)))
# print('子模型個數：{}'.format(len(algo.steps[-1][-1].estimators_)))
joblib.dump(algo, './07_adaboosting.algo')
