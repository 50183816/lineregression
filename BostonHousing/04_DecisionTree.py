# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import  Pipeline
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.externals import joblib
# 1 加载数据
datafile_path = '../datas/boston_housing.data'
# print(help(pd.read_csv))
#return_X_Y：分开返回特征属性和目标属性
X,Y =  load_boston(return_X_y=True) #pd.read_csv(datafile_path,sep='\\s+')
# print()

# 2 数据清洗

# 3 拆分数据集
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=27,shuffle=True)

#4 特征工程

#5 s算法模型构建
algo = Pipeline(steps=[('scaler',StandardScaler()),
                       # ('poly',PolynomialFeatures(degree=3)),
                       ('lr',DecisionTreeRegressor(max_depth=20,random_state=27,max_features=10))])

#6 训练模型
algo.fit(x_train,y_train)
trainscore = algo.score(x_train,y_train)

y_predict = algo.predict(x_test)
testscore = algo.score(x_test,y_test)
#7 检测模型效果
print('模型训练数据集R2得分:{}'.format(trainscore))
print('模型测试数据集R2得分:{}'.format(testscore))
# print('模型测试数据集R2得分:{}'.format(r2_score(y_test,y_predict)))
print('属性重要性：{}'.format(algo.steps[-1][-1].feature_importances_))

joblib.dump(algo,'./04_decisiontree.algo')