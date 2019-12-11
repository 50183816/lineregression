# _*_ codig utf8 _*_
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import  sklearn.preprocessing as pp
import sklearn.linear_model as lmd

datas = pd.read_csv('../datas/winequality-white.csv',delimiter=';')
print(datas.info)
print(datas.head(10))

X = datas.iloc[:,0:-1]
Y =datas.iloc[:,-1]
#print(X.describe())
# print(Y.head(5))
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

algo = Pipeline(steps=[
    ('scalar',pp.PolynomialFeatures(degree=3)),
    #('lr', lmd.Ridge(alpha=0.5,fit_intercept=True,random_state=23,max_iter=2000))
#('lr', lmd.LassoCV(alphas=[0.001,0.005,0.01,0.05,0.1,0.5,1],fit_intercept=True,random_state=23,max_iter=2000,))
    ('lr',lmd.ElasticNet(alpha=0.005))
])#LinearRegression(fit_intercept=True,normalize=False)

prefix = algo.fit(x_train,y_train)

predict = algo.predict(x_test).astype(np.int)
score = algo.score(x_test,y_test)

print('参数为：{}'.format(algo.steps[-1][-1].coef_))
print('截距项为：{}'.format(algo.steps[-1][-1].intercept_))
print('R2=%f'% score)