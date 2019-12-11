# _*_ codig utf8 _*_
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as pp
import sklearn.linear_model as lmd
import sklearn.metrics as mtx
import matplotlib as mpl

# 0 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.加载数据
datas = pd.read_csv('../datas/breast-cancer-wisconsin.data', delimiter=',', header=None)
print(datas.info)
print(datas.describe())

# 2.数据清洗
datas.replace('?',0,inplace=True)
# 3.拆分训练集和测试集
X = datas.iloc[:, 1:-1]
Y = datas.iloc[:, -1]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=23)

# 4.构建模型，训练算法并预测
algo = Pipeline(steps=[
    ('scalar',pp.StandardScaler()),
    ('lr',lmd.LogisticRegression())
])
# 5.考察训练和测试结果
algo.fit(x_train,y_train)
train_score =algo.score(x_train,y_train)

predict = algo.predict(x_test)

test_score = algo.score(x_test,y_test)

print('分類結果報表:\n{}'.format(mtx.classification_report(y_test,predict)))

print('訓練結果分數：{} 測試結果分數：{}'.format(train_score,test_score))

print('參數：{}'.format(algo.steps[-1][-1].coef_))
print('概率值：{}'.format(algo.predict_proba(x_test)))