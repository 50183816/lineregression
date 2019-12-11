# _*_ codig utf8 _*_

import numpy as np
import pandas as pd
import matplotlib as mpl
import  matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_iris
mpl.rcParams['font.sans-serif'] = [u'simHei']

#1. Load data
X,Y= load_iris(return_X_y=True)
#data.info()
#print(data.head())

#2 Get property attribute and target antribute x, y
#print(y[:5])
#scalar = StandardScaler()
#print(x)
#3 split test dataset and training  dataset
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=22)
x1=np.mat(train_x)
y1=np.mat(train_y).reshape(-1,1)

algo = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('log',LogisticRegression())
])
algo.fit(x1,y1)
score = algo.score(x1,y1)
print('score={}'.format(score))
pred_y = algo.predict(test_x)
test_score = algo.score(test_x,test_y)
prob = algo.predict_proba(test_x)
print('testing sample score={}'.format(test_score))

print(pred_y[:10])
print(prob[:10])