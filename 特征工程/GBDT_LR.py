# _*_ codig utf8 _*_
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import auc, roc_curve
import warnings

warnings.filterwarnings('ignore')
np.random.seed(22)

X, Y = make_classification(n_samples=80000, n_features=5)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

gbdt = GradientBoostingClassifier(n_estimators=50)
gbdt.fit(x_train, y_train)

index = gbdt.apply(x_train)
# print(np.shape(index))
# print(index[0, :, :])
encoder = OneHotEncoder()
# print('before onehotencode:\n{}'.format(np.shape(index[:,:,0])))
# print('before onehotencode:\n{}'.format((index[0:5,:,0])))
index = encoder.fit_transform(index[:, :, 0])
# print('after onehotencode:\n{}'.format(np.shape(index)))
# print('after onehotencode:\n{}'.format((index.toarray()[0:5,:])))
# print(index)
lr = LogisticRegression()
lr.fit(index, y_train)
print(lr.classes_)
prob = lr.predict_proba(encoder.transform(gbdt.apply(x_test)[:, :, 0]))[:, 1]#类别1为正例

fpr, tpr, _ = roc_curve(y_test, prob)
auc_value = auc(fpr, tpr)
print('auc=%f' % auc_value)
