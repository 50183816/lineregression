# _*_ codig utf8 _*_
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import  sklearn.neighbors  as nbs
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib as  mpl
from sklearn.metrics import roc_curve
from sklearn.metrics import  auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

names=['sepal_len','sepal_width','petal_len','petal_width','cls']
data = pd.read_csv('../datas/iris.data',header=None,names=names)
#print(data.info())
m,n = data.shape
for i in np.arange(m):
   print(data['cls'][i])
   if data['cls'][i]=='Iris-setosa':
       data['cls'][i]=1
   elif data['cls'][i] == 'Iris-virginica':
       data['cls'][i] = 2
   elif data['cls'][i] == 'Iris-versicolor':
       data['cls'][i] = 3
   else:
       data['cls'][i] = np.NaN

data = data.dropna(how='any')
X=data.iloc[:,0:-1]
Y = data['cls'].astype(int)
#print(X)
#print(Y.dtype)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=23)
print('训练集有记录数%d,测试集有数据%d'%(x_train.shape[0],x_test.shape[0]))
knn = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='brute',leaf_size=30,p=2,metric='minkowski',metric_params=None)
#利用網格交叉驗證選擇參數
'''
estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)
'''
algo = GridSearchCV(estimator=knn,param_grid={
    'n_neighbors':[2,3,4,5,6],
    'weights':['uniform','distance'],
    'algorithm':['brute','kd_tree','ball_tree'],
    'leaf_size':[20,25,30,35]
},cv=3,refit=True)
algo.fit(x_train,y_train)
print('KNN最佳參數{}'.format(algo.best_params_))
print('KNN 最優模型：{}'.format(algo.best_estimator_))
print('KNN的训练结果：%f'%algo.best_score_)
#print(r)
#algo = algo.best_estimator_
predit = algo.predict(x_test)
r1 = algo.score(x_test,y_test)
neighbors = nbs.NearestNeighbors(3)
neighbors.fit(x_train,y_train)

myneighbors =  neighbors.kneighbors(x_train,3,return_distance=True)
print('*'*100)
print(myneighbors)

#Logistic Regression
logistic = LogisticRegression(penalty='l2',fit_intercept=True,max_iter=100)
logistic.fit(x_train,y_train)
r2 = logistic.score(x_test,y_test)
print('Logistics训练结果：%f'%r2)


predit2 = logistic.predict(x_test)
y_label = label_binarize(y_test,classes=[1,2,3])
print(y_label)
fpr,tpr,_ = roc_curve(y_label.ravel(),algo.predict_proba(x_test).ravel())
aucValue = auc(fpr,tpr)
print(aucValue)

fpr_log,tpr_log,_ =  roc_curve(y_label.ravel(),logistic.predict_proba(x_test).ravel())
auc_log = auc(fpr_log,tpr_log)
x_test_len = np.arange(len(x_test))
# plt.plot(x_test_len,y_test,'ro',markersize=7,label='真实值')
# plt.plot(x_test_len,predit,'bo',markersize=5,label='KNN预测值')
# plt.plot(x_test_len,predit2,'ko',markersize=3,label='Logistics预测值')
# plt.title('鸢尾花分类预测，准确度：KNN=%f Logis=%f'% (r1,r2))
# plt.legend(loc='lower right')

#plt.figure(facecolor='white')
fig,plts = plt.subplots(2,1)

plts[0].plot(x_test_len,y_test,'ro',markersize=7,label='真实值')
plts[0].plot(x_test_len,predit,'bo',markersize=5,label='KNN预测值')
plts[0].plot(x_test_len,predit2,'ko',markersize=3,label='Logistics预测值')
plts[0].set_title('鸢尾花分类预测，准确度：KNN=%f Logis=%f'% (r1,r2))
plts[0].legend(loc='lower right')

plts[1].plot(fpr,tpr,'r',label='KNN ROC曲线,AUC= %f'% aucValue)
plts[1].plot(fpr_log,tpr_log,'b',label='Logistics ROC曲线 AUC= %f'% auc_log)
plts[1].plot([0,1],[0,1],c='#a0a0a0',lw=2,ls='--')
#plts[1].xlim(-0.01, 1.02)  # 设置X轴的最大和最小值
# plts[1].ylim(-0.01, 1.02)  # 设置y轴的最大和最小值
# plts[1].xticks(np.arange(0, 1.1, 0.1))
# plts[1].yticks(np.arange(0, 1.1, 0.1))
plts[1].grid(b=True, ls=':')
plts[1].legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
#plts[1].xlim(0,1)
#auc(x_test,y_test)

# plt.show()
# plt.figure( facecolor='w')
# plt.plot(fpr,tpr,c='r',lw=2,label=u'KNN算法,AUC=%.3f' % aucValue)
# plt.plot(fpr_log,tpr_log,c='g',lw=2,label=u'Logistics算法,AUC=%.3f' % auc_log)
# plt.plot((0,1),(0,1),c='#a0a0a0',lw=2,ls='--')
# # plt.xlim(-0.01, 1.02)#设置X轴的最大和最小值
# # plt.ylim(-0.01, 1.02)#设置y轴的最大和最小值
# # plt.xticks(np.arange(0, 1.1, 0.1))
# # plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xlabel('False Positive Rate(FPR)', fontsize=16)
# plt.ylabel('True Positive Rate(TPR)', fontsize=16)
# plt.grid(b=True, ls=':')
# plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
# plt.title(u'鸢尾花数据Logistic和KNN算法的ROC/AUC', fontsize=18)
# plt.show()