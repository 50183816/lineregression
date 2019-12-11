# _*_ codig utf8 _*_
# -- encoding:utf-8 --
"""
只要是机器学习领域，编程的流程基本和该文件中的内容一致
Create by ibf on 2018/11/4
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/household_power_consumption_1000_2.txt'
df = pd.read_csv(path, sep=';')
# df.info()

# 2. 数据清洗
# 将问号改成nan
# inplace: 该参数的含义是指是否在当前对象上直接修改，默认为False，表示在新对象上修改，原对象不改变；设置为True就表示直接在当前原始对象上修改
df.replace('?', np.nan, inplace=True)
# 删除为nan的数据
# axis：指定按照什么维度来删除数据，0表示第一维，也就是DataFrame中的行。1表示列
# how：指定进行什么样的删除操作，any表示只要出现任意一个特征属性为nan，那么就删除当前行或者当前列。all表示只有当所有的特征属性值均为nan的时候，才删除当前行或者当前列
df = df.dropna(axis=0, how='any')
# df.info()
import time
def time_format(dt):
    time_str=time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    return [time_str.tm_year,time_str.tm_mon,time_str.tm_mday,time_str.tm_hour,time_str.tm_min,time_str.tm_sec]


# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df.iloc[:, 0:2]
Y = df.iloc[:, 4].astype(np.float32)
X=X.apply(lambda row: pd.Series(time_format(row)),axis=1)
print(X)

# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
#poly = PolynomialFeatures(degree=3)
#x_train = poly.fit_transform(x_train,y_train)
#x_test = poly.fit_transform(x_test,y_test)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
"""
特征工程也就是特征数据转换，直白来讲就是将数据从A --> B
在转换的时候会基于一定的转换规则，这个转换规则你可以认为是函数，所以这个转换规则需要从训练数据中学习
NOTE: 特征工程也是需要训练的，和算法模型一样
"""
"""
sklearn中所有特征工程以及算法的API都基本一样，主要API如下：
fit：基于传入的x和y进行模型的训练
transform：使用训练好的模型参数对传入的x做一个数据的转换操作，该API一般出现在特征工程中
fit_transform: 是fit和transform两个API的组合，基于传入的x和y首先做一个模型训练操作，然后基于训练好的模型对x做一个转换操作，该API一般出现在特征工程中
predict: 使用训练好的模型对传入的特征属性x做一个模型预测的操作，得到对应的预测值，该API一般出现在算法模型中
source: 对传入的x和y使用训练好的模型进行预测，然后将预测值和实际值进行比较得到模型的评估指标，在分类算法中，该API返回准确率；在回归算法中，该API返回的是R2
"""

# StandardScaler: 对特征属性中的每一列都进行转换操作，将每列特征数据转换为服从均值为0、方差为1的高斯分布数据
# a. 创建对象
#scaler = StandardScaler()
"""
方式一：
# b. 模型训练(从训练数据中找转换函数)
scaler.fit(x_train, y_train)
# c. 使用训练好的模型对训练数据做一个转换操作
x_train = scaler.transform(x_train)
"""
# b. 直接模型训练+数据转换
#x_train = scaler.fit_transform(x_train, y_train)

# 6. 模型对象的构建
"""
fit_intercept=True, 模型是否训练截距项，默认为True，表示训练，False表示不训练
normalize=False, 在模型训练之前是否对数据做一个归一化的处理，默认表示不进行，该参数一般不改
copy_X=True, 对于训练数据是否copy一份再训练，默认是
n_jobs=1 指定使用多少个线程来训练模型，默认为1
NOTE: 除了fit_intercept之外，其它参数基本不修改
"""
#lr = LinearRegression(fit_intercept=True)

algo = Pipeline(
    steps=[
        ('poly', PolynomialFeatures(degree=10)),
        ('lr',LinearRegression(fit_intercept=True))
    ])
# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
# a. 查看模型训练好的相关参数
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['lr'].coef_))
print("截距项值:{}".format(algo.get_params()['lr'].intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
#x_test = scaler.transform(x_test)
print("模型在测试数据上的效果(R2)：{}".format(algo.score(x_test, y_test)))

# 9. 模型保存\模型持久化
"""
方式一：直接保存预测结果
方式二：将模型持久化为磁盘文件
方式三：将模型参数保存数据库
"""
# 方式二：将模型持久化为磁盘文件
#joblib.dump(scaler, './model/scaler.m')
joblib.dump(algo, './model/algo.m')

# 10. NOTE：画图看一下效果
predict_y = algo.predict(x_test)

t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', label=u'真实值')
plt.plot(t, predict_y, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.show()
