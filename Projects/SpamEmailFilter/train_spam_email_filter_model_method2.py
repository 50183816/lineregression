# _*_ codig utf8 _*_
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score
from time import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.加载处理院会数据
emails_train = np.load('../datas/processedemails1.npy')
emails_label_train = np.load('../datas/processedemaillabels.npy')
emails_test = np.load('../datas/processedemails_test.npy')
emails_label_test = np.load('../datas/processedemaillabels_test.npy')
extra_field_train = np.load('../datas/processedemails_extra_fileds_train.npy')
extra_field_test = np.load('../datas/processedemails_extra_fileds_test.npy')


# 2 特征提取
def extracthour(datestr):
    date_reg = r'([\d]{1,2}):[\d]{1,2}:[\d]{1,2}'
    result = re.findall(date_reg, datestr)
    if len(result) > 0:
        return int(result[0])
    return 0


def hasWeekday(date):
    weekday = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    date = date.lower()
    for w in weekday:
        if w in date:
            return 1

    return 0


# print(np.shape(extra_field_test))
# print(np.shape(emails_label_train.reshape(-1,1)))
combile_fields_labels = np.concatenate((extra_field_train, emails_label_train.reshape(-1, 1).astype(np.int)), axis=1)
df = pd.DataFrame(combile_fields_labels, columns=['from_add', 'to_add', 'send_date', 'content_sem', 'label'])

df.label = df.label.astype(np.int)
df.content_sem = df.content_sem.astype(np.float)
print('##' * 50)

from_address = df[df.label == 0].groupby(['from_add']).count().reset_index()
good_address = list(from_address['from_add'])
print(df.groupby(['from_add','label'])['label'].count())
unknowdate = df[df.send_date == 'unknown']
temp = [(len(d), d) for d in df.send_date]

tempdf = pd.DataFrame(temp, columns=['length', 'date'])
tempdf['length'] = tempdf['length'].astype(np.int)
df['date_length'] = tempdf['length']
df['hour'] = [extracthour(x) for x in df.send_date]

df['hasDate'] = [hasWeekday(d) for d in tempdf.date]
df['isgoodadd'] = [0 if fd in good_address else 1 for fd in df.from_add]
print(df.groupby(['hasDate', 'label'])['label'].count())

df = df.drop(['from_add', 'to_add', 'send_date', 'label', 'date_length', 'hour'], axis=1)
data_array_train = np.array(df)
# print(tempdf[tempdf.length])

combile_fields_labels_test = np.concatenate((extra_field_test, emails_label_test.reshape(-1, 1).astype(np.int)), axis=1)
df_test = pd.DataFrame(combile_fields_labels_test, columns=['from_add', 'to_add', 'send_date', 'content_sem', 'label'])
df_test.label = df_test.label.astype(np.int)
df_test.content_sem = df_test.content_sem.astype(np.float)
temp = [(len(d), d) for d in df_test.send_date]
tempdf = pd.DataFrame(temp, columns=['length', 'date'])
df_test['date_length'] = tempdf['length'].astype(np.int)
df_test['hour'] = [extracthour(x) for x in df_test.send_date]
df_test['hasDate'] = [hasWeekday(d) for d in tempdf.date]
df_test['isgoodadd'] = [0 if fd in good_address else 1 for fd in df_test.from_add]
df_test.drop(['from_add', 'to_add', 'send_date', 'label', 'date_length', 'hour'], axis=1, inplace=True)
data_array_test = np.array(df_test)

emails_train = np.concatenate((emails_train, data_array_train), axis=1)
emails_test = np.concatenate((emails_test, data_array_test), axis=1)


# count_label = df.groupby(by=['content_sem', 'label'], sort=True)['label'].agg(['count']).reset_index()
# print(count_label.dtypes)
# print(count.sort_values(ascending=False))
# print(count_label[count_label.label == '0']['content_len'])

# fig = plt.figure()
# plt.plot(df3['content_len'], df3['c1_range'], label=u'垃圾邮件比例')
# plt.plot(df3['content_len'], df3['c2_range'], label=u'正常邮件比例')
# plt.grid(True)
# plt.legend(loc = 0)
# label0_x = count_label[count_label.label == '0']['content_len']
# label0_y=count_label[count_label.label == '0']['count']
# plt.xticks(np.arange(15),label0_x)
# plt.bar(label0_x, label0_y, color='r',width=0.5)
# label1_x = count_label[count_label.label == '1']['content_len']
# label1_y=count_label[count_label.label == '1']['count']
# plt.bar(label1_x, label1_y, color='g',width=0.3)
# plt.show()
# 3 特征工程
# 4 模型训练
def benchmark(y_true, y_pred, datasettype, alg):
    acc_score_train = accuracy_score(y_true, y_pred)
    print('%s: Accuracy score in %s set is %f' % (alg, datasettype, acc_score_train))

    prec_score_train = precision_score(y_true, y_pred)
    print('%s: Precision score in %s set is %f' % (alg, datasettype, prec_score_train))

    report_train = classification_report(y_true, y_pred)
    print('{}: Classification Report on {} set:\n{}'.format(alg, datasettype, report_train))

    matrix = confusion_matrix(y_true, y_pred)
    print('{}: Confusion matrix on {} set:\n{}'.format(alg, datasettype, matrix))


flag = True
if flag:
    # guass = GaussianNB()
    # guass.fit(emails_train, emails_label_train)
    # predict_train = guass.predict(emails_train)
    # predict_test = guass.predict(emails_test)
    # benchmark(emails_label_train, predict_train, 'train', 'GaussianNB')
    # benchmark(emails_label_test, predict_test, 'test', 'GaussianNB')
    # print(np.shape(emails_train))
    # print(emails_train[0:10])
    bernoulli = BernoulliNB(alpha=1.0, binarize=0.0005)
    bernoulli.fit(emails_train, emails_label_train)
    predict_train = bernoulli.predict(emails_train)
    predict_test = bernoulli.predict(emails_test)

    benchmark(emails_label_train, predict_train, 'train', 'BernoulliNB')
    benchmark(emails_label_test, predict_test, 'test', 'BernoulliNB')
    parameters={'alpha':[1.0,0.5,0],'binarize':[0.0005,0.005,0.05,0.5,1]}
    gridSearch = GridSearchCV(estimator=bernoulli,param_grid=parameters)
    gridSearch.fit(emails_train,emails_label_train)
    print('best parameters:{}, best score is {}'.format(gridSearch.best_params_,gridSearch.best_score_))
    estimater = gridSearch.best_estimator_
    predict_test = estimater.predict(emails_test)
    benchmark(emails_label_test, predict_test, 'test', 'BernoulliNB')
    # logic = LogisticRegression(penalty='l2')
    # logic.fit(emails_train, emails_label_train)
    # predict_train = logic.predict(emails_train)
    # predict_test = logic.predict(emails_test)
    # benchmark(emails_label_train, predict_train, 'train', 'LogisticRegression')
    # benchmark(emails_label_test, predict_test, 'test', 'LogisticRegression')
