# _*_ codig utf8 _*_
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score

bunch_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=22)
data = bunch_data.data
target = bunch_data.target
target_names = bunch_data.target_names
print('{} - {} \n{}'.format(type(data), len(data), data[0:10]))
print('{} - {}\n{}'.format(type(target), target.shape, target[0:10]))
print('{} - {}\n{}'.format(type(target_names), len(target_names), target_names[0:10]))
print('==' * 100)
tfidf = TfidfVectorizer()
data = tfidf.fit_transform(data)
print(data.shape)
truncateSVD = TruncatedSVD(n_components=50)
data = truncateSVD.fit_transform(data)
print(data.shape)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.7, random_state=22)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
pred_train = gnb.predict(x_train)
pred_test = gnb.predict(x_test)


def benchmark(y_true, y_pred, datasettype, alg):
    acc_score_train = accuracy_score(y_true, y_pred)
    print('%s: Accuracy score in %s set is %f' % (alg, datasettype, acc_score_train))

    prec_score_train = precision_score(y_true, y_pred,average='micro')
    print('%s: Precision score in %s set is %f' % (alg, datasettype, prec_score_train))

    report_train = classification_report(y_true, y_pred)
    print('{}: Classification Report on {} set:\n{}'.format(alg, datasettype, report_train))

    matrix = confusion_matrix(y_true, y_pred)
    print('{}: Confusion matrix on {} set:\n{}'.format(alg, datasettype, matrix))

benchmark(y_train,pred_train,'train','GaussionNB')
benchmark(y_test,pred_test,'test','GaussionNB')