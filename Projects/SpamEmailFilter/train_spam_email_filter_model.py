# _*_ codig utf8 _*_
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,precision_score,accuracy_score
#1.加载处理院会数据
emails_train = np.load('../datas/processedemails1.npy')
emails_label_train = np.load('../datas/processedemaillabels.npy')
emails_test = np.load('../datas/processedemails_test.npy')
emails_label_test = np.load('../datas/processedemaillabels_test.npy')

# print(emails_train[0:5])
# print(emails_test[0:5])

#2 特征提取
#3 特征工程
#4 模型训练
def benchmark(y_true,y_pred,datasettype,alg):
    acc_score_train = accuracy_score(y_true, y_pred)
    print('%s: Accuracy score in %s set is %f' % (alg,datasettype,acc_score_train))

    prec_score_train = precision_score(y_true, y_pred)
    print('%s: Precision score in %s set is %f' % (alg,datasettype,prec_score_train))

    report_train = classification_report(y_true, y_pred)
    print('{}: Classification Report on {} set:\n{}'.format(alg,datasettype,report_train))

    matrix = confusion_matrix(y_true,y_pred)
    print('{}: Confusion matrix on {} set:\n{}'.format(alg, datasettype, matrix))

guass = GaussianNB()
guass.fit(emails_train,emails_label_train)
predict_train = guass.predict(emails_train)
predict_test = guass.predict(emails_test)
benchmark(emails_label_train,predict_train,'train','GaussianNB')
benchmark(emails_label_test,predict_test,'test','GaussianNB')

bernoulli = BernoulliNB(alpha=1.0)
bernoulli.fit(emails_train,emails_label_train)
predict_train = bernoulli.predict(emails_train)
predict_test = bernoulli.predict(emails_test)

benchmark(emails_label_train,predict_train,'train','BernoulliNB')
benchmark(emails_label_test,predict_test,'test','BernoulliNB')

logic = LogisticRegression(penalty='l2')
logic.fit(emails_train,emails_label_train)
predict_train = logic.predict(emails_train)
predict_test = logic.predict(emails_test)
benchmark(emails_label_train,predict_train,'train','LogisticRegression')
benchmark(emails_label_test,predict_test,'test','LogisticRegression')

# multi = MultinomialNB(alpha=1.0)
# multi.fit(emails_train,emails_label_train)
# predict_train = multi.predict(emails_train)
# predict_test = multi.predict(emails_test)
# benchmark(emails_label_train,predict_train,'train','MultinomialNB')
# benchmark(emails_label_test,predict_test,'test','MultinomialNB')