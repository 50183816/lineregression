# _*_ codig utf8 _*_
#处理原始数据
import io
import os
import numpy as np
from extractdatafromemail import ExtractDataFromEmail
import jieba
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import time

#1. 加载标签信息
dir = '../datas/emails'
indexfile = open('../datas/emailindex','r',encoding='utf-8')
index = indexfile.readline()
file_labels = []
while index:
    parts = index.split(' ')
    if len(parts) >= 2:
        file_labels.append([parts[1].strip().lstrip('../data'),1 if parts[0]=='spam' else 0])
    index=indexfile.readline()

print('there are {} lines in index file'.format(len(file_labels)))
print(file_labels[0:10])
indexfile.close()

#邮件内容中提取特征：内容，发件箱，发件时间，收件箱
emails=[]
emails_test=[]
email_labals=[]
email_labals_test=[]
contentextracter = ExtractDataFromEmail()
t0 = time.time()
for file in file_labels:
       filepath = os.path.join(dir,file[0])
       print('Process file {}'.format(filepath))
       filepath_parts = os.path.split(filepath)
       flag = int(filepath_parts[-1])
       # print(filepath)
       emailfile = open(filepath,'r',encoding='gb2312',errors='ignore')
       content = ''.join(emailfile.readlines())
       fields = contentextracter.transform(content)
       # splittedcontent = ' '.join(filter(lambda x:len(x)>1,jieba.cut(content)))
       if flag < 200:
            emails.append(fields)
       #      email_labals.append(file[1])
       else:
           emails_test.append(fields)
       #     email_labals_test.append(file[1])
       # print(content)
       # property = contentextracter.transform(content)
       # if len(property) == 4:
       #      property.append(filepath)
       #      emails.append(property)
       emailfile.close()
t1 = time.time()
print('file process done，time used: %.2fs, start convert to vector:'%(t1-t0))
print(emails[0:10])
print(emails_test[0:10])
np.save('../datas/processedemails_extra_fileds_train',emails)
np.save('../datas/processedemails_extra_fileds_test',emails_test)
# np.save('../datas/processedemaillabels',email_labals)
# emails_test = svddecomposition.fit_transform(tfidf.fit_transform(emails_test))
# np.save('../datas/processedemails_test',emails_test)
# np.save('../datas/processedemaillabels_test',email_labals_test)
# t2 = time.time()
# print('done,converting and saving used time: %.2s'%(t2-t1))