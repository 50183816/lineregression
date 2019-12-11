# _*_ codig utf8 _*_
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import warnings

warnings.filterwarnings('ignore')
class ExtractDataFromEmail(object):
    def __init__(self):
        pass
    def process_conent_length(self,len):
        if len <= 10:
            return 0
        elif len <= 100:
            return 1
        elif len <= 500:
            return 2
        elif len <= 1000:
            return 3
        elif len <= 1500:
            return 4
        elif len <= 2000:
            return 5
        elif len <= 2500:
            return 6
        elif len <= 3000:
            return 7
        elif len <= 4000:
            return 8
        elif len <= 5000:
            return 9
        elif len <= 10000:
            return 10
        elif len <= 20000:
            return 11
        elif len <= 30000:
            return 12
        elif len <= 50000:
            return 13
        else:
            return 14

    def process_content_sema(self,x):
        if x > 10000:
            return 0.5 / np.exp(np.log10(x) - np.log10(500)) + np.log(abs(x - 500) + 1) - np.log(abs(x - 10000)) + 1
        else:
            return 0.5 / np.exp(np.log10(x) - np.log10(500)) + np.log(abs(x - 500) + 1) + 1

    def transform(self,content):
        if content is None or len(content)<1 :
            return []
        #extract from
        re_exp_from = r'From:[\s\S]+?[<]{0,1}\w+([-+.]\w+)*@(\w+([-.]\w+)*\.\w+([-.]\w+)*)[>]{0,1}'
        result = re.findall(re_exp_from,content)
        fromemail = 'unknown'
        # print(result)
        if len(result)>0 and len(result[0]) == 4:
            fromemail = result[0][1].strip()
        # else:
        #     print(content)
        #     raise Exception('unmkown from email')
        re_exp_to = r'[\s]+To: \w+([-+.]\w+)*@(\w+([-.]\w+)*\.\w+([-.]\w+)*)'
        result = re.findall(re_exp_to,content)
        toemail = 'unknow'
        # print(result)
        if len(result)>0 and len(result[0]) == 4:
            toemail = result[0][1].strip()

        re_exp_date = r'Date:([\s\S]*?)\n(\.)*'
        result = re.findall(re_exp_date, content)
        date = 'unknown'
        # print(result)
        if len(result)>0 and len(result[-1]) == 2:
            content_date=result[-1][0].strip()
            if len(content_date) > 10:
                date = content_date.replace('(CST)','').replace('(EDT)','').replace('(PDT)','').strip()
                if len(content_date) == 80:
                    print(result)

        re_exp_content = r'(\s+)\n([\s\S]*)'
        result = re.findall(re_exp_content, content)
        emailbody = 'unknown'
        # print(result)
        fresult = [fromemail, toemail, date,self.process_content_sema(len(content))]

        # if len(result)>0 and len(result[0]) == 2:
        #     emailbody = result[0][1].strip()
        #     ebs = emailbody.split('\n')
        #     datas = []
        #     for line in ebs:
        #         tokens = ' '.join(filter(lambda word: len(word.strip()) > 1, jieba.cut(line)))
        #         #tokens = jieba.cut(emailbody)
        #         print(tokens)
        #         datas.append(tokens)
        #     tfidf = TfidfVectorizer()
        #     tokens = tfidf.fit_transform(datas)
        #     print(tokens)
        #     tsvd = TruncatedSVD(n_components=20)
        #     tokens = tsvd.fit_transform(tokens)
        #     print(np.shape(tokens))
        #     fresult.extend(tokens)
        return fresult


if __name__ == '__main__':
    extractor = ExtractDataFromEmail()
    result = extractor.transform('''
Received: from web15010.mail.cnb.yahoo.com (web15010.mail.cnb.yahoo.com [202.165.103.67])
	by spam-gw.ccert.edu.cn (MIMEDefang) with ESMTP id j8RAWWEo019333
	for <ling@ccert.edu.cn>; Tue, 27 Sep 2005 18:24:59 +0800 (CST)
Received: (qmail 79056 invoked by uid 60001); Tue, 27 Sep 2005 10:36:09 -0000
DomainKey-Signature: a=rsa-sha1; q=dns; c=nofws;
  s=s1024; d=yahoo.com.cn;
  h=Message-ID:Received:Date:From:Subject:To:In-Reply-To:MIME-Version:Content-Type:Content-Transfer-Encoding;
  b=nTJZi7ynGmEkIhpUoulGfmq4KtTII+DyAgkSLu7rKlaEsELjU0EDod2Zj+Tq2GQsbWP/m5e0osxCZbeymqn76hJjQKkj2RNIDC1d0tqLOwGvRSFUz65eqdzh28sfrzW3v13rGSgL8T2HO5gW5ZZcsLrFkiTTeNOXs3mV+SUdGAo=  ;
Message-ID: <20050927104342.79054.qmail@web15010.mail.cnb.yahoo.com>
Received: from [61.150.43.114] by web15010.mail.cnb.yahoo.com via HTTP; Tue, 27 Sep 2005 18:36:09 CST
Date: Tue, 27 Sep 2005 18:36:09 +0800 (CST)
From: liang ming <cao@yahoo.com.cn>
Subject: =?gb2312?B?UmU6IMzW0eG1scWuyfo=?=
To: ling@ccert.edu.cn
In-Reply-To: <00dd01c5c34e$cdecf0d0$26cb6fa6@liuwu>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="0-1673022566-1127817822=:78786"
Content-Transfer-Encoding: 8bit
都想飞跃。。
     标  题: Re: 讨厌当女生
     
     嗯哪
     新东方
     555
     : 介个……
     : 以前也是这个想法
     : 你十一还上课噢 cmft
     
     --
     发信人: UK (独立小熊), 信区: Single                     c∞c c  c
     标  题: Re: sigh                                        (..) (..)
                                                             O O  O O
             laaaaaaaaf.......收捕鸟了…… patpat松松……
             先是被人家说尾巴上没毛，现在又变成了坠落的松鼠……:P
     
     
    ''')
    print(result)