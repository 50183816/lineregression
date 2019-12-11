# _*_ codig utf8 _*_

import numpy as np
import pandas as pd


def combine(b):
    a = np.zeros((1, 34))
    for i in b:
        a[0][i] = 1
    # print(a)
    s = '0b'
    for i in a[0]:
        s += '%d' % i
    return int(s,2)

data = pd.read_csv('./data/redball.csv',header=None)
# data = data.astype(int)
redball = data.iloc[1:,2:-1]
print(type(redball))
for index,row in redball.iterrows():
    print(row)
    d = combine(map(int,list(row)))
    print(d)