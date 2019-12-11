# _*_ codig utf8 _*_

import numpy as np
import pandas as pd
# np.random.seed(22)

def random1(p):
    return 0 if np.random.random() <=p else 1

def random2(p):
    v = random1(p)
    if p == 0.5:
        return v
    if p > 0.5:
        if v == 1:
            return v
        v=random1(0.5/p)
        return v
    else:
        if v == 0:
            return v
        v = random1((0.5-p)/(1-p))
        return v

def random3(n,p):
    if n %2 == 0:
        s = int(np.ceil(np.log2(n)))
        v = ''.join(map(lambda x:str(int(x)),[random2(p) for i in  np.arange(s)]))
        print(v)
        v = int(v,2)+1
    else:
        n = n * 2
        v = random3(n,p)
        v = np.ceil(v / 2)
    if v > n:
        v = random3(n,p)
    return v

if __name__ == '__main__':
    t = [random1(0.7) for i in np.arange(100000)]
    print(sum(t))
    print(sum([random2(0.3) for i in np.arange(100000)]))
    random3(10, 0.3)
    # t = [random3(10, 0.3) for i in np.arange(10000)]
    # pdata = pd.DataFrame(t,columns=['number'])
    # print(pdata)
    # print(pdata.groupby(by=['number'])['number'].count())