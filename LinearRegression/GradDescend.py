# _*_ codig utf8 _*_

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'simHei']

def f(x):# 原函数
    return x**2

def h(x):# 导函数
    return 2*x

X=[]
Y=[]

x=2
step=0.8
f_change=f(x)
f_current=f(x)
X.append(x)
Y.append(f_change)
while f_change > 1e-10:
    x = x - step * h(x)
    tmp = f(x)
    f_change = np.abs(f_current-tmp)
    f_current = tmp
    X.append(x)
    Y.append(f_current)

print('结果为(%s,%s)' %(x,f_current))

fig = plt.figure()
X2 = np.arange(-2.1,2.15,0.05)
Y2 = X2**2

plt.plot(X2,Y2,'-',color='#666666')
plt.plot(X,Y,'bo--')
plt.title('$y=x^2$函数求解最小值，最终解为（%s,%s）'%(x,f_current))
plt.show()