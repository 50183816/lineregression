# _*_ codig utf8 _*_

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'simHei']

def f(x,y):# 原函数
    return x**2 + y ** 2

def h(x):# 导函数
    return 2*x

X=[]
Y=[]
Z=[]
x=2
y=2
step=0.1
f_change=f(x,y)
f_current=f(x,y)
X.append(x)
Y.append(y)
Z.append(f_current)
while f_change > 1e-10:
    x = x - step * h(x)
    y=y-step*h(y)
    f_change = f_current-f(x,y)
    f_current = f(x,y)
    X.append(x)
    Y.append(y)
    Z.append(f_current)

print('结果为(%s,%s)' %(x,y))
print(X)
fig = plt.figure()
ax=Axes3D(fig)
X2 = np.arange(-2,2,0.2)
Y2 = np.arange(-2,2,0.2)
X2,Y2=np.meshgrid(X2,Y2)
Z2=X2**2+Y2**2
ax.plot_surface(X2,Y2,Z2,rstride=1,cstride=1,cmap='rainbow')
ax.plot(X,Y,Z,'bo--')
#plt.plot(X2,Y2,'-',color='#666666')
#plt.plot(X,Y,'bo--')
ax.set_title('$y=x^2+y^2$函数求解最小值，最终解为（%.2f,%.2f,%.2f）'%(x,y,f_current))
plt.show()