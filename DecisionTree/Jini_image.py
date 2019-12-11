# _*_ codig utf8 _*_
import numpy as np
import matplotlib.pyplot as plt

def Jini(p):
    return 1-np.sum([np.square(pi) for pi in p])

x = np.arange(5)
y1=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
y2 = [0.2,0.3,0.4,0.1]
y3 =[0.5,0.5]
y4=[0.01,0.9,0.08,0.01]
y5=[1.0]
y=[Jini(y1),Jini(y2),Jini(y3),Jini(y4),Jini(y5)]

print(y)
# x = np.linspace(0,1,200)
# y1=[np.square(xi) for xi in x]
# x2 = np.linspace(0,1,20)
# y2 =[np.square(xi) for xi in np.linspace(0,1,20)]

plt.plot(x,y,'ro')
# plt.plot(x,y1,color='r')
# plt.plot(x,x,color='b')
# plt.plot(x2,y2,color='k')
plt.show()