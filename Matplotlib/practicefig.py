# _*_ codig utf8 _*_
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)
plt.plot(np.random.randn(30).cumsum(),'kx--')
plt.subplots_adjust(wspace=0,hspace=0)
