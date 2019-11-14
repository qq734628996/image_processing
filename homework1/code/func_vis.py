import matplotlib.pyplot as plt
import numpy as np

PI=np.pi

x=np.arange(0,2*PI,0.1)
y1=np.sin(x)
y2=np.cos(x)
y3=x**2

plt.plot(x,y1,'r',label='$y=\sin x$')
plt.plot(x,y2,'g',label='$y=\cos x$')
plt.plot(x,y3,'b',label='$y=x^2$')

plt.legend()
plt.savefig('func_vis.png')
plt.show()