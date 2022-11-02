import numpy as np
import matplotlib.pyplot as plt

def convertz(x):
	z = np.zeros(2)
	z[0] = x[0]**3 - x[1]
	z[1] = x[0]*x[1]
	return z

x1 = np.array([1,0])
x2 = np.array([-1,0])

plt.scatter(1,0,color = 'r')
plt.scatter(-1,0,color = 'b')

p1z1 = np.arange(-1.5,1.51,0.01)
p1z2 = p1z1**3 # z1 = x1 ** 3 - x2 = 0

p1x2 = np.arange(min(p1z2),max(p1z2)+0.01,0.01)
p1x1 = np.zeros(len(p1x2)) # x1 = 0


plt.scatter(1,0,color = 'r')
plt.scatter(-1,0,color = 'b')
plt.plot(p1x1,p1x2,label = 'x-space')
plt.plot(p1z1,p1z2,label = 'z-space')
plt.legend()
plt.show()
