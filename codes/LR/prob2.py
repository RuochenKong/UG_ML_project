import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def generatedata(rad, thk, sep, n, x1=0, y1=0):
	#original 1
	X1 = x1
	Y1 = y1

	#original 2
	X2 = X1 + rad + thk / 2
	Y2 = Y1 - sep
    
	Theta = np.random.uniform(0, 2*np.pi, n)
	R = np.random.uniform(rad, rad+thk, n)

	y = 2 * (Theta < np.pi) - 1
	X = np.zeros((n, 2))
	X[y > 0] = np.array([X1, Y1]) 
	X[y < 0] = np.array([X2, Y2])
	X[:, 0] += np.cos(Theta) * R
	X[:, 1] += np.sin(Theta) * R
	return X, y

def PLA(X, y, eta=1, max_step=np.inf):
	n, d = X.shape
	w = np.zeros(d)
	t = 0
	i = 0
	last = 0
	while not checkSep(X, y, w) and t < max_step:
		if np.sign(X[i, :].dot(w) * y[i]) <= 0:
			t += 1
			w += eta * y[i] * X[i, :]
			last = i

		i += 1
		if i == n:
			i = 0
	return t, last, w


def checkSep(X, y, w):
	n = X.shape[0]
	num = np.sum(X.dot(w) * y > 0)
	return num == n



rad = 10
thk = 5
sep = 5
N = 2000

X, y = generatedata(rad, thk, sep, N)
plt.figure(1)
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1, color = 'r')
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1, color = 'b')
plt.axis('square')

X_treat = np.c_[np.ones(N), X]
t, last, w = PLA(X_treat, y)

r = 2 * (rad + thk)
a1 = np.array([-r,r])
b1 = - (w[0] + w[1] * a1) / w[2]

plt.plot(a1, b1, c="purple")
plt.title('PLA')

plt.figure(2)

w1 = inv(X_treat.T.dot(X_treat)).dot(X_treat.T).dot(y)

a2 = np.array([-r,r])
b2 = - (w1[0] + w1[1] * a1) / w1[2]

plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1, color = 'r')
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1, color = 'b')
plt.axis('square')
plt.plot(a2, b2, c="purple")
plt.title('Linear Regression')
plt.show()