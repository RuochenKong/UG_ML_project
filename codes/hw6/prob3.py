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
N = 2000
n = 30
Sep = np.arange(0.2,5.2,0.2)

T=np.array([])

for sep in Sep:
	t1 = 0
	for i in range(n):
		print(i)
		X, y = generatedata(rad, thk, sep, N)
		X_treat = np.c_[np.ones(N), X]
		t, last, w = PLA(X_treat, y)
		t1 += t
	T = np.append(T, t1 / n)
    
plt.plot(Sep, T, c = 'purple')
plt.title('sep vs. # of Itr')
plt.xlabel('sep')
plt.ylabel('# of Itr')
plt.show()