import numpy as np
from cvxopt import matrix, solvers

def convertz(x):
	z = np.zeros(2)
	z[0] = x[0]**3 - x[1]
	z[1] = x[0]*x[1]
	return z

def k(x1,x2):
	z1 = convertz(x1)
	z2 = convertz(x2)
	return z1.dot(z2)

x = np.array([[1,0],[-1,0]])
y = np.array([1,-1])


Q = np.zeros((2,2))
A = np.zeros((4,2))
for i in range(2):
	for j in range(2):
		Q[i][j] = y[i]*y[j]*k(x[i],x[j])
	A[i+2][i] = 1
A[0][:] = y[:]
A[1][:] = -y[:]

p = -np.ones(2)
c = np.zeros(4)

Qmat = matrix(Q, tc='d')
pmat = matrix(p, tc='d')
Amat = -1*matrix(A, tc='d') # multiply by -1 to change >= to <=
cmat = matrix(c, tc='d')
sol=solvers.qp(Qmat, pmat, Amat, cmat)

print(Q)
print(A)
alpha = sol['x']
w = alpha[0]*y[0]*x[0]
w += alpha[1]*y[1]*x[1]
b = y[1] - w.dot(x[1])

print('\n\n  [{:>4},{:>4},{:>4}] \n= [{:>4.2f},{:>4.2f},{:>4.2f}]'.format('b','w1','w2',b,w[0],w[1]))
