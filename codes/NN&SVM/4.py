import numpy as np
import math
import random
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


def parse(filename):
	file = open(filename)
	d = []
	for line in file:
		tmp = line.strip().split(' ')
		l = []
		for i in range(1,len(tmp)):
			l.append(float(tmp[i]))
		y = 1
		if float(tmp[0]) != 1:
			y = -1
		d.append([y,l])
	file.close()
	return d

def symmetry(l):
	reform = []
	for i in range(16):
		tmp = []
		for j in range(16):
			tmp.append(l[i*16+j])
		reform.append(tmp)
	s = 0
	for i in range(16):
		for j in range(8):
			s += abs(reform[i][j] - reform[i][15-j])
	return -s/128

def density(l):
	return sum(l)/len(l)

def convert(Data):
	reform = []
	d1 = [[],[]]
	dn1 = [[],[]]
	minsym = 100
	maxsym = -100
	minden = 100
	maxden = -100
	for i in range(len(Data)):
		l = Data[i][1]
		sym = symmetry(l)
		den = density(l)
		minsym = min(sym,minsym)
		minden = min(den,minden)
		maxsym = max(sym,maxsym)
		maxden = max(den,maxden)
		tmpx = np.array([den,sym])
		tmpy = 2*(Data[i][0] == 1) - 1
		reform.append([tmpx,tmpy])

	a1 = 2/(maxsym - minsym)
	a2 = 2/(maxden - minden)
	b1 = 1 - a1*maxsym
	b2 = 1 - a2*maxden

	for i in range(len(reform)):
		reform[i][0][0] = a2 * reform[i][0][0] + b2
		reform[i][0][1] = a1 * reform[i][0][1] + b1
		if reform[i][1] == 1:
			d1[0].append(reform[i][0][0])
			d1[1].append(reform[i][0][1])
		else:
			dn1[0].append(reform[i][0][0])
			dn1[1].append(reform[i][0][1])

	return reform,d1,dn1


def Kernel(x1,x2):
	tmp = x1.dot(x2)
	s = 1
	for i in range(8):
		s += tmp**(i+1)

	return s

def sumK(alpha,Data,xs,C):
	s = 0
	for i in range(300):
		if alpha[i] < 0:
			continue
		xi = Data[i][0]
		yi = Data[i][1]
		a = alpha[i]
		a = min(a,C)
		s += a*yi*Kernel(xi,xs)
	return s

D1 = parse('ZipDigits.test')
D2 = parse('ZipDigits.train')
D = D1 + D2
D,D1,Dn1 = convert(D)

random.shuffle(D)
Dtrain = D[:300]
Dtest = D[300:]

Q = np.zeros((300,300))
A = np.zeros((302,300))

for i in range(300):
	xi = Dtrain[i][0]
	yi = Dtrain[i][1]
	for j in range(300):
		xj = Dtrain[j][0]
		yj = Dtrain[j][1]
		Q[i][j] = yi*yj*Kernel(xi,xj)
	A[i+2][i] = 1
	A[0][i] = yi
	A[1][i] = -yi


p = -np.ones(300)
c = np.zeros(302)

Qmat = matrix(Q, tc='d')
pmat = matrix(p, tc='d')
Amat = -matrix(A, tc='d') # multiply by -1 to change >= to <=
cmat = matrix(c, tc='d')
sol=solvers.qp(Qmat, pmat, Amat, cmat)

C = 0.1

alpha = sol['x']
index = -1
for i in range(300):
	if alpha[i] > 0 and alpha[i] < C:
		index = i
		break

xs = Dtrain[i][0]
ys = Dtrain[i][1]
b = ys - sumK(alpha,Dtrain,xs,C)

x1 = np.arange(-1,1.01,0.01)
x2 = np.arange(-1,1.01,0.01)

a = [[],[]]
d = [[],[]]

t = 1
for i in x1:
	for j in x2:
		xt = [i,j]
		yh = sumK(alpha,Dtrain,xt,C) + b
		if yh > 0:
			a[0].append(i)
			a[1].append(j)
		else:
			d[0].append(i)
			d[1].append(j)
		print('t =',t)
		t += 1

N = len(Dtest)
Etest = 0
for i in range(N):
	xn = Dtest[i][0]
	yn = Dtest[i][1]
	yhat = sumK(alpha,Dtrain,xn,C) + b
	if yhat*yn < 0:
		Etest += 1
	print(i)
Etest /= N
print(Etest)

c1 = [1,0.7,0.7]
c2 = [0.7,0.7,1]

plt.scatter(a[0],a[1],10,color = c1, label = 'g(x) = +1')
plt.scatter(d[0],d[1],10,color = c2,label = 'g(x) = -1')

plt.scatter(D1[0],D1[1],1,color = 'r',marker ='.',label = 'Digit 1')
plt.scatter(Dn1[0],Dn1[1],1,color = 'b',marker ='.',label = 'not Digit 1')

tit = 'C = %.2f'%(C)
plt.xlabel('Density')
plt.ylabel('Symmetry')
plt.legend()
plt.title(tit)
plt.show()