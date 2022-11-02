import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random
import math

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
	d11 = []
	d12 = []
	dn11 = []
	dn12 = []
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
		reform.append([[den,sym],Data[i][0]])

	a1 = 2/(maxsym - minsym)
	a2 = 2/(maxden - minden)
	b1 = 1 - a1*maxsym
	b2 = 1 - a2*maxden

	for i in range(len(reform)):
		reform[i][0][0] = a2 * reform[i][0][0] + b2
		reform[i][0][1] = a1 * reform[i][0][1] + b1
		if reform[i][1] == 1:
			d11.append(reform[i][0][0])
			d12.append(reform[i][0][1])
		else:
			dn11.append(reform[i][0][0])
			dn12.append(reform[i][0][1])

	return reform,[d11,d12],[dn11,dn12]

def Gaussian(z):
	a = - 0.5 * (z ** 2)
	return math.exp(a)

def d(a,b):
	a = np.array(a)
	b = np.array(b)
	return np.linalg.norm(a - b)

def RBF(k,Data,x):
	r = 2/math.sqrt(k)
	phy = []
	for i in range(len(Data)):
		z = calz(x,Data[i][0],r)
		phy.append(Gaussian(z))
	sumphy = sum(phy)

	g = 0
	for i in range(len(Data)):
		w = Data[i][1]/sumphy
		g += w*phy[i]

	if g >= 0: return 1
	return -1

def step(c,D):
	maxdis = 0 
	maxind = -1
	for i in range(len(D)):
		tmp = 2
		for j in range(len(c)):
			tmp = min(d(D[i],c[j]),tmp)
		if tmp > maxdis:
			maxdis = tmp
			maxind = i
	return maxind

def cluster(c,D):
	clus = []
	for i in range(len(c)):
		clus.append([])
	for i in range(len(D)):
		dis = 2
		index = -1
		for j in range(len(c)):
			tmp = d(D[i],c[j])
			if tmp < dis:
				dis = tmp
				index = j
		clus[index].append(i)
	return clus

def kcenters(Dx,k):
	centers = [random.choice(Dx)]
	for i in range(k-1):
		ncind = step(centers,X) 
		centers.append(X[ncind])
	return centers

def refine(center,Dx,k):
	for i in range(3):
		S = cluster(center,Dx)
		for j in range(k):
			sum1 = 0
			sum2 = 0
			for n in range(len(S[j])):
				index = S[j][n]
				sum1 += Dx[index][0]
				sum2 += Dx[index][1]
			sum1 = sum1 / len(S[j])
			sum2 = sum2 / len(S[j])
			center[j][0] = sum1
			center[j][1] = sum2

def formZl(a, center,r):
	zline = []
	for i in range(k):
		dis = d(a,center[i])
		zline.append(Gaussian(dis/r))
	return zline

def formZ(Dx,center,r):
	z = []
	for i in range(len(Dx)):
		xn = Dx[i]
		z.append(formZl(xn,center,r))
	return z

def calw(rz,ry):
	zh = np.array(rz)
	yh = np.array(ry)
	M = zh.T.dot(zh)
	b = zh.T.dot(yh)
	return inv(M).dot(b)

D1 = parse('ZipDigits.test')
D2 = parse('ZipDigits.train')
D = D1 + D2
D,D1,Dn1 = convert(D)


random.shuffle(D)
Dtrain = D[:300]
Dtest = D[300:]

X = []
y = []
for i in range(300):
	X.append(Dtrain[i][0])
	y.append(Dtrain[i][1])

n = 50
K = []
E = []
minid = 0
for i in range(n):
	K.append(i+1)

for i in range(n):
	k = K[i]
	r = 2/math.sqrt(k)
	c = kcenters(X,k)
	#refine(c,X,k)
	Z = formZ(X,c,r)
	e = 0
	for j in range(300):
		zn = Z[j]
		yn = y[j]
		yd = y[:j] + y[j+1:]
		Zd = Z[:j] + Z[j+1:]
		w = calw(Zd,yd)
		ys = w.T.dot(zn)
		if ys * yn < 0:
			e += 1
	e /= 300
	E.append(e)
	if E[i] < E[minid]:
		minid = i 

	print('k =',k,'e =',e)

k = K[minid]
Ecv = E[minid] 

Ko = K[:minid] + K[minid+1:]
Eo = E[:minid] + E[minid+1:]

lab = 'optimal k = {} with Ecv = {:.4f}%'.format(k,Ecv*100)
plt.figure(1)
plt.bar(Ko,Eo)
plt.bar(k,Ecv,color = 'r',label = lab)
plt.legend()
plt.title('k vs. Ecv')
plt.xlabel('k')
plt.ylabel('Ecv')


r = 2/math.sqrt(k)
c = kcenters(X,k)
Z = formZ(X,c,r)
w = calw(Z,y)

Ein = 0
for i in range(300):
	xn = Dtrain[i][0]
	yn = Dtrain[i][1]
	zn = formZl(xn,c,r)
	ys = w.T.dot(zn)
	if ys * yn < 0:
		Ein += 1
	print('i =',i)
Ein /= 300


Etest = 0
for i in range(len(Dtest)):
	xn = Dtest[i][0]
	yn = Dtest[i][1]
	zn = formZl(xn,c,r)
	ys = w.T.dot(zn)
	if ys * yn < 0:
		Etest += 1
	print('i =',i)
Etest /= len(Dtest)

x1 = np.arange(-1,1.01,0.01)
x2 = np.arange(-1,1.01,0.01)

# y = +1
a1 = []
a2 = []

# y = -1
b1 = []
b2 = []

t = 1
for i in x1:
	for j in x2:
		xn = [i,j]
		zn = formZl(xn,c,r)
		ys = w.T.dot(zn)
		if ys > 0:
			a1.append(i)
			a2.append(j)
		else:
			b1.append(i)
			b2.append(j)
		print('t =',t)
		t += 1

c1 = [1,0.7,0.7]
c2 = [0.7,0.7,1]

tit = 'k = {:<3d} Etest = {:.4f}%\n          Ein = {:.4f}%'.format(k,Etest*100,Ein*100)
plt.figure(2)
plt.scatter(a1,a2,10,color = c1, label = 'g(x) = +1')
plt.scatter(b1,b2,10,color = c2,label = 'g(x) = -1')

plt.scatter(D1[0],D1[1],1,color = 'r',marker ='.',label = 'Digit 1')
plt.scatter(Dn1[0],Dn1[1],1,color = 'b',marker ='.',label = 'not Digit 1')

plt.xlabel('Density')
plt.ylabel('Symmetry')
plt.legend()
plt.title(tit)
plt.show()

