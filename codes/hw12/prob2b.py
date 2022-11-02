import numpy as np
import math
import random
import matplotlib.pyplot as plt

def tanh(x):
	r = np.zeros((len(x),1))
	for i in range(len(x)):
		r[i] = math.tanh(x[i])
	return r

def dtanh(x):
	d = len(x)
	one = np.ones((d,1))
	return one - np.multiply(x,x)

# reform the original functions
# do not need theta
def forward(x0,W):
	d = len(x0)
	x = [x0]
	s = []
	for i in range(1,L+1):
		sl = (W[i].T).dot(x[i-1])
		s.append(sl)
		if i == L:
			ts = sl
		else:
			ts = tanh(sl)
		xl = np.concatenate(([[1]],ts))
		x.append(xl)
	return x,s

def back(x,s,W,y):
	d = [[]]*(L+1)
	dL = np.array([2*(x[L][1] - y)])
	d[L] = dL

	for i in range(L-1,0,-1):
		tmp = W[i+1].dot(d[i+1])
		r = (W[i+1].dot(d[i+1]))[1:dim[i]+1]
		
		dl = np.multiply(dtanh(x[i])[1:dim[i]+1],r)
		d[i] = dl
		

	return d

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
		tmpx = np.array([[1],[den],[sym]])
		tmpy = 2*(Data[i][0] == 1) - 1
		reform.append([tmpx,tmpy])

	a1 = 2/(maxsym - minsym)
	a2 = 2/(maxden - minden)
	b1 = 1 - a1*maxsym
	b2 = 1 - a2*maxden

	for i in range(len(reform)):
		reform[i][0][1] = a2 * reform[i][0][1] + b2
		reform[i][0][2] = a1 * reform[i][0][2] + b1
		if reform[i][1] == 1:
			d1[0].append(reform[i][0][1])
			d1[1].append(reform[i][0][2])
		else:
			dn1[0].append(reform[i][0][1])
			dn1[1].append(reform[i][0][2])

	return reform,d1,dn1

def NN(Train,W,eta,lam):
	N = len(Train)
	G1 = 0*W[1]
	G2 = 0*W[2]
	ein = 0
	for i in range(N):
		xn = Train[i][0]
		yn = Train[i][1]
		rx,s = forward(xn,W)
		delta = back(rx,s,W,yn)
		ein += (rx[-1][1][0] - yn) ** 2
		G1 += rx[0].dot(delta[1].T)
		G2 += rx[1].dot(delta[2].T)
	ein /= 4*N
	G1 += 2*lam*W[1]
	G1 /= N
	G2 += 2*lam*W[2]
	G2 /= N
	return ein,[[],G1,G2]

def ErrorLam(Train,W,lam):
	N = len(Train)
	ein = 0
	for i in range(N):
		xn = Train[i][0]
		yn = Train[i][1]
		rx,s = forward(xn,W)
		ein += (rx[-1][1][0] - yn) ** 2
	ein /= 4*N

	left = 0
	for i in range(len(W[1])):
		for j in range(len(W[1][0])):
			left += W[1][i][j]**2

	for i in range(len(W[2])):
		 left += W[2][i][0]

	left = left * lam / N
	ein += left

	return ein

def Error(Train,W):
	N = len(Train)
	ein = 0
	for i in range(N):
		xn = Train[i][0]
		yn = Train[i][1]
		rx,s = forward(xn,W)
		ein += (rx[-1][1][0] - yn) ** 2
	ein /= 4*N
	return ein

D1 = parse('ZipDigits.test')
D2 = parse('ZipDigits.train')
D = D1 + D2
D,D1,Dn1 = convert(D)

random.shuffle(D)
Dtrain = D[:300]
Dtest = D[300:]

m = 10
L = 2
dim = [2,m,1]
w1 = np.zeros((3,m))
w2 = np.zeros((m+1,1))
w = [[],w1,w2]

for i in range(3):
	for j in range(m):
		w1[i][j] = random.random()

for i in range(m+1):
	w2[i] = random.random()

E = []
T = []
alpha = 1.7
beta = 0.7
eta = 0.01
t = 0
lam = 0.01/300
maxt = 10000
while t < maxt:
	e, G = NN(Dtrain,w,eta,lam)
	t += 1
	T.append(t)
	wp1 = w[1] - eta*G[1]
	wp2 = w[2] - eta*G[2]
	wp = [[],wp1,wp2]
	ep = ErrorLam(Dtrain,wp,lam)
	if ep <= e:
		w = wp
		eta = alpha*eta
	else:
		eta = beta*eta
	E.append(e)
	print(t)


plt.figure(1)
plt.loglog(T,E)

x1 = np.arange(-1,1.01,0.01)
x2 = np.arange(-1,1.01,0.01)

a = [[],[]]
b = [[],[]]

t = 1
for i in x1:
	for j in x2:
		xn = np.array([[1],[i],[j]])
		rx,s = forward(xn,w)
		if rx[-1][1] > 0:
			a[0].append(i)
			a[1].append(j)
		else:
			b[0].append(i)
			b[1].append(j)
		print('t =',t)
		t += 1

print(Error(Dtest,w))

c1 = [1,0.7,0.7]
c2 = [0.7,0.7,1]

plt.figure(2)
plt.scatter(a[0],a[1],10,color = c1, label = 'g(x) = +1')
plt.scatter(b[0],b[1],10,color = c2,label = 'g(x) = -1')

plt.scatter(D1[0],D1[1],1,color = 'r',marker ='.',label = 'Digit 1')
plt.scatter(Dn1[0],Dn1[1],1,color = 'b',marker ='.',label = 'not Digit 1')

plt.xlabel('Density')
plt.ylabel('Symmetry')
plt.legend()
plt.show()

plt.show()
