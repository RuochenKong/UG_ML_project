import numpy as np
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

def d(a,b):
	a = np.array(a)
	b = np.array(b)
	return np.linalg.norm(a - b)

def sortByD(x,data):
	xs = []
	for i in data:
		dis = d(x,i[0])
		xs.append((dis,i[0],i[1]))
	xs.sort()
	return xs

def knn(k,sD):
	s = 0
	for i in range(k):
		s += sD[i][2]
	return s

D1 = parse('ZipDigits.test')
D2 = parse('ZipDigits.train')
D = D1 + D2
D,D1,Dn1 = convert(D)

random.shuffle(D)
Dtrain = D[:300]
Dtest = D[300:]

K = []
E = []
n = 149
for i in range(n):
	E.append(0)
	k = 2*i + 1
	K.append(k)

for i in range(300):
	xn = Dtrain[i][0]
	yn = Dtrain[i][1]
	Dp = Dtrain[:i] + Dtrain[i+1:]
	Dnew = sortByD(xn,Dp)
	for j in range(n):
		ys = knn(K[j],Dnew)
		if ys*yn < 0:
			E[j] = E[j] + 1
	print('n = %d'%(i))

minid = 0
for i in range(n):
	E[i] = E[i]/300
	if E[i] < E[minid]:
		minid = i


minid = 0
for i in range(n):
	if E[i] < E[minid]:
		minid = i

k = K[minid]
Emin = E[minid]

Ko = K[:minid] + K[minid+1:]
Eo = E[:minid] + E[minid+1:]

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
		Dnew = sortByD([i,j],Dtrain)
		ys = knn(k,Dnew)
		if ys > 0:
			a1.append(i)
			a2.append(j)
		else:
			b1.append(i)
			b2.append(j)
		print(t)
		t += 1

c1 = [1,0.7,0.7]
c2 = [0.7,0.7,1]


Etest = 0
for i in range(len(Dtest)):
	xn = Dtest[i][0]
	yn = Dtest[i][1]
	Dnew = sortByD(xn,Dtrain)
	ys = knn(k,Dnew)
	if ys*yn < 0:
		Etest += 1
	print('i =',i)
Etest = Etest/len(Dtest)


Ein = 0
for i in range(len(Dtrain)):
	xn = Dtrain[i][0]
	yn = Dtrain[i][1]
	Dnew = sortByD(xn,Dtrain)
	ys = knn(k,Dnew)
	if ys*yn < 0:
		Ein += 1
	print('i =',i)
Ein /= 300


lab = 'optimal k = {} with Ecv = {:.4f}%'.format(k,Emin*100)
plt.figure(1)
plt.bar(Ko,Eo)
plt.bar(k,Emin,color = 'r',label = lab)
plt.legend()
plt.title('k vs. Ecv')
plt.xlabel('k')
plt.ylabel('Ecv')

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