import math
import numpy as np

def tanh(x):
	r = np.zeros((len(x),1))
	for i in range(len(x)):
		r[i] = math.tanh(x[i])
	return r

def dtanh(x):
	d = len(x)
	one = np.ones((d,1))
	return one - np.multiply(x,x)

def id(x):
	return x

def did(x):
	d = len(x)
	return np.ones((d,1))

def forward(x0,W,theta):
	d = len(x0)
	x = [x0]
	s = []
	for i in range(1,L+1):
		sl = (W[i].T).dot(x[i-1])
		s.append(sl)
		if i == L:
			ts = theta(sl)
		else:
			ts = tanh(sl)
		xl = np.concatenate(([[1]],ts))
		x.append(xl)
	return x,s

def back(x,s,W,y,dtheta):
	d = [[]]*(L+1)
	dL = 2*(x[L][1] - y)*(dtheta(x[L][1]))
	d[L] = dL
	for i in range(L-1,0,-1):
		r = (W[i+1].dot(d[i+1]))[1:dim[i]+1]
		dl = np.multiply(dtanh(x[i])[1:dim[i]+1],r)
		d[i] = dl 
	return d

def printr(G,type):
	print('For '+type+':')
	print('G(1):')
	print(G[1]) 
	print()
	print('G(2):')
	print(G[2]) 
	print()


#set up
m = 2
L = 2
dim = [2,m,1]
w0 = 0.25

x1 = np.array([[1],[1],[2]])
y = 1
w1 = w0*np.ones((3,m))
w2 = w0*np.ones((m+1,1))
w = [[],w1,w2]

#problem 1a
x,s = forward(x1,w,tanh)
d = back(x,s,w,y,dtanh)

Ge = [[]] * 3
Ge[1] = x[0].dot(d[1].T)
Ge[2] = x[1].dot(d[2].T)
print('Problem 1a')
printr(Ge,'tanh')
print('-'*30)	

x,s = forward(x1,w,id)
d = back(x,s,w,y,did)

Ge = [[]] * 3
Ge[1] = x[0].dot(d[1].T)
Ge[2] = x[1].dot(d[2].T)
printr(Ge,'identity')

print('='*30)
print('Problem 1b: Numerical Gradient')

# For tanh, w1
Gn1 = np.zeros((3,m))
for i in range(3):
	for j in range(m):
		w1[i][j] = 0.2501
		x,s = forward(x1,w,tanh)
		d = back(x,s,w,y,dtanh)
		eplus = (x[-1][1] - y)**2
		w1[i][j] = 0.2499
		x,s = forward(x1,w,tanh)
		d = back(x,s,w,y,dtanh)
		eminus = (x[-1][1] - y)**2
		Gn1[i][j] = (eplus-eminus)/0.0002
		w1[i][j] = 0.25

# For tanh, w2
Gn2 = np.zeros((m+1,1))
for i in range(m+1):
	w2[i] = 0.2501
	x,s = forward(x1,w,tanh)
	d = back(x,s,w,y,dtanh)
	eplus = (x[-1][1] - y)**2
	w2[i] = 0.2499
	x,s = forward(x1,w,tanh)
	d = back(x,s,w,y,dtanh)
	eminus = (x[-1][1] - y)**2
	Gn2[i] = (eplus-eminus)/0.0002
	w2[i] = 0.25
printr([[],Gn1,Gn2],'tanh')
print('-'*30)

# For identify, w1
Gn1 = np.zeros((3,m))
for i in range(3):
	for j in range(m):
		w1[i][j] = 0.2501
		x,s = forward(x1,w,id)
		d = back(x,s,w,y,did)
		eplus = (x[-1][1] - y)**2
		w1[i][j] = 0.2499
		x,s = forward(x1,w,id)
		d = back(x,s,w,y,did)
		eminus = (x[-1][1] - y)**2
		Gn1[i][j] = (eplus-eminus)/0.0002
		w1[i][j] = 0.25

# For identify, w2
Gn2 = np.zeros((m+1,1))
for i in range(m+1):
	w2[i] = 0.2501
	x,s = forward(x1,w,id)
	d = back(x,s,w,y,did)
	eplus = (x[-1][1] - y)**2
	w2[i] = 0.2499
	x,s = forward(x1,w,id)
	d = back(x,s,w,y,did)
	eminus = (x[-1][1] - y)**2
	Gn2[i] = (eplus-eminus)/0.0002
	w2[i] = 0.25

printr([[],Gn1,Gn2],'identify')