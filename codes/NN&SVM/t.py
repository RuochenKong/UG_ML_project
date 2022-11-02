import numpy as np


a = np.array([1,2])
w = np.array([[0.1,0.2],[0.3,0.4]])

print(a.dot(w))
print(np.concatenate(([1],a)))
z = np.zeros(2)
print(np.multiply(a,a))
