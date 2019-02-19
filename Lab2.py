import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(100)
classA = np.concatenate( (np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, .5]) )
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate (( classA , classB ))
targets = np.concatenate(( np.ones(classA.shape[0]), -np.ones(classB.shape[0]) ))

N = inputs.shape[0] # Number of rows (samples)
C = .03

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute , :]
targets = targets[permute]

def kernel_linear(x, y):
	return np.transpose(x) @ y

def kernel_polynomial(x, y, degree):
	return (np.transpose(x) @ y + 1) ** degree

def kernel_RBF(x, y, sigma):
	return math.exp( -np.linalg.norm(x-y, 2) / (2*(sigma**2)) )

def objective(alphas):
	alph = 0
	for i, ai in enumerate(alphas):
		for j, aj in enumerate(alphas):
			alph += ai * aj * targets[i] * targets[j] * kernel_polynomial(inputs[i], inputs[j], 4)

	alph /= 2
	alph -= sum(alphas)
	return alph

def zerofun(alpha):
	tot = 0
	for i, ai in enumerate(alpha):
		tot += i * ai
	return tot


start = np.zeros(N) #Initial guess of the alpha-vector
B = [(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}
ret = minimize (objective, start, bounds=B, constraints=XC)
alpha = ret['x']
print(alpha)





# plt.plot([p[0] for p in classA],
# 		 [p[1] for p in classA],
# 		 ’b. ’)
# plt.plot([p[0] for p in classB],
# 		 [p[1] for p in classB],
# 		 ’r. ’)
# plt.axis(’equal’) #Force same scale on both axes
# plt.savefig(’svmplot.pdf’) #Save a copy in a file
# plt.show() #Show the plot on the screen