import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tabulate import tabulate

np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, .5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate ((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]  # Number of rows (samples)
C = 3.0

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
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
			alph += ai * aj * targets[i] * targets[j] * kernel_linear(inputs[i], inputs[j])

	alph /= 2
	alph -= sum(alphas)
	return alph


def zerofun(alphas):
	return np.transpose(targets) @ alphas

start = np.zeros(N)  # Initial guess of the alpha-vector
B = [(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}
ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']

if ret['success']:
	print("Minimize function was successfull :)\n")
else:
	print("Minimize function was not successfull :(\n")

non_zero_alpha = []
for index, value in enumerate(alpha):
	if value > 10 ** (-5):
		non_zero_alpha.append((inputs[index], targets[index], value))

print(tabulate(non_zero_alpha, headers=['input', 'target', 'alpha']), "\n")

# Bias calculation
# First: find support vector. This "corresponds to any point with an α-value
# larger than zero, but less than C"
sv = 0
for i, entry in enumerate(non_zero_alpha):
	if entry[2] < C:
		sv = i
		break
# Now calulate bias:
bias = 0
for entry in non_zero_alpha:
	bias += entry[2] * entry[1] * kernel_linear(non_zero_alpha[sv][0], entry[0])
bias -= non_zero_alpha[sv][1]
print("Calculated bias:", bias)


def indicator(x, y):
	ind = 0
	for entry in non_zero_alpha:
		ind += entry[2] * entry[1] * kernel_linear([x, y], entry[0])
	ind -= bias
	return ind


# Plotting:
plt.plot([p[0] for p in classA],
		[p[1] for p in classA],
		'b.')
plt.plot([p[0] for p in classB],
		[p[1] for p in classB],
		'r.')
plt.axis('equal')  # Force same scale on both axes

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1, 0, 1), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.savefig('svmplot.pdf')  #  Save a copy in a file
plt.show()  # Show the plot on the screen
