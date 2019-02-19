import numpy , random , math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt

numpy.random.seed(100)
classA = numpy.concatenate ( 
	(numpy.random.randn(10, 2) ∗ 0.2 + [1.5, 0.5],
	 numpy.random.randn(10, 2) ∗ 0.2 + [−1.5, 0.5]))
classB = numpy.random.randn(20, 2) ∗ 0.2 + [0.0, −0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
	(numpy.ones(classA.shape[0]),
	 −numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


def kerlel_linear():
	return 

def objective(alphas):
	ret = 0
	for i in range(len(alphas)):
		for j in range(len(alphas)):
			ret = ret + alphas[i] * alphas[j] 
				  * targets[i] * targets[j]

	ret = ret / 2 - sum (alphas)
	return ret

start = numpy.zeros(N) #Initial guess of the aöpha-vector

ret = minimize (objective, start, bounds=B, constraints=XC)
alpha = ret [’x’]





plt.plot([p[0] for p in classA],
		 [p[1] for p in classA],
		 ’b. ’)
plt.plot([p[0] for p in classB],
		 [p[1] for p in classB],
		 ’r. ’)
plt.axis(’equal’) #Force same scale on both axes
plt.savefig(’svmplot.pdf’) #Save a copy in a file
plt.show() #Show the plot on the screen