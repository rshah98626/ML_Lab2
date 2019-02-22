import numpy as np
import svm
import random
import math


# circle dataset
def generate_circle_points(radius, num_points):
	classA = np.zeros([num_points, 2])
	classB = np.zeros([num_points, 2])
	for i in range(num_points):
		p = random.random() * 2 * math.pi
		r = radius * math.sqrt(random.random())
		classA[i][0] = math.cos(p) * r
		classA[i][1] = math.sin(p) * r

	for i in range(num_points):
		# figure out how to generate points outside
		p = random.random() * 2 * math.pi
		r = (radius+1) * math.sqrt(random.random())
		classB[i][0] = math.cos(p) * r
		classB[i][1] = math.sin(p) * r

	return classA, classB


# regular dataset
np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, .5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
classA, classB = generate_circle_points(1, 40)
testSVM = svm.SVM(classA, classB, C=1, kernel='RBF', degree=3, sigma=1)



