import numpy as np
import matplotlib.pyplot as plt
import svm

np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, .5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

testSVM = svm.SVM(classA, classB)

# Separate the support vectors:
plotting_classA = []
plotting_classB = []
plotting_classA_SV = []
plotting_classB_SV = []
for i, entry_i in enumerate(classA):
	sv = False
	for j, entry_j in enumerate(testSVM.non_zero_alpha):
		if np.array_equal(entry_i[0], entry_j[0][0]) & np.array_equal(entry_i[1], entry_j[0][1]):
			plotting_classA_SV.append(entry_i)
			sv = True
			break
	if not sv:
		plotting_classA.append(entry_i)

for i, entry_i in enumerate(classB):
	sv = False
	for j, entry_j in enumerate(testSVM.non_zero_alpha):
		if np.array_equal(entry_i[0], entry_j[0][0]) & np.array_equal(entry_i[1], entry_j[0][1]):
			plotting_classB_SV.append(entry_i)
			sv = True
			break
	if not sv:
		plotting_classB.append(entry_i)

# Plotting of datapoints:
plt.plot([p[0] for p in plotting_classA],
		[p[1] for p in plotting_classA],
		'b.')
plt.plot([p[0] for p in plotting_classB],
		[p[1] for p in plotting_classB],
		'r.')

# Plotting of support vectors:
plt.plot([p[0] for p in plotting_classA_SV],
		[p[1] for p in plotting_classA_SV],
		'b+', markersize=12)
plt.plot([p[0] for p in plotting_classB_SV],
		[p[1] for p in plotting_classB_SV],
		'r+', markersize=12)

plt.axis('equal')  # Force same scale on both axes

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[testSVM.indicator(x, y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, 0, colors='black', linewidths=1)
plt.contour(xgrid, ygrid, grid, (-1, 1), colors='green', linewidths=1, linestyles='dashed')

plt.savefig('svmplot.pdf')  # Save a copy in a file
plt.show()  # Show the plot on the screen
