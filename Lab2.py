import numpy as np
import matplotlib.pyplot as plt
import svm

np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, .5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

testSVM = svm.SVM(classA, classB)

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
grid = np.array([[testSVM.indicator(x, y) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1, 0, 1), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.savefig('svmplot.pdf')  # Save a copy in a file
plt.show()  # Show the plot on the screen
