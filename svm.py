import numpy as np
import random
import math
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, classA, classB, C=0.1, filename='', kernel='linear', sigma=1.0, degree=3):
        self.inputs = np.concatenate((classA, classB))
        self.targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

        N = self.inputs.shape[0]  # Number of rows (samples)
        self.filename = filename
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        # C = 3.0

        # shuffle data
        permute = list(range(N))
        random.shuffle(permute)
        self.inputs = self.inputs[permute, :]
        self.targets = self.targets[permute]

        # Calculate helper variable P (used for
        self.P = np.transpose(np.matrix(self.targets)) * self.targets
        for i in range(self.P.shape[0]):
            for j in range(self.P.shape[1]):
                self.P[i, j] *= self.kernel_caller(self.inputs[i], self.inputs[j])

        start = np.zeros(N)  # Initial guess of the alpha-vector
        B = [(0, C) for b in range(N)]
        XC = {'type': 'eq', 'fun': self.zerofun}
        # find minimum of objective function
        ret = minimize(self.objective, start, bounds=B, constraints=XC)
        alpha = ret['x']

        if ret['success']:
            print("Minimize function was successfull :)\n")
        else:
            print("Minimize function was not successfull :(\n")

        # calculate the alpha values that drive SVM
        self.non_zero_alpha = []
        for index, value in enumerate(alpha):
            if value > 10 ** (-5):
                self.non_zero_alpha.append((self.inputs[index], self.targets[index], value))

        print(tabulate(self.non_zero_alpha, headers=['input', 'target', 'alpha']), "\n")

        # Bias calculation
        # First: find support vector. This "corresponds to any point with an α-value larger than zero, but less than C"
        sv = 0
        for i, entry in enumerate(self.non_zero_alpha):
            if entry[2] < C - 10 ** (-5):
                print(entry[2], "<", C - 10 ** (-5))
                sv = i
                break

        # Now calulate bias:
        self.bias = 0
        for entry in self.non_zero_alpha:
            self.bias += entry[2] * entry[1] * self.kernel_caller(self.non_zero_alpha[sv][0], entry[0])
        self.bias -= self.non_zero_alpha[sv][1]
        print("Calculated bias:", self.bias)

        self.calc_plot_points(classA, classB)
        self.plotter(classA, classB)

    # The kernel caller function is used to specify, which kernel & kernel parameters should be used:
    def kernel_caller(self, x, y):
        if self.kernel == 'linear':
            return self.kernel_linear(x, y)
        elif self.kernel == 'polynomial':
            return self.kernel_polynomial(x, y, self.degree)
        else:
            return self.kernel_RBF(x, y, self.sigma)

    def kernel_linear(self, x, y):
        return np.transpose(x) @ y

    def kernel_polynomial(self, x, y, degree):
        return (np.transpose(x) @ y + 1) ** degree

    def kernel_RBF(self, x, y, sigma):
        return math.exp(-np.linalg.norm(x - y, 2) / (2 * (sigma ** 2)))

    def objective(self, alphas):
        alph = np.dot(self.P, np.transpose(alphas))
        alph = np.dot(alphas, np.transpose(alph))

        alph /= 2
        alph -= sum(alphas)
        return alph

    def zerofun(self, alphas):
        return np.transpose(self.targets) @ alphas

    def indicator(self, x, y):
        ind = 0
        for entry in self.non_zero_alpha:
            ind += entry[2] * entry[1] * self.kernel_caller([x, y], entry[0])
        ind -= self.bias
        return ind

    def calc_plot_points(self, classA, classB):
        self.plotting_classA = []
        self.plotting_classB = []
        self.plotting_classA_SV = []
        self.plotting_classB_SV = []
        for i, entry_i in enumerate(classA):
            sv = False
            for j, entry_j in enumerate(self.non_zero_alpha):
                if np.array_equal(entry_i[0], entry_j[0][0]) & np.array_equal(entry_i[1], entry_j[0][1]):
                    self.plotting_classA_SV.append(entry_i)
                    sv = True
                    break
            if not sv:
                self.plotting_classA.append(entry_i)

        for i, entry_i in enumerate(classB):
            sv = False
            for j, entry_j in enumerate(self.non_zero_alpha):
                if np.array_equal(entry_i[0], entry_j[0][0]) & np.array_equal(entry_i[1], entry_j[0][1]):
                    self.plotting_classB_SV.append(entry_i)
                    sv = True
                    break
            if not sv:
                self.plotting_classB.append(entry_i)

    def plotter(self, classA, classB):
        # Plotting:
        plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
        plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
        plt.plot([p[0] for p in self.plotting_classA_SV], [p[1] for p in self.plotting_classA_SV], 'b+', markersize=12)
        plt.plot([p[0] for p in self.plotting_classB_SV], [p[1] for p in self.plotting_classB_SV], 'r+', markersize=12)
        plt.axis('equal')  # Force same scale on both axes
        plt.xlabel('x1')
        plt.ylabel('x2')

        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-4, 4)
        grid = np.array([[self.indicator(x, y) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid, 0, colors='black', linewidths=1)
        plt.contour(xgrid, ygrid, grid, (-1, 1), colors='green', linewidths=1, linestyles='dashed')

        if self.filename:
            plt.savefig('figures/' + self.filename + '.jpg')  # Save a copy in a file
        plt.show()  # Show the plot on the screen
