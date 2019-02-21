import numpy as np
import random
import math
from scipy.optimize import minimize
from tabulate import tabulate


class SVM:
    def __init__(self, classA, classB, C=3.0):
        self.inputs = np.concatenate((classA, classB))
        self.targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

        N = self.inputs.shape[0]  # Number of rows (samples)
        # C = 3.0

        # shuffle data
        permute = list(range(N))
        random.shuffle(permute)
        self.inputs = self.inputs[permute, :]
        self.targets = self.targets[permute]

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
            if entry[2] < C:
                sv = i
                break

        # Now calulate bias:
        self.bias = 0
        for entry in self.non_zero_alpha:
            self.bias += entry[2] * entry[1] * self.kernel_linear(self.non_zero_alpha[sv][0], entry[0])
        self.bias -= self.non_zero_alpha[sv][1]
        print("Calculated bias:", self.bias)

    def kernel_linear(self, x, y):
        return np.transpose(x) @ y

    def kernel_polynomial(self, x, y, degree):
        return (np.transpose(x) @ y + 1) ** degree

    def kernel_RBF(self, x, y, sigma):
        return math.exp(-np.linalg.norm(x - y, 2) / (2 * (sigma ** 2)))

    def objective(self, alphas):
        alph = 0
        for i, ai in enumerate(alphas):
            for j, aj in enumerate(alphas):
                alph += ai * aj * self.targets[i] * self.targets[j] * self.kernel_linear(self.inputs[i], self.inputs[j])
        alph /= 2
        alph -= sum(alphas)
        return alph

    def zerofun(self, alphas):
        return np.transpose(self.targets) @ alphas

    def indicator(self, x, y):
        ind = 0
        for entry in self.non_zero_alpha:
            ind += entry[2] * entry[1] * self.kernel_linear([x, y], entry[0])
        ind -= self.bias
        return ind
