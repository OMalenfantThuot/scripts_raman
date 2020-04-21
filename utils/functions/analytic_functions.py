import numpy as np


class Sin:
    def __init__(self):
        pass

    def value(self, x):
        return np.sin(x)

    def first_derivative(self, x):
        return np.cos(x)

    def second_derivative(self, x):
        return -np.sin(x)


class Cos:
    def __init__(self):
        pass

    def value(self, x):
        return np.cos(x)

    def first_derivative(self, x):
        return -np.sin(x)

    def second_derivative(self, x):
        return -np.cos(x)


class LennardJones:
    def __init__(self, E0=1.0, d=1.0):
        self.E0 = E0
        self.d = d

    def value(self, x):
        return 4 * self.E0 * ((self.d / x) ** 12 - (self.d / x) ** 6)

    def first_derivative(self, x):
        return 4 * self.E0 * (-12 * self.d ** 12 / x ** 13 + 6 * self.d ** 6 / x ** 7)

    def second_derivative(self, x):
        return 4 * self.E0 * (156 * self.d ** 12 / x ** 14 - 42 * self.d ** 6 / x ** 8)


class Quadratic:
    def __init__(self):
        pass

    def value(self, x):
        return x ** 2

    def first_derivative(self, x):
        return 2 * x

    def second_derivative(self, x):
        return 2 * np.ones(len(x))


class Polynomial_order4:
    def __init__(self):
        pass

    def value(self, x):
        return x ** 4

    def first_derivative(self, x):
        return 4 * x ** 3

    def second_derivative(self, x):
        return 12 * x ** 2
