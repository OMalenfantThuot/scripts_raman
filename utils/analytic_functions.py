import math as m

class Sin:
    def __init__(self):
        pass

    def value(self, x):
        return m.sin(x)

    def first_derivative(self, x):
        return m.cos(x)

    def second_derivative(self, x):
        return -m.sin(x)

class Cos:
    def __init__(self):
        pass

    def value(self, x):
        return m.cos(x)

    def first_derivative(self, x):
        return -m.sin(x)

    def second_derivative(self, x):
        return -m.cos(x)
