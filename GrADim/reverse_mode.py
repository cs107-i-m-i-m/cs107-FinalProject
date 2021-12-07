import numpy as np

from GrADim import Gradim

class ReverseMode(Gradim):
    def __init__(self, value):
        self.value = value
        # The children attribute is a list of tuples. Each tuple contains the following node in the evaluation tree
        # and the constant by which it needs to be multiplied to compute the derivative
        self.children = []

    def __add__(self, other):
        if type(other) != self.__class__:
            new = ReverseMode(self.value + other)
        else:
            new = ReverseMode(self.value + other.value)
            other.children.append((1, new))
        self.children.append((1, new))
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        new = ReverseMode(- self.value)
        self.children.append((-1, new))
        return new

    def __sub__(self, other):
        return self.__add__(- other)

    def __rsub__(self, other):
        return - self.__sub__(other)

    def __mul__(self, other):
        if type(other) != self.__class__:
            new = ReverseMode(self.value * other)
            self.children.append((other, new))
        else:
            new = ReverseMode(self.value * other.value)
            other.children.append((self.value, new))
            self.children.append((other.value, new))
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        new = ReverseMode(self.value ** power)
        self.children.append((power * self.value ** (power - 1), new))
        return new

    def __truediv__(self, other):
        if type(other) != self.__class__:
            new = ReverseMode(self.value/other)
            self.children.append((1/other, new))
        else:
            new = ReverseMode(self.value/other.value)
            other.children.append((- self.value/other.value**2, new))
            self.children.append((1/other.value, new))
        return new

    def __rtruediv__(self, other):
        if type(other) != self.__class__:
            new = ReverseMode(other/self.value)
            self.children.append((-other/self.value**2, new))
        else:
            new = ReverseMode(other.value/self.value)
            other.children.append((1/self.value, new))
            self.children.append((- other.value/self.value**2, new))
        return new

    @property
    def derivative(self):
        if not self.children:
            return 1
        return sum([const * child.derivative for const, child in self.children])

    def sqrt(self):
        return self**0.5

    def exp(self):
        new = ReverseMode(np.exp(self.value))
        self.children.append((np.exp(self.value), new))
        return new

    def sin(self):
        new = ReverseMode(np.sin(self.value))
        self.children.append((np.cos(self.value), new))
        return new

    def cosec(self):
        return 1/Gradim.sin(self)

    def cos(self):
        new = ReverseMode(np.cos(self.value))
        self.children.append((-np.sin(self.value), new))
        return new

    def sec(self):
        return 1/Gradim.cos(self)

    def tan(self):
        new = ReverseMode(np.tan(self.value))
        self.children.append((1 + np.tan(self.value)**2, new))
        return new

    def cot(self):
        return 1/Gradim.tan(self)

    def log(self):
        new = ReverseMode(np.log(self.value))
        self.children.append((1/self.value, new))
        return new

    def arcsin(self):
        new = ReverseMode(np.arcsin(self.value))
        self.children.append((1/np.sqrt(1 - self.value**2), new))
        return new

    def arccos(self):
        new = ReverseMode(np.arccos(self.value))
        self.children.append((- 1/np.sqrt(1 - self.value**2), new))
        return new

    def arctan(self):
        new = ReverseMode(np.arctan(self.value))
        self.children.append((1/(1 + self.value**2), new))
        return new


if __name__ == "__main__":
    X = ReverseMode(.5)

    def g(x):
        return Gradim.arcsin(x) + Gradim.arccos(x) + Gradim.arctan(x)

    print(g(X).value)
    print(X.derivative)
