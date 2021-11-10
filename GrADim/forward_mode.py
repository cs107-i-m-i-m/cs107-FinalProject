import numpy as np
import inspect


class ForwardMode:
    def __init__(self, value, derivative=1):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(self.value + other, self.derivative)
        return ForwardMode(self.value + other.value, self.derivative + other.derivative)

    def __radd__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(other + self.value, self.derivative)
        return ForwardMode(other.value + self.value, other.derivative + self.derivative)

    def __neg__(self):
        return ForwardMode(-self.value, -self.derivative)

    def __sub__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(self.value - other, self.derivative)
        return ForwardMode(self.value - other.value, self.derivative - other.derivative)

    def __rsub__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(other - self.value, - self.derivative)
        return ForwardMode(other.value - self.value, other.derivative - self.derivative)

    def __mul__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(other * self.value, other * self.derivative)
        return ForwardMode(self.value * other.value, self.derivative * other.value + self.value * other.derivative)

    def __rmul__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(self.value * other, self.derivative * other)
        return ForwardMode(other.value * self.value, other.derivative * self.value + other.value * other.derivative)

    def __div__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(self.value / other, self.derivative / other)
        return ForwardMode(self.value / other.value, (self.derivative * other.value - self.value * other.derivative) / other.value ** 2)

    def __rdiv__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(other / self.value, other / self.derivative)
        return ForwardMode(other.value / self.value, (self.value * other.derivative - self.derivative * other.value) / self.value ** 2)

    def exp(self):
        return ForwardMode(np.exp(self.value), self.derivative * np.exp(self.value))

    def sin(self):
        return ForwardMode(np.sin(self.value), self.derivative * np.cos(self.value))

    def cos(self):
        return ForwardMode(np.cos(self.value), - self.derivative * np.sin(self.value))

    def tan(self):
        return ForwardMode(np.tan(self.value), self.derivative * (1 + np.tan(self.value)**2))

if __name__ == "__main__":
    X = ForwardMode(2)
    def f(x):
        return -x*x + 2*x + 4

    def g(x):
        return ForwardMode.exp(x) + 2 * ForwardMode.sin(x) * ForwardMode.cos(x)

    Y = f(X)
    print(Y.value)
    print(Y.derivative)

    Y2 = g(X)
    print(Y2.value)
    print(Y2.derivative)
