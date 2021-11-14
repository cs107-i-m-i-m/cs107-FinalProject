import numpy as np

from GrADim.GrADim import Gradim

class ForwardMode(Gradim):
    
    #Initialize the value and derivative, if the derivative is not provided set it to 1. (Essentailly dual number Value + e derivative)
    def __init__(self, value, derivative=1): 
        self.value = value
        self.derivative = derivative
        
    #For a single valued function, Jacobian is the derivative    
    def __Jacobian__(self): 
        return self.derivative
    
    #Check if the other object to add is a similar class object or just a constant, and handle them appropiately
    
    def __add__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(self.value + other, self.derivative)
        return ForwardMode(self.value + other.value, self.derivative + other.derivative)

    def __radd__(self, other):
        return self.__add__(other)

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
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        return ForwardMode(self.value ** power, power * self.derivative * self.value ** (power-1))

    def __truediv__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(self.value / other, self.derivative / other)
        return ForwardMode(self.value / other.value, (self.derivative * other.value - self.value * other.derivative) / other.value ** 2)

    def __rtruediv__(self, other):
        if type(other) != self.__class__:
            return ForwardMode(other / self.value, - other * self.derivative / self.value**2)
        return ForwardMode(other.value / self.value, (self.value * other.derivative - self.derivative * other.value) / self.value ** 2)
    
    def __abs__(self): #does the absolute affect the derivative too??
        return ForwardMode(np.abs(self.value), self.derivative )
    
    def __sqrt__(self):
        return ForwardMode(self.value ** 0.5, 0.5 * self.derivative * self.value ** (-0.5))
        
    def exp(self):
        return ForwardMode(np.exp(self.value), self.derivative * np.exp(self.value))

    def sin(self):
        return ForwardMode(np.sin(self.value), self.derivative * np.cos(self.value))
    
    def cosec(self):
        return 1 / self.sin(self)

    def cos(self):
        return ForwardMode(np.cos(self.value), - self.derivative * np.sin(self.value))
    
    def sec(self):
        return 1 / self.cos(self)

    def tan(self):
        return ForwardMode(np.tan(self.value), self.derivative * (1 + np.tan(self.value)**2))
    
    def cot(self):
        return 1 / self.tan(self)
    
    def log(self):
        return ForwardMode(np.log(self.value), self.derivative * (1/self.value))
    
    def arcsin(self):
        return ForwardMode(np.arcsin(self.value), self.derivative * (1/np.srt(1 - self.value**2)))
    
    def arccosec(self):
        return ForwardMode(np.arccosec(self.value), - self.derivative * (1/np.srt(self.value**2 - 1)) * 1/np.abs(self.value))
    
    def arccos(self):
        return ForwardMode(np.arccos(self.value), - self.derivative * (1/np.srt(1 - self.value**2)))
    
    def arcssec(self):
        return ForwardMode(np.arcsec(self.value), self.derivative * (1/np.srt(self.value**2 - 1)) * 1/np.abs(self.value))
    
    def arctan(self):
        return ForwardMode(np.arctan(self.value), self.derivative * (1/(1 + self.value**2)))
    
    def arccot(self):
        return ForwardMode(np.arccot(self.value), - self.derivative * (1/(1 + self.value**2)))
    
    def Newton_Raphson(fun,x0,eps,epochs):
        xn = x0
        for i in range(epochs):
            X = ForwardMode(x0)
            y = fun(X)
            e = np.float(y.value) / np.float(y.derivative)
            xn = xn - e
            if e < eps:
                print("The root found is: ", xn)
                return eps
                
            print("Max epochs reached, the closest root value is: ", xn)
            return eps

if __name__ == "__main__":
    X = ForwardMode(2)
    def f(x):
        return -1+X
    

    def g(x):
        return ForwardMode.exp(ForwardMode.sin(x))**0.2 + 2 * ForwardMode.sin(x) * ForwardMode.cos(x)

    Y = f(X)
    print(Y.value)
    print(Y.derivative)

    Y2 = g(X)
    #print(Y2.value)
    #print(Y2.derivative)
    
    
        
    ForwardMode.Newton_Raphson(f,1,0.01,100)
            
        
