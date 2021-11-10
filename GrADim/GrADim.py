import numpy as np
import inspect

class autodiff():
    
    def __init__(self, fun):
        self.fun = fun
    
    def forward(self, x):
        
        return x
    
    def Jacobian(self):
        
        return(self.fun)
    
    def __add__(self, b):
        
        return (self + b)
    
    def __radd__(self, b):
        
        return (b + self)
    
    def __mul__(self, b):
        
        return (self * b)
    
    def __rmul__(self, b):
        
        return (b * self)
    
    
    def __sub__(self, b):
        
        return (self - b)
    
    
    def __rsub__(self, b):
        
        return (b - self)
    
    def __div__(self, b):
        
        return (self / b)
    
    def __rdiv__(self, b):
        
        return (b / self)
    
    def __repr__(self):
        return (inspect.getsource(self.fun)) #dumb way to get function definition
    
    
if __name__ == "__main__":
   print("Blah")
   
   def fun(x):
       return x
       
   ad = autodiff(fun)
   a = np.pi
   
   print(ad.forward(a))
   print(repr(ad))
   
   #root finding
   #optimization
   #other uses of autograd