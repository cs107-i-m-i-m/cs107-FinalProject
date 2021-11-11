import pytest
import forward_mode
import numpy as np
#import GrADim


def test_add():
    X = ForwardMode(2)
    def f(x):
        return x + 4
    Y = f(X)
    assert Y.value == 6
    assert Y.derivative == 1

def test_radd():
    X = ForwardMode(2)
    def g(x):
        return x + ForwardMode(4)
    Y = g(X)
    assert Y.value == 6
    assert Y.derivative == 1

def test_sub():
    X = ForwardMode(2)
    def f(x):
        return -x
    Y = f(X)
    assert Y.value == -2
    assert Y.derivative == -1
    
def test_sub():
    X = ForwardMode(2)
    def f(x):
        return x - 4
    Y = f(X)
    assert Y.value == -2
    assert Y.derivative == 1

def test_rsub():
    X = ForwardMode(2)
    def g(x):
        return x - ForwardMode(4)
    Y = g(X)
    assert Y.value == -2
    assert Y.derivative == 1

def test_mul():
    X = ForwardMode(2)
    def f(x):
        return x * 4
    Y = f(X)
    assert Y.value == 8
    assert Y.derivative == 4

def test_rmul():
    X = ForwardMode(2)
    def g(x):
        return x * ForwardMode(4)
    Y = g(X)
    assert Y.value == 8
    assert Y.derivative == 4

def test_pow():
    X = ForwardMode(2)
    def f(x):
        return x ** 4
    Y = f(X)
    assert Y.value == 16
    assert Y.derivative == 24

def test_div():
    X = ForwardMode(2)
    def f(x):
        return x / 4
    Y = f(X)
    assert Y.value == .5
    assert Y.derivative == .25

def test_rdiv():
    X = ForwardMode(2)
    def g(x):
        return x / ForwardMode(4)
    Y = g(X)
    assert Y.value == .5
    assert Y.derivative == .25

def test_exp():
    X = ForwardMode(2)
    def f(x):
        return ForwardMode.exp(x)
    Y = f(X)
    assert Y.value == np.exp(2)
    assert Y.derivative == np.exp(2)

def test_sin():
    X = ForwardMode(2)
    def f(x):
        return ForwardMode.sin(x)
    Y = f(X)
    assert Y.value == np.sin(2)
    assert Y.derivative == np.cos(2)

def test_cos():
    X = ForwardMode(2)
    def f(x):
        return ForwardMode.cos(x)
    Y = f(X)
    assert Y.value == np.cos(2)
    assert Y.derivative == np.sin(2)

def test_tan():
    X = ForwardMode(2)
    def f(x):
        return ForwardMode.tan(x)
    Y = f(X)
    assert Y.value == np.tan(2)
    assert Y.derivative == (1/np.cos(2))**2

if __name__=='__main__':
    test_add()

