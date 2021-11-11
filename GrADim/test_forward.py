import pytest
from forward_mode import *
import numpy as np


def test_add_left():
    X = ForwardMode(2)
    def f(x):
        return x + 4
    Y = f(X)
    assert Y.value == 6
    assert Y.derivative == 1

def test_add_right():
    X = ForwardMode(2)
    def f(x):
        return 4 + x
    Y = f(X)
    assert Y.value == 6
    assert Y.derivative == 1

def test_radd():
    X = ForwardMode(2)
    X2 = ForwardMode(4)
    def g(x):
        return x + X2
    Y = g(X)
    assert Y.value == 6
    assert Y.derivative == 2

def test_neg():
    X = ForwardMode(2)
    def f(x):
        return -x
    Y = f(X)
    assert Y.value == -2
    assert Y.derivative == -1
    
def test_sub_left():
    X = ForwardMode(2)
    def f(x):
        return x - 4
    Y = f(X)
    assert Y.value == -2
    assert Y.derivative == 1

def test_sub_right():
    X = ForwardMode(2)
    def f(x):
        return 4 - x
    Y = f(X)
    assert Y.value == 2
    assert Y.derivative == -1

def test_rsub():
    X = ForwardMode(2)
    X2 = ForwardMode(4)
    def g(x):
        return x - X2
    Y = g(X)
    assert Y.value == -2
    assert Y.derivative == 0

def test_mul_left():
    X = ForwardMode(2)
    def f(x):
        return x * 4
    Y = f(X)
    assert Y.value == 8
    assert Y.derivative == 4

def test_mul_right():
    X = ForwardMode(2)
    def f(x):
        return 4 * x
    Y = f(X)
    assert Y.value == 8
    assert Y.derivative == 4

def test_rmul():
    X = ForwardMode(2)
    X2 = ForwardMode(4)
    def g(x):
        return x * X2
    Y = g(X)
    assert Y.value == 8
    assert Y.derivative == 6

def test_pow():
    X = ForwardMode(2)
    def f(x):
        return x ** 4
    Y = f(X)
    assert Y.value == 16
    assert Y.derivative == 32

def test_truediv_left():
    X = ForwardMode(2)
    def f(x):
        return x / 4
    Y = f(X)
    assert Y.value == .5
    assert Y.derivative == .25

def test_truediv_right():
    X = ForwardMode(2)
    def f(x):
        return 4 / x
    Y = f(X)
    assert Y.value == 2
    assert Y.derivative == -1
    
def test_rtruediv():
    X = ForwardMode(2)
    X2 = ForwardMode(4)
    def g(x):
        return x / X2
    Y = g(X)
    assert Y.value == .5
    assert Y.derivative == 1/8

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
    assert Y.derivative == -np.sin(2)

def test_tan():
    X = ForwardMode(2)
    def f(x):
        return ForwardMode.tan(x)
    Y = f(X)
    assert Y.value == np.tan(2)
    assert Y.derivative == (1/np.cos(2))**2

if __name__=='__main__':
    test_add_left()
    test_add_right()
    test_radd()
    test_neg()
    test_sub_left()
    test_sub_right()
    test_rsub()
    test_mul_left()
    test_mul_right()
    test_rmul()
    test_pow()
    test_truediv_left()
    test_truediv_right()
    test_rtruediv()
    test_exp()
    test_sin()
    test_cos()
    test_tan()
