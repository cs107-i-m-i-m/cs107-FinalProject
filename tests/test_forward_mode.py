import sys
import numpy as np
import pytest
from GrADim.forward_mode import ForwardMode
from GrADim.GrADim import Gradim


# Run using """python -m pytest tests/test_forward_mode.py"""
class TestForwardMode:
    X = ForwardMode(2)
    X1 = ForwardMode(0)
    X2 = ForwardMode(-1)
    multiple_X = ForwardMode(np.array([1, 2, 3]))

    float_equality_threshold = 1e-8

    def test_init(self):
        assert self.X.value == 2
        assert self.X.derivative == 1

    def test_add_left(self):
        Y = self.X + 4
        assert Y.value == 6
        assert Y.derivative == 1

    def test_add_right(self):
        Y = 4 + self.X
        assert Y.value == 6
        assert Y.derivative == 1

    def test_neg(self):
        Y = - self.X
        assert Y.value == -2
        assert Y.derivative == -1

    def test_sub_left(self):
        Y = self.X - 4
        assert Y.value == -2
        assert Y.derivative == 1

    def test_sub_right(self):
        Y = 4 - self.X
        assert Y.value == 2
        assert Y.derivative == -1

    def test_sub_both(self):
        def function_multiple_inputs(x):
            return x[1]-x[0]
        Y = function_multiple_inputs(self.multiple_X)
        assert Y.value == 1
        assert (Y.derivative == np.array([-1,1,0])).all()
    
    def test_mul_left(self):
        Y = self.X * 4
        assert Y.value == 8
        assert Y.derivative == 4

    def test_mul_right(self):
        Y = 4 * self.X
        assert Y.value == 8
        assert Y.derivative == 4

    def test_pow(self):
        Y = self.X ** 4
        assert Y.value == 16
        assert Y.derivative == 32

    def test_truediv_left(self):
        Y = self.X / 4
        assert Y.value == .5
        assert Y.derivative == .25

    def test_truediv_right(self):
        Y = 4 / self.X
        assert Y.value == 2
        assert Y.derivative == -1

    def test_truediv_both(self):
        def function_multiple_inputs(x):
            return x[1]/x[0]
        Y = function_multiple_inputs(self.multiple_X)
        assert Y.value == 2
        assert (Y.derivative == np.array([-2,1,0])).all()

    def test_sqrt(self):
        Y =  Gradim.sqrt(self.X)
        assert Y.value == 2 ** 0.5
        assert Y.derivative == 0.5 * 2 ** (-0.5)

    def test_exp(self):
        Y = Gradim.exp(self.X)
        assert Y.value == np.exp(2)
        assert Y.derivative == np.exp(2)

    def test_sin(self):
        Y = Gradim.sin(self.X)
        assert Y.value == np.sin(2)
        assert Y.derivative == np.cos(2)

    def test_cosec(self):
        Y = Gradim.cosec(self.X)
        assert Y.value == 1 /np.sin(2)
        assert Y.derivative == -1 * np.cos(2) / ((np.sin(2))**2)

    def test_cos(self):
        Y = Gradim.cos(self.X)
        assert Y.value == np.cos(2)
        assert Y.derivative == -np.sin(2)

    def test_sec(self):
        Y = Gradim.sec(self.X)
        assert Y.value == 1/np.cos(2)
        assert Y.derivative ==  1/np.cos(2) * np.tan(2)

    def test_tan(self):
        Y = Gradim.tan(self.X)
        assert Y.value == np.tan(2)
        assert Y.derivative == (1/np.cos(2))**2

    def test_cot(self):
        Y = Gradim.cot(self.X)
        assert Y.value == 1 / np.tan(2)
        assert Y.derivative == -1 * (1 /np.sin(2)) ** 2

    def test_log(self):
        Y = Gradim.log(self.X)
        assert Y.value == np.log(2)
        assert Y.derivative == 1/2
    
    def test_arcsin(self):
        self.X = ForwardMode(.5)
        Y = Gradim.arcsin(self.X)
        assert Y.value == np.arcsin(.5)
        assert Y.derivative == 1 / np.sqrt(1 - .5 ** 2)
    '''
    def test_arccosec(self):
        Y = Gradim.arccosec(self.X)
        assert Y.value == np.arcsin(1/2)
        assert Y.derivative == (-1 / np.sqrt(2 **2 - 1)) * 1/np.abs(2)
    '''
    def test_arccos(self):
        self.X = ForwardMode(.5)
        Y = Gradim.arccos(self.X)
        assert Y.value == np.arccos(.5)
        assert Y.derivative == -1 / np.sqrt(1 - .5 ** 2)
    '''
    def test_arcsec(self):
        Y = Gradim.arcsec(self.X)
        assert Y.value == np.arccos(1/2)
        assert Y.derivative == (1/np.sqrt(2**2 - 1)) * 1/np.abs(2)
    '''
    def test_arctan(self):
        Y = Gradim.arctan(self.X)
        assert Y.value == np.arctan(2)
        assert Y.derivative == 1 / (1 + 2 ** 2)
    '''
    def test_arccot(self):
        Y = Gradim.arccot(self.X)
        assert Y.value == np.acot(2)
        assert Y.derivative == -1 / (1 + 2 ** 2)
    '''
    def test_complex_function(self):
        Y = Gradim.exp(self.X) * Gradim.sin(self.X) + Gradim.tan(self.X)/Gradim.sin(self.X)
        true_value = np.exp(2) * np.sin(2) + np.tan(2)/np.sin(2)
        true_derivative = np.exp(2) * (np.cos(2) + np.sin(2)) + np.sin(2)/(np.cos(2)**2)
        assert np.abs(Y.value - true_value) < self.float_equality_threshold
        assert np.abs(Y.derivative - true_derivative) < self.float_equality_threshold
    
    def test_polynomial(self):
        Y = self.X**3 + self.X - 2
        assert Y.value == 8
        assert Y.derivative == 13

    def test_trig(self):
        Y = Gradim.cot(self.X) + 2 * Gradim.cosec(self.X) - Gradim.tan(self.X)
        assert Y.value == 1/np.tan(2) + 2 / np.sin(2) - np.tan(2)
        assert Y.derivative == -1/(np.sin(2)**2)-2/(np.tan(2)*np.sin(2))-1/(np.cos(2)**2)

    def test_exp_sqrt(self):
        Y = Gradim.sqrt(self.X) + 2 * self.X * Gradim.exp(self.X)
        assert Y.value == np.sqrt(2) + 4 * np.exp(2)
        assert Y.derivative == 1/(2 * np.sqrt(2)) + 6 * np.exp(2)

    def test_trig_power(self):
        Y = Gradim.sin(self.X) ** .2 - Gradim.cos(self.X) ** 3
        assert Y.value == np.sin(2)** .2 - np.cos(2)** 3
        assert Y.derivative == .2*(np.sin(2)**(-.8))*np.cos(2) + 3 * np.cos(2)** 2 * np.sin(2)

    def test_inverse_log(self):
        Y = 1/Gradim.log(self.X) + 2/(self.X**2)
        assert Y.value == 1/np.log(2) + .5
        assert Y.derivative == -.5*np.log(2)**(-2)-.5

    def test_function_multiple_inputs(self):
        def function_multiple_inputs(x):
            return Gradim.cos(x[0]) + Gradim.exp(x[2])*x[1]
        Y = function_multiple_inputs(self.multiple_X)
        assert Y.value == np.cos(1)+np.exp(3)*2
        assert (Y.derivative == np.array([-np.sin(1), np.exp(3), 2*np.exp(3)])).all()

    def test_function_multiple_outputs(self):
        @ForwardMode.multiple_outputs
        def function_multiple_outputs(x):
            return 3*x, Gradim.sin(x), Gradim.sqrt(x)
        Y = function_multiple_outputs(self.X)
        assert (Y.value == np.array([6, np.sin(2),2**.5])).all()
        assert (Y.derivative == np.array([3, np.cos(2),(.5)*2**(-.5)])).all()

    def test_function_multiple_inputs_and_outputs(self):
        @ForwardMode.multiple_outputs
        def function_multiple_inputs_and_outputs(x):
            return x[0] + 2*x[1] * x[2], x[0] - x[2]
        Y = function_multiple_inputs_and_outputs(self.multiple_X)
        assert (Y.value == np.array([13, -2])).all()
        assert (Y.derivative == np.array([[1,6,4], [1,0,-1]])).all()
