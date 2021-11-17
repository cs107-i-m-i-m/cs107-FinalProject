import numpy as np

from GrADim.forward_mode import *
from GrADim.GrADim import *


# Run using """python -m pytest tests/test_forward_mode.py"""
class TestForwardMode:
    X = ForwardMode(2)
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

    def test_abs(self):
        Y = abs(self.X)
        assert Y.value == 2
        assert Y.derivative == 1

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

    def test_abs(self):
        Y = Gradim.abs(self.X)
        assert Y.value == 2
        assert Y.derivative == 1
    '''
    def test_arcsin(self):
        Y = Gradim.arcsin(self.X)
        assert Y.value == np.arcsin(2)
        assert Y.derivative == 1 / np.sqrt(1 - 2 ** 2)

    def test_arccosec(self):
        Y = Gradim.arccosec(self.X)
        assert Y.value == np.arccosec(2)
        assert Y.derivative == (-1 / np.sqrt(2 **2 - 1)) * 1/np.abs(self.value)

    def test_arccos(self):
        Y = Gradim.arccos(self.X)
        assert Y.value == np.acsc(2)
        assert Y.derivative == -1 / np.sqrt(1 - 2 ** 2)

    def test_arcsec(self):
        Y = Gradim.arcsec(self.X)
        assert Y.value == np.asec(2)
        assert Y.derivative == (1/np.sqrt(2**2 - 1)) * 1/np.abs(2)

    def test_arctan(self):
        Y = Gradim.arctan(self.X)
        assert Y.value == np.arctan(2)
        assert Y.derivative == 1 / (1 + 2 ** 2)

    def test_arccot(self):
        Y = Gradim.arccot(self.X)
        assert Y.value == np.acot(2)
        assert Y.derivative == -1 / (1 + 2 ** 2)
    
    def test_complex_function(self):
        Y = Gradim.exp(self.X) * Gradim.sin(self.X) + Gradim.tan(self.X)/Gradim.sin(self.X)
        true_value = np.exp(2) * np.sin(2) + np.tan(2)/np.sin(2)
        true_derivative = np.exp(2) * (np.cos(2) + np.sin(2)) + np.sin(2)/(np.cos(2)**2)
        assert np.abs(Y.value - true_value) < self.float_equality_threshold
        assert np.abs(Y.derivative - true_derivative) < self.float_equality_threshold
    '''
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

    def test_inverse(self):
        Y = 1/self.X + 2/(self.X**2)
        assert Y.value == 1
        assert Y.derivative == -3/4

    def test_log_abs(self):
        X = ForwardMode(-2)
        Y = -3 * Gradim.log( 7 * Gradim.abs(self.X) )
        assert Y.value == -3* np.log(14)
        assert Y.derivative == -3/2
