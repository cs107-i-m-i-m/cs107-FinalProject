import numpy as np

from GrADim.forward_mode import ForwardMode
from GrADim.GrADim import Gradim


# Run using """python -m pytest tests/test_forward.py"""
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
        Y = - self.X
        assert Y.value == 2
        assert Y.derivative == 1

    def test_sqrt(self):
        Y =  self.X
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
        assert Y.derivative == -1 * ((np.sin(2)) ** 2) * np.cos(2)

    def test_cos(self):
        Y = Gradim.cos(self.X)
        assert Y.value == np.cos(2)
        assert Y.derivative == -np.sin(2)

     def test_sec(self):
        Y = Gradim.sec(self.X)
        assert Y.value == 1/np.cos(2)
        assert Y.derivative == -1 * ((np.cos(2)) ** 2) * np.sin(2)

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
        Y = Gradim.arcsin(self.X)
        assert Y.value == np.arcsin(2)
        assert Y.derivative == 1 / np.sqrt(1 - 2 ** 2)

    def test_arccosec(self):
        Y = Gradim.arccosec(self.X)
        assert Y.value == np.arccosec(2)
        assert Y.derivative == (-1 / np.sqrt(2 **2 - 1)) * 1/np.abs(self.value)

    def test_arccos(self):
        Y = Gradim.arccos(self.X)
        assert Y.value == np.arccos(2)
        assert Y.derivative == -1 / np.sqrt(1 - 2 ** 2)

    def test_arcsec(self):
        Y = Gradim.arcsec(self.X)
        assert Y.value == np.arcsec(2)
        assert Y.derivative == (1/np.sqrt(2**2 - 1)) * 1/np.abs(2))

    def test_arctan(self):
        Y = Gradim.arctan(self.X)
        assert Y.value == np.arctan(2)
        assert Y.derivative == 1 / (1 + 2 ** 2)

    def test_arccot(self):
        Y = Gradim.arccot(self.X)
        assert Y.value == np.arccot(2)
        assert Y.derivative == -1 / (1 + 2 ** 2)

    def test_complex_function(self):
        Y = Gradim.exp(self.X) * Gradim.sin(self.X) + Gradim.tan(self.X)/Gradim.sin(self.X)
        true_value = np.exp(2) * np.sin(2) + np.tan(2)/np.sin(2)
        true_derivative = np.exp(2) * (np.cos(2) + np.sin(2)) + np.sin(2)/(np.cos(2)**2)
        assert np.abs(Y.value - true_value) < self.float_equality_threshold
        assert np.abs(Y.derivative - true_derivative) < self.float_equality_threshold
