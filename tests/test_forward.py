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

    def test_exp(self):
        Y = Gradim.exp(self.X)
        assert Y.value == np.exp(2)
        assert Y.derivative == np.exp(2)

    def test_sin(self):
        Y = Gradim.sin(self.X)
        assert Y.value == np.sin(2)
        assert Y.derivative == np.cos(2)

    def test_cos(self):
        Y = Gradim.cos(self.X)
        assert Y.value == np.cos(2)
        assert Y.derivative == -np.sin(2)

    def test_tan(self):
        Y = Gradim.tan(self.X)
        assert Y.value == np.tan(2)
        assert Y.derivative == (1/np.cos(2))**2

    def test_complex_function(self):
        Y = Gradim.exp(self.X) * Gradim.sin(self.X) + Gradim.tan(self.X)/Gradim.sin(self.X)
        true_value = np.exp(2) * np.sin(2) + np.tan(2)/np.sin(2)
        true_derivative = np.exp(2) * (np.cos(2) + np.sin(2)) + np.sin(2)/(np.cos(2)**2)
        assert np.abs(Y.value - true_value) < self.float_equality_threshold
        assert np.abs(Y.derivative - true_derivative) < self.float_equality_threshold
