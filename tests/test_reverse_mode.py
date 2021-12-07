import numpy as np

from GrADim.reverse_mode import ReverseMode
from GrADim.GrADim import Gradim


# Run using """python -m pytest tests/test_reverse_mode.py"""
class TestReverseMode:
    float_equality_threshold = 1e-8

    def test_addition_c(self):
        x = ReverseMode(3)
        g = x + 2
        assert (float(g.value) == 5.0) & (float(x.derivative) == 1.0)

    def test_addition_v(self):
        x = ReverseMode(3)
        y = ReverseMode(2)
        g = x + y
        assert (float(g.value) == 5.0) & (float(x.derivative) == 1.0) & (float(y.derivative) == 1.0)

    def test_radd_c(self):
        x = ReverseMode(3)
        g = 2 + x
        assert (float(g.value) == 5.0) & (float(x.derivative) ==  1.0)

    def test_neg_c(self):
        x = ReverseMode(3)
        g = -x
        assert (float(g.value) == -3.0) & (float(x.derivative) ==  -1.0)

    def test_subtraction_c(self):
        x = ReverseMode(3)
        g = x - 2
        assert (float(g.value) == 1.0) & (float(x.derivative) == 1.0)

    def test_subtraction_v(self):
        x = ReverseMode(3)
        y = ReverseMode(2)
        g = x - y
        assert (float(g.value) == 1.0) & (float(x.derivative) == 1.0) & (float(y.derivative) == -1.0)

    def test_rsub_c(self):
        x = ReverseMode(3)
        g = 2 - x
        assert (float(g.value) == -1.0) & (float(x.derivative) ==  -1.0)

    def test_multiplication_c(self):
        x = ReverseMode(3)
        g = x * 2
        assert (float(g.value) == 6.0) & (float(x.derivative) == 2.0)

    def test_multiplication_v(self):
        x = ReverseMode(3)
        y = ReverseMode(2)
        g = x * y
        assert (float(g.value) == 6.0) & (float(x.derivative) == 2.0) & (float(y.derivative) == 3.0)

    def test_rmul_c(self):
        x = ReverseMode(3)
        g = 2 * x
        assert (float(g.value) == 6.0) & (float(x.derivative) ==  2.0)

    def test_power_c(self):
        x = ReverseMode(3)
        g = x ** 2
        assert (float(g.value) == 9.0) & (float(x.derivative) == 6.0)

    # TODO Uncomment when function power is finished
    # def test_power_v(self):
    #     x = ReverseMode(3)
    #     y = ReverseMode(2)
    #     g = x ** y
    #     assert (float(g.value) == 9.0) & (float(x.derivative) == 6.0) & (float(y.derivative) == np.log(3.0)*9.0)

    def test_division_c(self):
        x = ReverseMode(3)
        g = x / 2
        assert (float(g.value) == 1.5) & (float(x.derivative) == 0.5)

    def test_division_v(self):
        x = ReverseMode(3)
        y = ReverseMode(2)
        g = x / y
        assert (float(g.value) == 1.5) & (float(x.derivative) == 0.5) & (float(y.derivative) == -0.75)

    def test_rdiv_c(self):
        x = ReverseMode(3)
        g = 2 / x
        assert (float(g.value) == (2/3)) & (float(x.derivative) ==  -(2/9))

    def test_sqrt_c(self):
        x = ReverseMode(3)
        g = Gradim.sqrt(x)
        assert (float(g.value) == 3**0.5) & (float(x.derivative) ==  0.5 * 3 ** (-0.5))

    def test_exp(self):
        x = ReverseMode(3)
        g = Gradim.exp(x)
        assert (float(g.value) == np.exp(3)) & (float(x.derivative) ==  np.exp(3))

    def test_sin(self):
        x = ReverseMode(3)
        g = Gradim.sin(x)
        assert (float(g.value) == np.sin(3)) & (float(x.derivative) == np.cos(3))

    def test_cosec(self):
        x = ReverseMode(3)
        g = Gradim.cosec(x)
        assert (float(g.value) == 1 / np.sin(3)) & (float(x.derivative) == -1 * np.cos(3) / ((np.sin(3))**2))

    def test_cos(self):
        x = ReverseMode(3)
        g = Gradim.cos(x)
        assert (float(g.value) == np.cos(3)) & (float(x.derivative) == -np.sin(3))

    def test_sec(self):
        x = ReverseMode(3)
        g = Gradim.sec(x)
        assert (float(g.value) == 1/np.cos(3)) & (float(x.derivative) == 1/np.cos(3) * np.tan(3))

    def test_tan(self):
        x = ReverseMode(3)
        g = Gradim.tan(x)
        assert (float(g.value) == np.tan(3)) & (float(x.derivative) == 1 + np.tan(3)**2)

    def test_cot(self):
        x = ReverseMode(3)
        g = Gradim.cot(x)
        assert (float(g.value) == 1/np.tan(3)) & (np.abs(float(x.derivative) + 1 * (1 /np.sin(3)) ** 2) < self.float_equality_threshold)

    def test_log(self):
        x = ReverseMode(3)
        g = Gradim.log(x)
        assert (float(g.value) == np.log(3)) & (float(x.derivative) == 1/3)

    def test_arcsin(self):
        x = ReverseMode(.5)
        g = Gradim.arcsin(x)
        assert (float(g.value) == np.arcsin(.5)) & (float(x.derivative) == 1 / np.sqrt(1 - .5 ** 2))

    def test_arccos(self):
        x = ReverseMode(.5)
        g = Gradim.arccos(x)
        assert (float(g.value) == np.arccos(.5)) & (float(x.derivative) ==-1 /np.sqrt(1 - .5 ** 2))

    def test_arctan(self):
        x = ReverseMode(.5)
        g = Gradim.arctan(x)
        assert (float(g.value) == np.arctan(.5)) & (float(x.derivative) ==  1 / (1 + .5 ** 2))
