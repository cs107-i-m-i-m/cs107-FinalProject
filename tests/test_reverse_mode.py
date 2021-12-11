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

    def test_power_v(self):
        x = ReverseMode(3)
        y = ReverseMode(2)
        g = x ** y
        assert (float(g.value) == 9.0) & (float(x.derivative) == 6.0) & (float(y.derivative) == np.log(3.0)*9.0)

    def test_rpower_c(self):
        x = ReverseMode(3)
        g = 2**x
        assert (g.value == 8) & (x.derivative == np.log(2)*8)

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

    def test_eq(self):
        x = ReverseMode(3)
        x1 = ReverseMode(3)
        assert x == x1
        assert x == 3
        assert 3 == x

    def test_ne(self):
        x = ReverseMode(3)
        x1 = ReverseMode(2)
        assert x != x1
        assert x != 2
        assert 2 != x

    def test_lt(self):
        x = ReverseMode(3)
        x1 = ReverseMode(2)
        assert x1 < x
        assert x1 < 3
        assert -1 < x1

    def test_gt(self):
        x = ReverseMode(3)
        x1 = ReverseMode(2)
        assert x > x1
        assert x1 > -1
        assert 3 > x1

    def test_le(self):
        x = ReverseMode(3)
        x1 = ReverseMode(3)
        x2 = ReverseMode(2)
        assert x2 <= x
        assert x1 <= x
        assert -1 <= x
        assert 3 <= x
        assert x <= 3
        assert x <= 4

    def test_ge(self):
        x = ReverseMode(3)
        x1 = ReverseMode(3)
        x2 = ReverseMode(2)
        assert x >= x2
        assert x >= x1
        assert 4 >= x
        assert 3 >= x
        assert x >= 2
        assert x >= 3

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

    def test_ln(self):
        x = ReverseMode(3)
        g = Gradim.ln(x)
        assert (float(g.value) == np.log(3)) & (float(x.derivative) == 1/3)

    def test_log(self):
        x = ReverseMode(2)
        g = Gradim.log(x, base=2)
        assert (float(g.value) == 1) & (float(x.derivative) == 1/(2*np.log(2)))

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

    def test_sinh(self):
        x = ReverseMode(3)
        g = Gradim.sinh(x)
        assert (g.value == (np.exp(3) - np.exp(-3)) / 2) & (x.derivative == (np.exp(3) + np.exp(-3)) / 2)

    def test_cosh(self):
        x = ReverseMode(3)
        g = Gradim.cosh(x)
        assert (g.value == (np.exp(3) + np.exp(-3)) / 2) & (x.derivative == (np.exp(3) - np.exp(-3)) / 2)

    def test_tanh(self):
        x = ReverseMode(3)
        g = Gradim.tanh(x)
        assert (g.value == (np.exp(3) - np.exp(-3)) / (np.exp(3) + np.exp(-3))) & (np.abs(x.derivative - 4 * (np.exp(3) + np.exp(-3)) ** (-2)) < self.float_equality_threshold)

    def test_logistic(self):
        x = ReverseMode(3)
        g = Gradim.logistic(x)
        assert (g.value == 1/(1+np.exp(-3))) & (x.derivative == (1+np.exp(-3))**(-2)*np.exp(-3))

    def test_complex_function_with_seed(self):
        x = ReverseMode(3, derivative=2)
        g = x * Gradim.sin(x)
        assert (g.value == 3*np.sin(3)) & (x.derivative == 2 * np.sin(3) + 2 * 3 * np.cos(3))

    def test_multiple_inputs_function(self):
        x = ReverseMode(np.array([2, 3]))
        g = x[0] + 2*x[1]
        assert (g.value == 8) & ((x.derivative == np.array([1, 2])).all())

    def test_multiple_outputs_function(self):
        x = ReverseMode(3)
        @ReverseMode.multiple_outputs
        def f(x):
            return 2*x, x+3
        g = f(x)
        assert ((g.value == np.array([6, 6])).all()) & ((x.derivative == np.array([2, 1])).all())

    def test_multiple_inputs_and_outputs_function(self):
        x = ReverseMode(np.array([2, 3]))
        @ReverseMode.multiple_outputs
        def f(x):
            return x[0] + x[1], x[0] * x[1]
        g = f(x)
        assert ((g.value == np.array([5, 6])).all()) & ((x.derivative == np.array([[1, 1], [3, 2]])).all())