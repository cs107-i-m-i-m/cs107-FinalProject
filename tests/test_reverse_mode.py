import numpy as np

from GrADim.reverse_mode import ReverseMode
from GrADim.GrADim import Gradim


# Run using """python -m pytest tests/test_reverse_mode.py"""
class TestReverseMode:
    x = ReverseMode(3)
    y = ReverseMode(2)
    float_equality_threshold = 1e-8

    def test_addition_c():
    g = x + 2
    assert (float(g.value) == 5.0) & (float(x.derivative) == 1.0)

    def test_addition_v():
    g = x + y
    assert (float(g.value) == 5.0) & (float(x.derivative) == 1.0) & (float(y.derivative) == 1.0)

    def test_radd_c():
    g = 2 + x 
    assert (float(g.value) == 5.0) & (float(x.derivative) ==  1.0)

    def test_neg_c():
    g = -x 
    assert (float(g.value) == -3.0) & (float(x.derivative) ==  -1.0)

    def test_subtraction_c():
    g = x - 2
    assert (float(g.value) == 1.0) & (float(x.derivative) == 1.0)

    def test_subtraction_v():
    g = x - y
    assert (float(g).value) == 1.0) & (float(x.derivative) == 1.0) & (float(y.derivative) == 1.0)

    def test_rsub_c():
    g = 2 - x 
    assert (float(g.value) == -1.0) & (float(x.derivative) ==  -1.0)

    def test_multiplication_c():
    g = x * 2
    assert (float(g.value) == 6.0) & (float(x.derivative) == 2.0)

    def test_multiplication_v():
    g = x * y
    assert (float(g.value) == 6.0) & (float(x.derivative) == 2.0) & (float(y.derivative) == 2.0)

    def test_rmul_c():
    g = 2 * x 
    assert (float(g.value) == 6.0) & (float(x.derivative) ==  2.0)

    def test_power_c():
    g = x ** 2
    assert (float(g.value) == 9.0) & (float(x.derivative) == 6.0)

    def test_power_v():
    g = x ** y
    assert (float(g.value) == 9.0) & (float(x.derivative) == 6.0) & (float(y.derivative) == np.log(3.0)*9.0)

    def test_division_c():
    g = x / 2
    assert (float(g.value) == 1.5) & (float(x.derivative) == 0.5)

    def test_division_v():
    g = x / y
    assert (float(g.value) == 1.5) & (float(x.derivative) == 0.5) & (float(y.derivative) == -0.75)

    def test_rdiv_c():
    g = 2 / x
    assert (float(g.value) == (2/3)) & (float(x.derivative) ==  -(2/9))

    def test_sqrt_c():
    g = gradim.sqrt(x)
    assert (float(g.value) == 3**0.5)) & (float(x.derivative) ==  0.5 * 3 ** (-0.5))

    def test_exp():
    g = Gradim.exp(x)
    assert (float(g.value) == np.exp(3)) & (float(x.derivative) ==  np.exp(3)) 

    def test_sin():
    g = Gradim.sin(x)
    assert (float(g.value) == np.sin(3)) & (float(x.derivative) == np.cos(3)) 

    def test_cosec():
    g = Gradim.cosec(x)
    assert (float(g.value) == 1 / np.sin(3)) & (float(x.derivative) == -1 * np.cos(3) / ((np.sin(3))**2)

    def test_cos():
    g = Gradim.cos(x)
    assert (float(g.value) == np.cos(3)) & (float(x.derivative) == -np.sin(3)) 

    def test_sec():
    g = Gradim.sec(x)
    assert (float(g.value) == 1/np.cos(3)) & (float(x.derivative) == -1/np.cos(3) * np.tan(3))

    def test_tan(): np.sqrt(1 - 2 ** 2)

    g = Gradim.tan(x)
    assert (float(g.value) == np.tan(3)) & (float(x.derivative) == 1/(np.cos(3)**2))

    def test_cot():
    g = Gradim.cot(x)
    assert (float(g.value) == 1/np.tan(3)) & (float(x.derivative) == -1 * (1 /np.sin(3)) ** 2)

    def test_log():
    g = Gradim.log(x)
    assert (float(g.value) == np.log(3)) & (float(x.derivative) == 1/3)

    def test_arcsin():
    g = Gradim.arcsin(x)
    assert (float(g.value) == np.arcsin(3)) & (float(x.derivative) == 1 / np.sqrt(1 - 3 ** 2)

    def test_arccos():
    g = Gradim.arccos(x)
    assert (float(g.value) == np.arccos(3)) & (float(x.derivative) ==-1 /np.sqrt(1 - 3 ** 2))

    def test_arctan():
    g = Gradim.arctan(x)
    assert (float(g.value) == np.arctan(3)) & (float(x.derivative) ==  1 / (1 + 3 ** 2)










       

   











