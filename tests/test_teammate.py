from autora.theorist.teammate import ExampleRegressor

def test():
    theorist = ExampleRegressor()
    assert theorist is not None


# Note: We encourage you to adjust this test and write more tests.


import torch
import pytest
from torch import nn

# Import your module definitions (adjust import path as needed)
from src.autora.theorist.teammate.candidate_functions import (
    Plus, Subtract, Mult, Div, Ln, Exp, Constant, Pow
)


def test_plus():
    op = Plus()
    a, b = torch.tensor(2.), torch.tensor(3.)
    out = op(a, b)
    assert torch.isclose(out, torch.tensor(5.))
    assert op.eq_string("x", "y") == "x+y"
    assert isinstance(op, nn.Module)


def test_subtract():
    op = Subtract()
    a, b = torch.tensor(5.), torch.tensor(2.)
    out = op(a, b)
    assert torch.isclose(out, torch.tensor(3.))
    assert op.eq_string("a", "b") == "a-b"


def test_mult():
    op = Mult()
    a, b = torch.tensor(4.), torch.tensor(2.)
    out = op(a, b)
    assert torch.isclose(out, torch.tensor(8.))
    assert op.eq_string("x", "y") == "x*y"


def test_div():
    op = Div()
    a, b = torch.tensor(6.), torch.tensor(2.)
    out = op(a, b)
    assert torch.isclose(out, torch.tensor(3.))
    assert op.eq_string("a", "b") == "a/b"


def test_ln():
    op = Ln()
    a = torch.tensor(2.718281828)
    out = op(a)
    assert torch.allclose(out, torch.tensor(1.0), atol=1e-5)
    assert op.eq_string("x") == "ln(x)"


def test_exp():
    op = Exp()
    a = torch.tensor(1.)
    out = op(a)
    assert torch.allclose(out, torch.tensor(2.718281828), atol=1e-3)
    assert op.eq_string("x") == "expx"


def test_constant():
    op = Constant(5)
    out = op()
    assert torch.isclose(out, torch.tensor(5.))
    # eq_string here takes a number, not a label string
    assert op.eq_string(5) == "5.0"


def test_pow():
    op = Pow(2)
    a = torch.tensor(3.)
    out = op(a)
    assert torch.isclose(out, torch.tensor(9.))
    assert op.eq_string("x") == "x^2.0"


if __name__ == "__main__":
    pytest.main([__file__])
