from enum import Enum

_eps = 1e-14


def double_is_zero(x):
    return x <= _eps
