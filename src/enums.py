""" File with some useful enums """

from enum import Enum
from .network import *
from .lr_schedules import *
from functools import partial


class Losses(Enum):
    """ Enum with the available losses """
    CROSSENTROPY = CrossEntropyCost
    QUADRATIC = QuadraticCost

    def __str__(self):
        return self.name.lower()


class Activations(Enum):
    """ Enum with the available activations functions """
    RELU = Network.relu
    SIGMOID = Network.sigmoid

    def __str__(self):
        return self.name.lower()


class Schedules(Enum):
    """ Enum with the available learning rate schedules """
    CONSTANT = partial(lambda lr, itr, epochs: CLR(
        base_lr=lr, max_lr=lr, mode='triangular'))
    TRIANGULAR = partial(lambda lr, itr, epochs: CLR(
        base_lr=lr, max_lr=5*lr, mode='triangular', step_size=max(1, itr // 30)
        if epochs <= 3 else 5*itr))
    TRIANGULAR2 = partial(lambda lr, itr, epochs: CLR(
        base_lr=lr, max_lr=7*lr, mode='triangular2', step_size=max(1, itr // 35)
        if epochs <= 3 else 5*itr))
    EXP_RANGE = partial(lambda lr, itr, epochs: CLR(
        base_lr=lr, max_lr=5*lr, mode='exp_range', step_size=max(1, itr // 40)
        if epochs <= 3 else 5*itr))