""" File containing implementation of some learning rate schedules """

import numpy as np


class CLR(object):
    """
    The method is described in paper : https://arxiv.org/abs/1506.01186.
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    Args:
        itr     The learning rate increasing factor is calculated based on this
                iteration number.
        base_lr The lower boundary for learning rate which will be used as
                initial learning rate during test run. It is adviced to start
                from small learning rate value like 1e-4.
        max_lr  The upper boundary for learning rate. This value defines
                amplitude for learning rate increase (max_lr-base_lr).
    """

    def __init__(self, base_lr, max_lr, mode, step_size=2000, gamma=0.99994):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        self.itr = 0

    def calc(self):
        self.itr += 1
        return self.calc_lr()

    def calc_lr(self):
        cycle = np.floor(1 + self.itr/(2*self.step_size))
        x = np.abs(self.itr/self.step_size - 2*cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr)*max(0, 1-x)
        if self.mode == 'triangular2':
            lr /= float(2**(cycle-1))
        elif self.mode == 'exp_range':
            lr *= self.gamma**self.itr
        return lr
