import cProfile
import functools
import io
import os
import pstats
import random
import sys
from warnings import warn

import numpy as np
import torch


def normalize(aa):
    bb = aa - aa.min()
    return bb / bb.max()


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_slope(x, y):
    """Returns the slope of a linear regressor with intercept zero."""
    if (sum(x < 0) > 0) or (sum(y < 0) > 0):
        warn("x and/or y contain negative values. Slope may not reflect the desired quantity.")
    return np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]


def bits_per_image(nits, n_images):
    """

    Args:
        nits:     array containing entropy in nits (natural logarithm, base e)
        n_images: number of images to normalize by

    Returns:      array containing entropy per image in bits (logarithm with base 2)

    """
    return nits / (np.log(2) * n_images)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class ClearPrint:
    def __init__(self):
        self.prev_msg = ""
        self.prev_msg_end = "\r"

    def __call__(self, msg, end="\n"):
        if self.prev_msg_end == "\r":
            print(len(self.prev_msg) * " ", end="\r")
        print(msg, end=end)
        self.prev_msg = msg
        self.prev_msg_end = end


cprint = ClearPrint()


def inf_nan_check(func):
    @functools.wraps(func)
    def wrapper_inf_nan_check(*args, **kwargs):
        outputs = func(*args, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for output in outputs:
            if torch.is_tensor(output):
                assert not (torch.isnan(output).any() or torch.isinf(output).any()), "None or inf value encountered!"

            if type(output).__module__ == "numpy":
                assert not (np.isnan(output).any() or np.isinf(output).any()), "None or inf value encountered!"

        return outputs

    return wrapper_inf_nan_check


class DataCollector:
    def __init__(self, variables):
        self.all_data = {var: [] for var in variables}

    def store_new_data(self):
        for var in self.all_data.keys():
            self.all_data[var].append(globals()[var])

    def get_data(self):
        stacked_data = {}
        for var, values in self.all_data.items():
            if isinstance(values[0], np.ndarray):
                stacked_data[var] = np.stack(values)
            elif isinstance(values[0], torch.Tensor):
                stacked_data[var] = torch.stack(values)
            else:
                stacked_data[var] = values
        return stacked_data
