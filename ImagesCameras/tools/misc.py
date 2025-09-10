import collections.abc
import math
import ntpath
import os
import shutil
import stat
import time


# Display functions ##################################

def print_tuple(t, link=' '):
    t = [*t]
    string = ''
    for el in t:
        string += str(el)
        string += link
    return string


def path_leaf(path):
    if os.path.isdir(path):
        res = ntpath.split(path)[-1]
    else:
        res = ntpath.split(path)[-1].split(".")[:-1]
    if isinstance(res, list):
        if len(res) > 1:
            res = ''.join(res)
        else:
            res = res[0]
    return res


def paired_keys(value: dict, occ: bool = False):
    keys = list(value.keys())
    occ = occ if 'new_occ' in keys else False
    if occ:
        keys.remove('new')
        keys.remove('ref')
    ref = sorted([k for k in keys if 'ref' in k])
    new = sorted([k for k in keys if 'new' in k])
    return zip(new, ref)


def time_fct(func, reps=1, exclude_first=False, verbose=True):
    reps = max(1, reps)

    def wrapper(*args, **kwargs):
        res = None
        if exclude_first:
            start = time.time()
            res = func(*args, **kwargs)
            first = time.time() - start
        start = time.time()
        for i in range(reps):
            res = func(*args, **kwargs)
        timed = time.time() - start
        if verbose:
            print("------------------------------------ TIME FUNCTION ---------------------------------------------")
            try:
                print(
                    f"Function {func.__name__} executed  {reps} times in : {timed} seconds, average = {timed / reps} seconds"
                    f"{f', first occurence : {first}' if exclude_first else ''}        \r",)
            except AttributeError:
                print(
                    f"\nFunction {func.__class__.__name__} executed  {reps} times in : {timed} seconds, average = {timed / reps} seconds"
                    f"{f', first occurence : {first}' if exclude_first else ''}        \r",)
            print("------------------------------------------------------------------------------------------------")
            return res
        else:
            return res, timed / reps

    return wrapper


def name_generator(idx, max_number=10e4):
    k_str = str(idx)
    digits = 1 if max_number < 10 else int(math.log10(max_number)) + 1
    current_digits = 1 if idx < 10 else int(math.log10(idx)) + 1
    for i in range(digits - current_digits):
        k_str = '0' + k_str
    return k_str
