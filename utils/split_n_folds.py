"""
Split files into n-folds.
"""

import os
import shutil
from glob import glob
import random


def shuffle2(data):
    """shuffle elements in data, but not in place"""
    data2 = data[:]
    random.shuffle(data2)
    return data2


def get_subdirs(base_dir):
    """get subdirectories"""
    return [i for i in glob(base_dir + "/*/") if os.path.isdir(i)]


def get_orig_struct(base_dir):
    """represent original file structure in a dictory"""
    assert os.path.isdir(base_dir)
    sub_dirs = get_subdirs(base_dir)
    original_structure = {subd: os.listdir(subd) for subd in sub_dirs}
    return original_structure


def make_folds_dict(orig_struct_dict, n):
    """
    Create dictionary representing fold-1/fold-2/.../fold-n/ file structure
    """
    assert isinstance(orig_struct_dict, dict)
    n_folds_dict = {'fold-{}'.format(i): {} for i in range(1, n+1)}

    for key, value in orig_struct_dict.items():
        fold_data_list = n_folds_split(data=value, n=n)
        for i in range(1, n+1):
            n_folds_dict['fold-{}'.format(i)].update({key: fold_data_list[i-1]})
    return n_folds_dict


def n_folds_split(data, n):
    each_fold_data_nums = int(round(float(len(data)) / n))
    fold_data_nums_list = [each_fold_data_nums for i in range(1, n)]
    fold_data_nums_list.append(len(data)-each_fold_data_nums*(n-1))
    data = shuffle2(data)
    fold_data_list = []
    for i in range(n):
        fold_data_list.append(data[each_fold_data_nums*i:sum(fold_data_nums_list[:i+1])])
    # check nothing's been lost
    assert sum(len(i) for i in fold_data_list) == len(data)
    return fold_data_list


def get_label(filename):
    """
    Get label from base_dir filename as seen in the dictioinaries
    e.g. if the key is "example/data/apples", this will return "apples"
    """
    return os.path.basename(os.path.normpath(filename))


def check_dir(filename):
    """
    Given a path, checks the sub-directories exists, if not then tries to
    create them
    """
    subdir_path = os.path.dirname(filename)
    try:
        os.makedirs(subdir_path)
    except OSError:
        if os.path.isdir(subdir_path):
            pass
        else:
            err_msg = "failed to create directory {}".format(subdir_path)
            raise RuntimeError(err_msg)


def create_path_lists(source, destination, n):
    """
    Return two lists, the original filepaths and the destination filepaths
    """
    orig_struct = get_orig_struct(base_dir=source)
    n_folds_dict = make_folds_dict(orig_struct, n)
    original_path = []
    new_path = []
    for fold_name, subdict in n_folds_dict.items():
        for class_labels, files in subdict.items():
            orig = [os.path.join(class_labels, i) for i in files]
            dest = [os.path.join(destination, fold_name,
                    get_label(class_labels), i) for i in files]
            original_path.extend(orig)
            new_path.extend(dest)
    return original_path, new_path


def n_folds_split_dir(source, destination, n):
    original_paths, new_paths = create_path_lists(
        source=source, destination=destination, n=n)
    for i, j in zip(original_paths, new_paths):
        check_dir(j)  # check if the destination directory exists, create it if needed
        shutil.copy2(i, j)

n_folds_split_dir(source='/media/ouc/4T_A/DuAngAng/datasets/F4K/fish_image/',
                  destination='/media/ouc/4T_A/DuAngAng/datasets/F4K/5-folds/', n=5)
