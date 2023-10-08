"""
Split QUT_fish images into subdirectories
"""

import os
import shutil
from itertools import accumulate

import numpy as np

from numpy import loadtxt
from numpy.random.mtrand import shuffle

np.random.seed(2018)

basedir = '/media/ouc/4T_A/DuAngAng/datasets/QUT_fish/QUT_fish_data/'

image_basedir = basedir + 'images/cropped/'

index_filename = 'final_all_index.txt'

# 'class number', 'class name', 'controlled/insitu/uncontrolled/sketches/rubbish', 'filename', 'id'
index_file_list = loadtxt(basedir + index_filename, dtype=str, delimiter='=')   # numpy.ndarray
# print(len(index_file_list))     # 4411
# print(index_file_list)

# Ignore 'sketches' images, 444 in total
sketches_image_indices = index_file_list[:, 2] == 'sketches'
# print(len(index_file_list[sketches_image_indices]))  # 444
index_file_list = index_file_list[~sketches_image_indices]

# Ignore 'rubbish' images, 7 in total
rubbish_image_indices = index_file_list[:, 2] == 'rubbish'
# print(len(index_file_list[sketches_image_indices]))  # 7
index_file_list = index_file_list[~rubbish_image_indices]

# print(len(index_file_list))     # 3960
# print(index_file_list)

sample_class_name_list = [class_name for class_name in index_file_list[:, 1]]
# print(sample_class_name_list)
# class_name_list = np.unique(sample_class_name_list)  # `unique` will break the original order
class_name_list = []
for class_name in sample_class_name_list:
    if class_name not in class_name_list:
        class_name_list.append(class_name)
# print(len(class_name_list))     # 482
# print(class_name_list)
# Create directories for training set and test set
for class_name in class_name_list:
    if not os.path.exists(basedir + 'train/' + class_name):
        os.makedirs(basedir + 'train/' + class_name)
    if not os.path.exists(basedir + 'test/' + class_name):
        os.makedirs(basedir + 'test/' + class_name)

# count the number of samples in each class
class_samples_count_list = np.zeros(len(class_name_list), dtype=int)
# print(len(class_samples_count_list))
for i, class_name in enumerate(class_name_list):
    class_samples_count_list[i] = sum([class_name == name for name in sample_class_name_list])
# print(class_samples_count_list)

# print('min class number: {}'.format(np.argmin(class_samples_count_list)+1))     # 12
# print('min class samples num: {}'.format(np.min(class_samples_count_list)))     # 2
# print('max class number: {}'.format(np.argmax(class_samples_count_list)+1))     # 74
# print('max class samples num: {}'.format(np.max(class_samples_count_list)))     # 26

class_samples_index_list = [i for i in accumulate(class_samples_count_list)]
# print(class_samples_index_list)
# print(len(class_samples_index_list))

# Count the images of different type
# image_type_list = [type for type in index_file_list[:, 2]]
# print(image_type_list)
# print(np.unique(image_type_list))
# print(sum([type == 'controlled' for type in image_type_list]))      # 2483
# print(sum([type == 'insitu' for type in image_type_list]))          # 1385
# print(sum([type == 'uncontrolled' for type in image_type_list]))    # 92


# There are three kinds of images: 'controlled', 'insitu' and 'uncontrolled'
# we prefer to make 'controlled' images as the training set,
#  and make 'uncontrolled' and 'insitu' images as the test set
def train_test_split(samples_list, train_test_ratio=0.5):
    # print(samples_list)
    test_set_samples_count = int(len(samples_list) * 0.5)
    # print(test_set_samples_count)
    uncontrolled_samples_list = samples_list[samples_list[:, 2] == 'uncontrolled']
    insitu_samples_list = samples_list[samples_list[:, 2] == 'insitu']
    controlled_samples_list = samples_list[samples_list[:, 2] == 'controlled']
    shuffle(uncontrolled_samples_list)
    shuffle(insitu_samples_list)
    shuffle(controlled_samples_list)
    rearranged_samples_list = []
    rearranged_samples_list.extend(uncontrolled_samples_list)
    rearranged_samples_list.extend(insitu_samples_list)
    rearranged_samples_list.extend(controlled_samples_list)
    test_set_samples_list = rearranged_samples_list[:test_set_samples_count]
    training_set_samples_list = rearranged_samples_list[test_set_samples_count:]
    return training_set_samples_list, test_set_samples_list


src_dst_pair_list = []
train_test_ratio = 0.5
print('Splitting...')
for i, class_samples_count in enumerate(class_samples_count_list):
    # print('the {}-th class, {} samples in total'.format(i, class_samples_count))
    if i == 0:
        the_ith_class_samples_list = index_file_list[0:class_samples_index_list[i]]
        # print('samples index {}:{}'.format(0, class_samples_index_list[i]))
    else:
        the_ith_class_samples_list = index_file_list[class_samples_index_list[i-1]:class_samples_index_list[i]]
        # print('samples index {}:{}'.format(class_samples_index_list[i-1], class_samples_index_list[i]))
    # print(the_ith_class_samples_list)

    # split the i-th class samples into train/test (0.5/0.5)
    training_set_samples_list, test_set_samples_list = train_test_split(the_ith_class_samples_list,
                                                                        train_test_ratio=train_test_ratio)

    for sample_item in training_set_samples_list:
        src_path = image_basedir + sample_item[3] + '.png'
        dst_path = basedir + 'train/' + sample_item[1] + '/' + sample_item[3] + '.png'
        src_dst_pair_list.append([src_path, dst_path])
    for sample_item in test_set_samples_list:
        src_path = image_basedir + sample_item[3] + '.png'
        dst_path = basedir + 'test/' + sample_item[1] + '/' + sample_item[3] + '.png'
        src_dst_pair_list.append([src_path, dst_path])

# print(src_dst_pair_list)

print('Start copying files...')
# Copy files to the training set and test set
for src_path, dst_path in src_dst_pair_list:
    shutil.copy(src_path, dst_path)
print('Completed!')










