import numpy as np
from scipy.io import loadmat
import os
import shutil

split_dict = loadmat('datasplits.mat')

print(split_dict.keys())

# train1 = np.array(split_dict['trn1'])
# val1 = np.array(split_dict['val1'])
# test1 = np.array(split_dict['tst1'])
#
# train2 = np.array(split_dict['trn2'])
# val2 = np.array(split_dict['val2'])
# test2 = np.array(split_dict['tst2'])
#
# train3 = np.array(split_dict['trn3'])
# val3 = np.array(split_dict['val3'])
# test3 = np.array(split_dict['tst3'])
#
#
# print(np.sum(train1) + np.sum(val1) + np.sum(test1))
# print(np.sum(train2) + np.sum(val2) + np.sum(test2))
# print(np.sum(train3) + np.sum(val3) + np.sum(test3))
#
# print(np.sum(train3) + np.sum(val2) + np.sum(test1))

# Make the directories
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('val'):
    os.mkdir('val')
if not os.path.exists('test'):
    os.mkdir('test')

# Make the sub-directories for each class under train/val/test dir
for set_dir in ['train', 'val', 'test']:
    for i in range(1, 18):
        if not os.path.exists(set_dir + '/' + str(i)):
            os.mkdir(set_dir + '/' + str(i))

train_ids = np.array(split_dict['trn1'])[0]
val_ids = np.array(split_dict['val1'])[0]
test_ids = np.array(split_dict['tst1'])[0]

for id in train_ids:
    filename = 'image_{:0>4d}.jpg'.format(id)
    class_name = str(int((id - 1) / 80 + 1))
    src_path = 'jpg/' + filename
    dst_path = 'train/' + class_name + '/' + filename
    print(src_path, dst_path)
    shutil.copy(src=src_path, dst=dst_path)

for id in val_ids:
    filename = 'image_{:0>4d}.jpg'.format(id)
    class_name = str(int((id - 1) / 80 + 1))
    src_path = 'jpg/' + filename
    dst_path = 'val/' + class_name + '/' + filename
    print(src_path, dst_path)
    shutil.copy(src=src_path, dst=dst_path)

for id in test_ids:
    filename = 'image_{:0>4d}.jpg'.format(id)
    class_name = str(int((id - 1) / 80 + 1))
    src_path = 'jpg/' + filename
    dst_path = 'test/' + class_name + '/' + filename
    print(src_path, dst_path)
    shutil.copy(src=src_path, dst=dst_path)

