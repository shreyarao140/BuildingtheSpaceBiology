from scipy.io import loadmat
import os
import shutil

image_labels_dict = loadmat('imagelabels.mat')
set_ids_dict = loadmat('setid.mat')

print(image_labels_dict.keys())
print(set_ids_dict.keys())

# Make the directories
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('val'):
    os.mkdir('val')
if not os.path.exists('test'):
    os.mkdir('test')

# Make the sub-directories for each class under train/val/test dir
for set_dir in ['train', 'val', 'test']:
    for i in range(1, 103):
        if not os.path.exists(set_dir + '/' + str(i)):
            os.mkdir(set_dir + '/' + str(i))

labels = image_labels_dict['labels'][0]
train_ids = set_ids_dict['trnid'][0]
val_ids = set_ids_dict['valid'][0]
test_ids = set_ids_dict['tstid'][0]

# Check the number of labels and ids equal
# print(len(labels))
# print(len(train_ids) + len(val_ids) + len(test_ids))

for id in train_ids:
    filename = 'image_{:0>5d}.jpg'.format(id)
    class_name = str(labels[id-1])
    src_path = 'jpg/' + filename
    dst_path = 'train/' + class_name + '/' + filename
    print(src_path, dst_path)
    shutil.copy(src=src_path, dst=dst_path)

for id in val_ids:
    filename = 'image_{:0>5d}.jpg'.format(id)
    class_name = str(labels[id-1])
    src_path = 'jpg/' + filename
    dst_path = 'val/' + class_name + '/' + filename
    print(src_path, dst_path)
    shutil.copy(src=src_path, dst=dst_path)

for id in test_ids:
    filename = 'image_{:0>5d}.jpg'.format(id)
    class_name = str(labels[id-1])
    src_path = 'jpg/' + filename
    dst_path = 'test/' + class_name + '/' + filename
    print(src_path, dst_path)
    shutil.copy(src=src_path, dst=dst_path)

