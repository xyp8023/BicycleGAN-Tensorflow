import os
from glob import glob

from scipy.misc import imread, imresize, imshow
import numpy as np
from tqdm import tqdm
import h5py
from shutil import copy
from random import sample


def read_image(path):
    image = imread(path)
    # if len(image.shape) != 3 or image.shape[2] != 3:
    #     print('Wrong image {} with shape {}'.format(path, image.shape))
    #     return None

    # split image
    h, w= image.shape
    assert w in [256, 512, 1200], 'Image size mismatch ({}, {})'.format(h, w)
    assert h in [128, 256, 600], 'Image size mismatch ({}, {})'.format(h, w)

    # imshow(image[:, :w/2])
    # imshow(image[:, w/2:])
    image_a = image[:, :w/2].astype(np.float32) / 255.0
    image_b = image[:, w/2:].astype(np.float32) / 255.0
    # imshow(image_a)
    # imshow(image_b)
    # range of pixel values = [-1.0, 1.0]
    image_a = image_a * 2.0 - 1.0
    image_b = image_b * 2.0 - 1.0
    image_a = image_a.reshape(h, h, 1)
    image_b = image_b.reshape(h, h, 1)

    return image_a, image_b

def read_images(base_dir):
    ret = []
    for dir_name in ['train', 'val']:
        data_dir = os.path.join(base_dir, dir_name)
        paths = glob(os.path.join(data_dir, '*.png'))
        print('# images in {}: {}'.format(data_dir, len(paths)))

        images_A = []
        images_B = []
        for path in tqdm(paths):
            image_A, image_B = read_image(path)
            if image_A is not None:
                images_A.append(image_A)
                images_B.append(image_B)
        ret.append((dir_name + 'A', images_A))
        ret.append((dir_name + 'B', images_B))
    return ret

def seperate_train_valid(base_dir, create_valid):
    for dir_name in ['train', 'val']:
        data_dir = os.path.join(base_dir, dir_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    if create_valid:
        N = (len(glob(base_dir + '/*.png')))
        print N
        val_idx = np.random.randint(1, N, 100)
        val_idx = sample(range(N), 100)
        # val_idx=np.random.choice(N, 100)
        print len(val_idx)
        for i in range(N):
            if i in val_idx:
                # print i
                os.rename(base_dir + '/'+str(i)+'.png', base_dir+'/val/'+str(i)+'.png')
                # copy(base_dir + '/'+str(i)+'.png', base_dir+'/val')
            else:
                os.rename(base_dir + '/'+str(i)+'.png', base_dir+'/train/'+str(i)+'.png')
                # copy(base_dir + '/' + str(i) + '.png', base_dir + '/train')


# base_dir = 'datasets/sss2depth/pix2pix_waterfall_full'
# seperate_train_valid(base_dir, create_valid=False)
# ret = read_images(base_dir)

def store_h5py(base_dir, dir_name, images, image_size):
    f = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'w')
    for i in range(len(images)):
        grp = f.create_group(str(i))
        if images[i].shape[0] != image_size:
            image = imresize(images[i], (image_size, image_size, 1))
            # range of pixel values = [-1.0, 1.0]
            image = image.astype(np.float32) / 255.0
            image = image * 2.0 - 1.0
            grp['image'] = image
        else:
            grp['image'] = images[i]
    f.close()

def convert_h5py(base_dir):
    print('Generating h5py file')
    # base_dir = os.path.join('datasets', task_name)
    data = read_images(base_dir)
    for dir_name, images in data:

        store_h5py(base_dir, dir_name, images, 256)


def read_h5py(base_dir, image_size):
    # base_dir = 'datasets/' + task_name
    paths = glob(os.path.join(base_dir, '*_{}.hy'.format(image_size)))
    if len(paths) != 4:
        convert_h5py(base_dir)
    ret = []
    for dir_name in ['trainA', 'trainB', 'valA', 'valB']:
        try:
            dataset = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'r')
        except:
            raise IOError('Dataset is not available. Please try it again')

        images = []
        for id in dataset:
            images.append(dataset[id]['image'].value.astype(np.float32))
        ret.append(images)
    return ret

def get_data(base_dir, image_size, convert_flag):

    # base_dir = os.path.join('datasets', task_name)
    # print('Check data %s' % base_dir)
    # if not os.path.exists(base_dir):
    #     print('Dataset not found. Start downloading...')
        # download_dataset(task_name)
    if convert_flag:
        convert_h5py(base_dir)

    # print('Load data %s' % task_name)
    train_A, train_B, test_A, test_B = \
        read_h5py(base_dir, image_size)
    return train_A, train_B, test_A, test_B

base_dir = 'datasets/sss2depth/pix2pix_waterfall_full'
# seperate_train_valid(base_dir, create_valid=False)
# train_A, train_B, test_A, test_B = get_data(base_dir, 256, convert_flag=True)
# print 'done'


# seperate_train_valid(base_dir, create_valid=True)