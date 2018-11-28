import numpy as np
# import matplotlib.pyplot as plt
import scipy.io

import os

def pad_image(img):
    """Make image square with padding."""

    # if square return img
    if img.shape[0] == img.shape[1]:

        return img

    s = max(img.shape)

    if len(img.shape) == 3:

        new_img = np.zeros((s,s,3))

    else:

        new_img = np.zeros((s,s))

    shape_diff = np.abs(np.array(img.shape)-np.array(new_img.shape))
    index = np.argmax(shape_diff)

    # this might be left and right
    top = int(np.floor(shape_diff[index]/2))
    bottom = shape_diff[index] - top

    if index == 0:

        new_img[top:-bottom,:] = img[:,:]

    elif index == 1:

        new_img[:,top:-bottom] = img[:,:]

    return new_img


def apply_bb(img, bb):
    """Crop img to only view area inside bounding box.
    bb = [slice(y_min, y_max), slice(x_min, x_max)]"""

    return img[bb[1],bb[0]]


def get_bb(file_num, train=True):
    """Get the bounding box information for the image with file_num."""

    if train:

        data_info = scipy.io.loadmat(os.environ['CARS_DATASET_PATH'] + "\\cars_devkit\\cars_train_annos.mat")

    else:

        data_info = scipy.io.loadmat(os.environ['CARS_DATASET_PATH'] + "\\cars_devkit\\cars_test_annos.mat")

    img_info = data_info['annotations'][0][file_num-1]

    x_min = img_info[0][0][0]
    y_min = img_info[1][0][0]
    x_max = img_info[2][0][0]
    y_max = img_info[3][0][0]

    return slice(x_min, x_max, 1), slice(y_min, y_max, 1)


# fig = plt.figure()
#
# for i in range(1, 10):
#
#     ax = fig.add_subplot(3,3,i)
#
#     bb = get_bb(i, train=False)
#
#     x = plt.imread(os.environ['CARS_DATASET_PATH'] + '\\cars_test\\'+('%05d' % i)+'.jpg')
#     x2 = apply_bb(x, bb)/255
#     plt.imshow(x2)
#
# fig = plt.figure()
# from skimage.transform import resize
# image_shape = (100, 100)
# for i in range(1, 10):
#
#     ax = fig.add_subplot(3,3,i)
#
#     bb = get_bb(i, train=False)
#
#     x = plt.imread(os.environ['CARS_DATASET_PATH'] + '\\cars_test\\'+('%05d' % i)+'.jpg')
#     x2 = apply_bb(x, bb)/255
#     x2= resize(x2, (image_shape[0], image_shape[1]))
#     plt.imshow(x2)
#
# fig = plt.figure()
#
# for i in range(1, 10):
#
#     ax = fig.add_subplot(3,3,i)
#
#     bb = get_bb(i, train=False)
#
#     x = plt.imread(os.environ['CARS_DATASET_PATH'] + '\\cars_test\\'+('%05d' % i)+'.jpg')
#     x2 = apply_bb(x, bb)
#     x3 = pad_image(x2)/255
#     plt.imshow(x3)
#
# fig = plt.figure()
#
# for i in range(1, 10):
#
#     ax = fig.add_subplot(3,3,i)
#
#     bb = get_bb(i, train=False)
#
#     x = plt.imread(os.environ['CARS_DATASET_PATH'] + '\\cars_test\\'+('%05d' % i)+'.jpg')
#     x2 = apply_bb(x, bb)
#     x3 = pad_image(x2)/255
#     x3 = resize(x3, (image_shape[0], image_shape[1]))
#     plt.imshow(x3)
#
# plt.show()

import glob
from skimage.transform import resize
# from skimage.io import imsave
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

def save_preprocessed(in_dir, out_dir, image_shape):

    skip = 0

    image_filepaths = glob.glob(in_dir)[skip:]

    if 'train' in in_dir:
        using_train = True
    elif 'test' in in_dir:
        using_train = False
    else:
        print('test or train are not in directory', in_dir)
        print('exiting')
        exit()

    if not os.path.exists(os.path.dirname(out_dir)):

        os.makedirs(out_dir)

    for i, filepath in enumerate(image_filepaths):

        img = load_img(image_filepaths[i])
        x = img_to_array(img)
        x = np.mean(x, axis=2)

        # use bounding box and pad image
        bb = get_bb(i+1+skip, train=using_train)
        x = apply_bb(x, bb)
        x = pad_image(x)

        x = resize(x, (image_shape[0], image_shape[1]))
        x.astype(int)

        plt.imsave(os.path.join(out_dir, ('%05d' % (i+1+skip))+'.jpg'), x, cmap='gray')


def save_modified(in_dir, out_dir, name_offset, augment_func):

    skip = 0

    image_filepaths = glob.glob(in_dir)[skip:8144]

    if not os.path.exists(os.path.dirname(out_dir)):

        os.makedirs(out_dir)

    for i, filepath in enumerate(image_filepaths):

        img = load_img(image_filepaths[i])

        x = augment_func(img)

        plt.imsave(os.path.join(out_dir, ('%05d' % (i+1+skip+name_offset))+'.jpg'), x)


if __name__ == "__main__":

    # for data_type in ('train', 'test'):

        # in_dir = os.path.join(os.environ['CARS_DATASET_PATH'], 'cars_'+data_type+'/*.jpg')
        # out_dir = os.path.join(os.environ['CARS_DATASET_PATH'], 'cars_'+data_type+'_preprocessed/')
        # image_shape = (100, 100, 1)
        #
        # save_preprocessed(in_dir, out_dir, image_shape)

    data_type = 'train'

    in_dir  = os.path.join(os.environ['CARS_DATASET_PATH'], 'cars_' + data_type + '_preprocessed')+'\\*.jpg'
    out_dir = os.path.dirname(in_dir)

    # flip
    # f = lambda  img: np.fliplr(img)
    #
    # save_modified(in_dir, out_dir, 8144, f)

    # add gaussian noise
    def g(img):

        x = img_to_array(img)
        noise = np.random.normal(0, 4, size=(100, 100, 3))
        np.clip(img+noise, 0, 255)
        return x.astype('uint8')

    save_modified(in_dir, out_dir, 2*8144, g)
