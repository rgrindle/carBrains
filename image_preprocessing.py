import numpy as np
# import matplotlib.pyplot as plt
import scipy.io

import os

def pad_image(img):
    """Make image square with padding."""

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