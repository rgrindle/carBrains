# from scipy.misc import imread, imsave, imresize
# import os
# import numpy as np
#
# def is_gray(f_name):
#     image = imread(f_name)
#     if(len(image.shape)<3):
#           rgb = np.stack((image,)*3, axis=-1)
#           imsave(f_name, rgb)
#           print(f_name)
#           return True
#     elif len(image.shape)==3:
#           return False
#     else:
#           print('others')
#
# files = os.listdir('C:\\Users\\Ryan\\Documents\\cars_dataset\\cars_train')
# # print(files)
#
# for f in files:
#
#     g = is_gray('C:\\Users\\Ryan\\Documents\\cars_dataset\\cars_train\\'+f)
#
#     if g:
#         print(f, g)


# import tensorflow as tf
# import numpy as np
# import PIL
#
import os
from scipy.io import loadmat

def get_label(image_number):
    x = loadmat(os.environ['CARS_DATASET_PATH'] + '/cars_devkit/cars_train_annos.mat')
    stuff = x['annotations'][0][image_number - 1]
    label = stuff[4][0][0]
    return label



# print(os.environ)
print(os.environ['CARS_DATASET_PATH'])

class_key = loadmat(os.environ['CARS_DATASET_PATH']+'/cars_devkit/cars_meta.mat')['class_names'][0]

image_number = 6221
print('get_label_test', get_label(image_number))
x = loadmat(os.environ['CARS_DATASET_PATH']+'/cars_devkit/cars_train_annos.mat')
stuff = x['annotations'][0][image_number-1]
label = stuff[4][0][0]
img = stuff[5]
print('stuff', stuff)
print('stuff\'s shape', stuff.shape)
print('label', label)
print('img', img)
print('label string', class_key[label-1])
# lon = x['lon']
# lat = x['lat']
# one-liner to read a single variable
# lon = loadmat('test.mat')['lon']

# filename_queue = tf.train.string_input_producer(['C:/Users/Ryan/Documents/cars_dataset/cars_train/00001.jpg']) #  list of files to read
#
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
#
# my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
#
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#   sess.run(init_op)
#
#   # Start populating the filename queue.
#
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#
#   for i in range(1): #length of your filename list
#     image = my_img.eval() #here is your image Tensor :)
#
#   print(image.shape)
#   PIL.Image.fromarray(np.asarray(image)).show()
#
#   coord.request_stop()
#   coord.join(threads)