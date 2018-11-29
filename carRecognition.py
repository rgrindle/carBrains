import time
import glob
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import image_preprocessing as ip
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from keras import layers
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.metrics import classification_report
from scipy.io import loadmat
import pandas as pd
import time
import datetime

"""
These are the main variables to change
"""
num_train_imgs = 8144        # max 8144
num_test_imgs = 8041         # max 8041
image_shape = (100, 100, 1)  # default: (28, 28, 1)

batch_size = 64              # default: 64
epochs = 24                  # default: 20
num_classes = 196            # default: 196

use_flipped = True
use_gaussian = True

repeat_size = 1

if use_flipped:

    repeat_size += 1

if use_gaussian:

    repeat_size += 2
    
num_train_imgs = repeat_size*num_train_imgs

def train_model():

    car_model = Sequential()

    car_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(image_shape[0], image_shape[1], image_shape[2])))
    car_model.add(layers.Activation("relu"))
    car_model.add(layers.BatchNormalization())
    car_model.add(MaxPooling2D((2, 2), padding='same'))
    car_model.add(Dropout(0.25))

    car_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    car_model.add(layers.Activation("relu"))
    car_model.add(layers.BatchNormalization())
    car_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    car_model.add(Dropout(0.25))

    car_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    car_model.add(layers.Activation("relu"))
    car_model.add(layers.BatchNormalization())
    car_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    car_model.add(Dropout(0.4))

    car_model.add(Flatten())

    car_model.add(Dense(128, activation='linear'))
    car_model.add(layers.Activation("relu"))
    car_model.add(layers.BatchNormalization())
    car_model.add(Dropout(0.3))

    car_model.add(Dense(num_classes, activation='softmax'))

    car_model.summary()

    car_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])

    return car_model


# returns an individual label
def get_label(image_number):
    stuff = fp['annotations'][0][image_number - 1]
    label = stuff[4][0][0] - 1
    return label


# returns matrix of numeric labels representing car model, make, year
def get_label_matrix(start, end):
    y=np.array(list(map(get_label, range(start, end))))
    y.reshape((end-start, 1))
    return y.astype('float32')


# returns matrix of image data
def get_image_matrix(directory, start, end):
    image_filepaths = glob.glob(directory)[start-1:end]
    matrix = np.zeros((len(image_filepaths), image_shape[0], image_shape[1]))

    if 'train' in directory:
        using_train = True
    elif 'test' in directory:
        using_train = False
    else:
        print('test or train are not in directory', directory)
        print('exiting')
        exit()

    for i, filepath in enumerate(image_filepaths):
        img = load_img(image_filepaths[i])
        x = img_to_array(img)
        x = x[:,:,0]
        # x = np.mean(x, axis=2)
        #
        # # use bounding box and pad image
        # bb = ip.get_bb(i+start, train=using_train)
        # x = ip.apply_bb(x, bb)
        # x = ip.pad_image(x)

        x = x / 255.
        # x = resize(x, (image_shape[0], image_shape[1]))
        matrix[i, :, :] = x
    return matrix.astype('float32')


start_time = time.time()
base_fp = os.environ['CARS_DATASET_PATH']
fp = loadmat(os.path.normpath(os.path.join(base_fp, "cars_devkit/cars_train_annos.mat")))
input_directory = os.path.join(base_fp, "cars_train_preprocessed/*.jpg")
train_X = get_image_matrix(input_directory, 1, num_train_imgs)
train_Y = get_label_matrix(1, 8144 + 1)
train_Y = np.array([train_Y,]*repeat_size).flatten()
fp = loadmat(os.path.normpath(os.path.join(base_fp, "cars_devkit/cars_test_annos.mat")))
input_directory = os.path.join(base_fp, "cars_test_preprocessed/*.jpg")
test_X = get_image_matrix(input_directory, 1, num_test_imgs)
test_Y = get_label_matrix(1, num_test_imgs + 1)
# Debugging:
# print("Start")
# print(train_Y)
# print(test_Y)

train_X = train_X.reshape(-1, image_shape[0], image_shape[1], image_shape[2])
test_X = test_X.reshape(-1, image_shape[0], image_shape[1], image_shape[2])
# Debugging:
# print("Train X")
# print(train_X.shape)
# print("Test X")
# print(test_X.shape)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
# Debugging:
# print("origional train y one hot test y one hot")
# print(train_Y_one_hot.shape)
# print("break")
# print(test_Y_one_hot.shape)

# Display the change for category label using one-hot encoding, only for debugging purposes
# print('Original label:', train_Y[0])
# print('After conversion to one-hot:', train_Y_one_hot[0])

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

# Debugging:
# print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

img_preprocessing_time = (time.time() - start_time)
start_time = time.time()

car_model = train_model()

# Debugging:
# print("train Y")
# print(train_Y_one_hot.shape)
car_train = car_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

model_train_time = (time.time() - start_time)
start_time = time.time()

# Debugging:
# print("test Y")
# print(test_Y_one_hot.shape)
test_eval = car_model.evaluate(test_X, test_Y_one_hot, verbose=0)

# Debugging:
# print("Full eval")
# print(test_eval)

model_test_time = (time.time() - start_time)

# Prints out the results on a per-class basis
predicted_classes = car_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]

ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
save_path = os.path.join('saved_data', ts)

if not os.path.exists(save_path):

    os.makedirs(save_path)

f = open(os.path.join(save_path, 'log.txt'), 'w')

car_model.summary(print_fn=f.write)

print("Testing Data Classification Report")
print(classification_report(test_Y, predicted_classes, target_names=target_names))

print("Testing Data Classification Report", file=f)
print(classification_report(test_Y, predicted_classes, target_names=target_names), file=f)

# Printing out the results:
print('Testing loss:', test_eval[0])
print('Testing accuracy:', test_eval[1])

print('Testing loss:', test_eval[0], file=f)
print('Testing accuracy:', test_eval[1], file=f)

"""
Original:
predicted_classes = fashion_model.predict(train_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print("Training Data Classification Report")
print(classification_report(train_label, predicted_classes, target_names=target_names))
"""

# Prepping results for plotting
accuracy = car_train.history['acc']
val_accuracy = car_train.history['val_acc']
loss = car_train.history['loss']
val_loss = car_train.history['val_loss']
epochs = range(len(accuracy))

print("\nImage preprocessing took %s seconds" % (img_preprocessing_time))
print("\nTraining the model took %s seconds" % (model_train_time))
print("\nTesting the model took %s seconds" % (model_test_time))

print("\nImage preprocessing took %s seconds" % (img_preprocessing_time), file=f)
print("\nTraining the model took %s seconds" % (model_train_time), file=f)
print("\nTesting the model took %s seconds" % (model_test_time), file=f)

f.close()

# save data
data = np.array([accuracy, val_accuracy, loss, val_loss]).T
df = pd.DataFrame(data, columns=['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])
df.to_csv(os.path.join(save_path, 'acc_and_loss_data.csv'), index_label='Epoch', header=True)

# Plotting the results
plt.figure()

def loss_fig(ax):
    ax.plot(epochs, loss, 'C0', label='Training loss')
    ax.plot(epochs, val_loss, 'C1', label='Validation loss')
    ax.set_title('Loss')
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

def acc_fig(ax):
    # plt.rcParams.update({'font.size': 12})
    ax.plot(epochs, accuracy, 'C0', label='Training Accuracy')
    ax.plot(epochs, val_accuracy, 'C1', label='Validation Accuracy')
    ax.set_title('Accuracy')
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Fraction of Images Correctly Predicted')
    ax.legend()


w = 6.4
h = 4.8

plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(2*w,h))
ax = fig.add_subplot(1,2,1)
loss_fig(ax)
ax = fig.add_subplot(1,2,2)
acc_fig(ax)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'acc_and_loss.png'))

fig = plt.figure(figsize=(w,h))
ax = fig.add_subplot(1,1,1)
loss_fig(ax)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'loss_only.png'))

fig = plt.figure(figsize=(w,h))
ax = fig.add_subplot(1,1,1)
acc_fig(ax)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'acc_only.png'))

# Saves the CNN to a file for analyzing later
car_model.save(os.path.join(save_path, "car_model_dropout.h5py"))
