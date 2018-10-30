"""
This is my attempt to far to get the fashion tutorial working with the images from our dataset.
I think we are close to getting it working.
The code is obviously very sloppy but we can clean it up later.
"""

import glob
import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from scipy.io import loadmat
from sklearn.metrics import classification_report

num_train_imgs = 1001 #max 8144 #this gets -1
num_test_imgs = 1001 #max 8041 #this gets -1
image_shape = (100, 100, 1)

def train_model():
    batch_size = 64
    epochs = 20
    num_classes = 196

    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(image_shape[0], image_shape[1], image_shape[2])))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation='softmax'))

    fashion_model.summary()

    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return (fashion_model)

# returns an individual label
def get_label(image_number):
    stuff = fp['annotations'][0][image_number - 1]
    label = stuff[4][0][0] - 1
    return label

# returns matrix of numeric labels representing car model, make, year
def get_label_matrix(start, end):
    y=np.array(list(map(get_label, range(start, end))))
    y.reshape((end-start, 1))
    return (y.astype('float32'))

# returns matrix of image data
def get_image_matrix(directory, start, end):
    image_filepaths = glob.glob(directory)[start:end]
    matrix = np.zeros((len(image_filepaths), image_shape[0], image_shape[1]))

    for i, filepath in enumerate(image_filepaths):
        img = load_img(image_filepaths[i])
        x = img_to_array(img)
        x = np.mean(x, axis=2)
        x = x / 255.
        x = resize(x, (image_shape[0], image_shape[1]))
        matrix[i, :, :] = x
    return (matrix.astype('float32'))

fp = loadmat(os.path.normpath(os.path.join(os.environ['CARS_DATASET_PATH'], "cars_devkit\\cars_train_annos.mat")))
input_directory = os.path.join(os.environ['CARS_DATASET_PATH'], "cars_train/*.jpg")
train_X = get_image_matrix(input_directory, 1, num_train_imgs)
train_Y = get_label_matrix(1, num_train_imgs)
fp = loadmat(os.path.normpath(os.path.join(os.environ['CARS_DATASET_PATH'], "cars_devkit\\cars_test_annos.mat")))
input_directory = os.path.join(os.environ['CARS_DATASET_PATH'], "cars_test/*.jpg")
test_X = get_image_matrix(input_directory, 1, num_test_imgs)
test_Y = get_label_matrix(1, num_test_imgs)
print("Start")
print(train_Y)
print(test_Y)

train_X = train_X.reshape(-1, image_shape[0], image_shape[1], image_shape[2])
test_X = test_X.reshape(-1, image_shape[0], image_shape[1], image_shape[2])
print("Train X")
print(train_X.shape)
print("Test X")
print(test_X.shape)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
print("origional train y one hot test y one hot")
print(train_Y_one_hot.shape)
print("break")
print(test_Y_one_hot.shape)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

batch_size = 64
epochs = 20
num_classes = 196

fashion_model = train_model()

print("train Y")
print(train_Y_one_hot.shape)
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

print("test Y")
print(test_Y_one_hot.shape)
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
print("Full eval")
print(test_eval)

test_eval2 = fashion_model.evaluate(train_X[0:10], train_Y_one_hot[0:10], verbose=0)
print("ts")
print(test_eval2)

predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print("Testing Data Classification Report")
print(classification_report(test_Y, predicted_classes, target_names=target_names))

"""
predicted_classes = fashion_model.predict(train_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print("Training Data Classification Report")
print(classification_report(train_label, predicted_classes, target_names=target_names))
"""

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
"""
plt.figure()
plt.rcParams.update({'font.size': 22})
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Proportion Images Correctly Predicted')
plt.legend()
plt.show()
"""
plt.figure()
plt.rcParams.update({'font.size': 22})
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


fashion_model.save("fashion_model_dropout.h5py")
