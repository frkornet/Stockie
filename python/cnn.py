#
# Based upon "Deep Learning with Python" by Jason Brownlee. 
# The code is packaged into functions so, all the different pieces
# are availble for learning, testing, and applying to Stockie.
# The focus is here on Part V: Convolutional Neural Networks
#

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils  import np_utils

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras.constraints import maxnorm 
from keras.optimizers  import SGD

from keras.datasets import mnist
from keras.datasets import cifar10

import matplotlib.pyplot as plt

from util import get_starttime, calc_runtime
from symbols import PICPATH

#
# Chapter 18: Project: Handwritten Digit Recognition
#
def initial_load_mnist_dataset():
    # load (downloaded if needed) the MNIST dataset 
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    
    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray')) 
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap('gray')) 
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray')) 
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    plt.show()

def mnist_mlp_baseline():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32') 
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test) 
    num_classes = y_test.shape[1]
    
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu')) 
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    
    # build, fit, and evaluate the model
    model = baseline_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, 
        batch_size=200, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=0) 
    
    # print error rate
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))



def mnist_simple_cnn():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to be [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # define a simple CNN model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu')) 
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
            metrics=['accuracy']) 
        return model
    
    # build, fit, and evaluate the model
    model = baseline_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, 
        batch_size=200)
    scores = model.evaluate(X_test, y_test, verbose=0) 
    
    # print error rate
    print("CNN Error: %.2f%%" % (100-scores[1]*100))

def mnist_larger_cnn():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to be [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train) 
    y_test = np_utils.to_categorical(y_test) 
    num_classes = y_test.shape[1]
    
    def larger_model():
        # create model
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu')) 
        model.add(MaxPooling2D())
        model.add(Conv2D(15, (3, 3), activation='relu')) 
        model.add(MaxPooling2D()) 
        model.add(Dropout(0.2))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
            metrics=['accuracy']) 
        return model
    
    # build, fit, and evaluate the model
    model = larger_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, 
        batch_size=200)
    scores = model.evaluate(X_test, y_test, verbose=0)

    # print error rate of larger model
    print("Large CNN Error: %.2f%%" % (100-scores[1]*100))    

#
# Chapter 19: Improve Model Performance With Image Augmentation
#
def load_n_plot_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() # create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()

def standardize_image():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    # reshape to be [samples][width][height][channels] 
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # define data preparation
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True) 
    
    # fit parameters from data
    datagen.fit(X_train)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9): 
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray')) 
        
        # show the plot
        plt.show()
        break

def zca_whiten_image():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    # reshape to be [samples][width][height][channels] 
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # define data preparation
    datagen = ImageDataGenerator(zca_whitening=True) 
    
    # fit parameters from data
    datagen.fit(X_train)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9): 
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray')) 
        
        # show the plot
        plt.show()
        break


def random_rotate_image():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    # reshape to be [samples][width][height][channels] 
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # define data preparation
    datagen = ImageDataGenerator(rotation_range=90) 
    
    # fit parameters from data
    datagen.fit(X_train)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9): 
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray')) 
        
        # show the plot
        plt.show()
        break

def random_shift_image():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    # reshape to be [samples][width][height][channels] 
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # define data preparation
    shift=0.2
    datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift) 
    
    # fit parameters from data
    datagen.fit(X_train)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9): 
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray')) 
        
        # show the plot
        plt.show()
        break

def random_flip_image():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    # reshape to be [samples][width][height][channels] 
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # define data preparation
    shift=0.2
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True) 
    
    # fit parameters from data
    datagen.fit(X_train)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9): 
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray')) 
        
        # show the plot
        plt.show()
        break

def save_augmented_image():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    # reshape to be [samples][width][height][channels] 
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # define data preparation
    shift=0.2
    datagen = ImageDataGenerator() 
    
    # fit parameters from data
    datagen.fit(X_train)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, 
                                save_to_dir=PICPATH, save_prefix='aug', 
                                save_format='png'):
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray')) 
        
        # show the plot
        plt.show()
        break


#
# Chapter 20: Project Object Recognition in Photographs
#




def initial_load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() 
    
    # create a grid of 3x3 images
    for i in range(0, 9): 
        plt.subplot(330 + 1 + i) 
        plt.imshow(X_train[i])

    # show the plot
    plt.show()    

def cifar10_simple_cnn():
    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', 
                     activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')) 
    
    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
    model.compile(loss='categorical_crossentropy', optimizer=sgd, 
                  metrics=['accuracy']) 
    model.summary()

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=epochs, batch_size=32)
    
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print("Accuracy: %.2f%%" % (scores[1]*100))

def cifar10_larger_cnn():
    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32') 
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train) 
    y_test = np_utils.to_categorical(y_test) 
    num_classes = y_test.shape[1]

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    epochs = 125
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.summary()

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64) 
    
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def print_runtime(func):
    start = get_starttime()
    func()
    calc_runtime(start, True)

if __name__ == "__main__":
    # funcs_to_run = [ initial_load_mnist_dataset, mnist_mlp_baseline,
    #                  mnist_simple_cnn, mnist_larger_cnn, load_n_plot_mnist,
    #                  standardize_image, zca_whiten_image,
    #                  random_rotate_image, random_shift_image, 
    #                  random_flip_image, save_augmented_image,
    #                  initial_load_cifar10, cifar10_simple_cnn
    #   ]
    #
    funcs_to_run = [ cifar10_larger_cnn
                ]

    for i, f in enumerate(funcs_to_run):
        stars="*"*80
        print(f'{stars}\n{stars}')
        print(f'run={i} function={f}\n')
        print_runtime(f)
        print(f'{stars}\n{stars}\n')
