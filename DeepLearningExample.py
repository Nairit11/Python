# Packages to prepare data
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

# Preparing data to train and validate model
def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)

# Sample Model Code
fashion_model = Sequential()
fashion_model.add(Conv2D(12, kernel_size=(3, 3),       # no,of filters in Conv layer and size of convolving tensor
                 activation='relu',                    # Activation function
                 input_shape=(img_rows, img_cols, 1))) # Input data only for first layer
fashion_model.add(Conv2D(12, (3, 3), activation='relu'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)

# Optimized Model
fashion_model_3 = Sequential()
fashion_model_3.add(Conv2D(24, kernel_size=(3, 3), strides=2,    # Strides determine stepsize of the convolving tensor in the training data, increases speed
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_3.add(Dropout(0.5))                                # Avoids overfitting, by randomly choosing 50% of the previous layer's data
fashion_model_3.add(Conv2D(24, (3, 3), strides=2, activation='relu'))
fashion_model_3.add(Dropout(0.5))
fashion_model_3.add(Conv2D(24, (3, 3), activation='relu'))
fashion_model_3.add(Dropout(0.5))
fashion_model_3.add(Flatten())
fashion_model_3.add(Dense(128, activation='relu'))
fashion_model_3.add(Dense(num_classes, activation='softmax'))

fashion_model_3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model_3.fit(x, y,
          batch_size= 128,
          epochs= 2,
          validation_split = 0.2)


