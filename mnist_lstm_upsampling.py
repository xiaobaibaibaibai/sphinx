'''
after we get prediction, we use upsampling for binary_corssentropy loss
# model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
'''

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, LSTM
from tensorflow.keras.layers import Flatten, Activation, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('input_shape: ', input_shape)

model = Sequential()
# (28, 1)
model.add(Conv1D(32, kernel_size=3,
                 activation='relu',
                 input_shape=(28, 1)))
# (None, 26, 32)
model.add(Conv1D(64,  3, activation='relu'))
# (None, 24, 64)
model.add(MaxPooling1D(pool_size=2))
# (None, 12, 64)
model.add(Dropout(0.25))
# (None, 12, 64)

model.add(Flatten())
model.add(Dense(768))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.25))
# (None, 768)

# print(model.summary())

lstm_model = Sequential()
lstm_model.add(Reshape((28, 768), input_shape=(28, 768)))
lstm_model.add(LSTM(1024, batch_input_shape=(32, 1, 768), dropout=0.15, return_sequences=True))
lstm_model.add(BatchNormalization())
lstm_model.add(Dense(768))


lstm_model.add(Reshape((28, 12, 64)))





print(lstm_model.summary())

series_input = Input(shape=(28, 28, 1))
encoded_series_input = TimeDistributed(model)(series_input)
series_output = lstm_model(encoded_series_input)
cnn_lstm_model = Model(inputs = series_input, outputs = series_output)