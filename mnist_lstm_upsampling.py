'''
after we get prediction, we use upsampling for binary_corssentropy loss
# model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, LSTM
from tensorflow.keras.layers import Flatten, Activation, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import TimeDistributed, LeakyReLU, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import keras
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train[:1000]
# x_train = x_train.reshape(x_train.shape[0], 4, 7, 7, 4)
# x_test = x_test.reshape(x_test.shape[0], 4, 7, 7, 4)
# input_shape = (4, 7, 7, 4)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('train samples: ', x_train.shape)
print('test samples, ', x_test.shape)
# print('input_shape: ', input_shape)


cnn_model = Sequential()

# (7, 7, 4)
cnn_model.add(Conv2D(8, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(7, 7, 4)))
cnn_model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
cnn_model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
# cnn_model.add(Conv2D(4, kernel_size=(2, 2), activation='relu'))
# cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
# cnn_model.summary()


lstm_model = Sequential()
lstm_model.add(LSTM(512, input_shape=(4, 288), dropout=0.15, return_sequences=True))
lstm_model.add(BatchNormalization())
lstm_model.add(LSTM(512, dropout=0.15, return_sequences=False))
lstm_model.add(Dense(512))
lstm_model.add(BatchNormalization())
lstm_model.add(LeakyReLU(alpha=.001))
lstm_model.add(Dense(256))
lstm_model.add(BatchNormalization())
lstm_model.add(LeakyReLU(alpha=.001))
# lstm_model.summary()



upsample_model = Sequential()
upsample_model.add(Reshape((16, 16, 1), input_shape=(1, 256)))
upsample_model.add(Conv2DTranspose(16, kernel_size=(4, 4), activation='relu'))
upsample_model.add(BatchNormalization())
upsample_model.add(Conv2DTranspose(32, kernel_size=(4, 4), activation='relu'))
upsample_model.add(BatchNormalization())
upsample_model.add(Conv2DTranspose(16, kernel_size=(4, 4), activation='relu'))
upsample_model.add(BatchNormalization())
upsample_model.add(Conv2DTranspose(1, kernel_size=(4, 4), activation='relu'))
upsample_model.add(BatchNormalization())
upsample_model.add(Reshape((4, 7, 7, 4)))
upsample_model.add(Reshape((28, 28)))
# upsample_model.summary()




cnn_input = Input(shape=(28, 28))
cnn_input1 = Reshape(target_shape=(4, 7, 7, 4))(cnn_input)
lstm_input = TimeDistributed(cnn_model)(cnn_input1)
lstm_output = lstm_model(lstm_input)
final_output = upsample_model(lstm_output)

cnn_lstm_model = Model(inputs=cnn_input, outputs=final_output)

def weighted_binary_crossentropy(weights):
    def w_binary_crossentropy(y_true, y_pred):
        return tf.keras.backend.mean(tf.nn.weighted_cross_entropy_with_logits(
            y_true,
            y_pred,
            weights,
            name=None
        ), axis=-1)
    return w_binary_crossentropy

weighted_loss = weighted_binary_crossentropy(weights=4)


def recall(y_true, y_pred):
    y_true = math_ops.cast(y_true, 'float32')
    y_pred = math_ops.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

cus_callback = []
cus_callback.append(
    ModelCheckpoint(
        filepath=os.path.join("checkpoints","uav-{epoch:02d}-{val_recall:.2f}.hdf5"),
        monitor='val_recall',
        mode='auto',
        save_best_only=True,
        save_weights_only=True,
        verbose=True
    )
)

cnn_lstm_model.compile(optimizer='adadelta', loss=weighted_loss, metrics=[recall])

cnn_lstm_model.fit(x_train, x_train,
                    epochs=12, batch_size=32,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=cus_callback)

