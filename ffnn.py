import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.utils import to_categorical

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 128))
EPOCHS = int(os.environ.get('EPOCHS', 25))
TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', 0.8))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 200))


def ffnn(x_data, y_data):

    x_train = np.array(x_data[: int(len(x_data) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_data[: int(len(y_data) * TRAINING_PERCENTAGE)])
    x_test = np.array(x_data[int(len(x_data) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_data[int(len(y_data) * TRAINING_PERCENTAGE):])

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
 
    model = Sequential([
        Dense(len(x_train[0]), input_dim=len(x_train[0])),
        LeakyReLU(alpha=0.3),
        Dropout(0.2),
        Dense(20),
        LeakyReLU(alpha=0.3),
        Dropout(0.2),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test)
    )
