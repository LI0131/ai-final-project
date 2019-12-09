import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical

BATCH_SIZE = os.environ.get('BATCH_SIZE', 16)
EPOCHS = os.environ.get('EPOCHS', 3)
TRAINING_PERCENTAGE = os.environ.get('TRAINING_PERCENTAGE', 0.8)


def ffnn(x_train, y_train):

    x_train = np.array(x_train[: int(len(x_train) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_train[: int(len(y_train) * TRAINING_PERCENTAGE)])
    x_test = np.array(x_train[int(len(x_train) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_train[int(len(y_train) * TRAINING_PERCENTAGE):])

    print(x_train[0])
    print([type(x) for x in x_train[0]])
    print(y_train[0])

    y_train = to_categorical(y_train, num_classes=100)
    y_test = to_categorical(y_test, num_classes=100)
 
    model = Sequential([
        Dense(len(x_train[0]), activation='relu', input_dim=len(x_train[0])),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dropout(0.2),
        Dense(100, activation='softmax')
    ])

    model.compile(
        loss='mean_squared_logarithmic_error',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test)
    )
