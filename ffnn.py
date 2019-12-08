import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

BATCH_SIZE = os.environ.get('BATCH_SIZE', 32)
EPOCHS = os.environ.get('EPOCHS', 3)
TRAINING_PERCENTAGE = os.environ.get('TRAINING_PERCENTAGE', 0.8)


def ffnn(x_train, y_train):

    x_train = np.array(x_train[: int(len(x_train) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_train[: int(len(y_train) * TRAINING_PERCENTAGE)])
    x_test = np.array(x_train[int(len(x_train) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_train[int(len(y_train) * TRAINING_PERCENTAGE):])

    model = Sequential([
        Dense(512, activation='sigmoid', input_dim=len(x_train[0])),
        Dropout(0.2),
        Dense(512, activation='sigmoid'),
        Dropout(0.2),
        Dense(1, activation='softmax')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test)
    )
