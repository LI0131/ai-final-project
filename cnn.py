import os
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras.layers import Flatten, Dense, BatchNormalization, Dropout

NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 200))
TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', 0.8))


def categorize(data):
    categorized = []
    for item in data:
        sub_arr = [0]*NUM_CLASSES
        sub_arr[100 + item] = 1
        categorized.append(np.array(sub_arr))
    return np.array(categorized)


def cnn(x_data, y_data):

    x_train = np.array(x_data[: int(len(x_data) * TRAINING_PERCENTAGE)])
    x_train = x_train.reshape(len(x_train), 22, 38 ,1)
    y_train = categorize(np.array(y_data[: int(len(y_data) * TRAINING_PERCENTAGE)]))
    
    x_test = np.array(x_data[int(len(x_data) * TRAINING_PERCENTAGE):])
    x_test = x_test.reshape(len(x_test), 22, 38 ,1)
    y_test = categorize(np.array(y_data[int(len(y_data) * TRAINING_PERCENTAGE):]))

    weight_decay = 1e-4

    model = Sequential([
        Conv2D(32, (3, 3),
            activation=LeakyReLU(alpha=0.25),
            input_shape=(len(x_data[0]), len(x_data[0][0]), 1),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        BatchNormalization(),
        Conv2D(32, (3, 3),
            activation=LeakyReLU(alpha=0.25),
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        Dropout(0.3),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3),
            activation=LeakyReLU(alpha=0.25),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        BatchNormalization(),
        Conv2D(64, (3, 3),
            activation=LeakyReLU(alpha=0.25),
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        Dropout(0.3),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation=LeakyReLU(alpha=0.25)),
        Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=100, 
        validation_data=(x_test, y_test)
    )
