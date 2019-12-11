import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input
from keras.utils import to_categorical

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
EPOCHS = int(os.environ.get('EPOCHS', 50))
TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', 0.8))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 200))


def categorize(data):
    categorized = []
    for item in data:
        sub_arr = [0]*NUM_CLASSES
        '''if item > 0:
            sub_arr[100 + item] = 1
        else:
            sub_arr[100 - item] = 1'''
        sub_arr[100 + item] = 1
        categorized.append(np.array(sub_arr))
    return np.array(categorized)


def ffnn(x_data, y_data):
    
    x_train = np.array(x_data[: int(len(x_data) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_data[: int(len(y_data) * TRAINING_PERCENTAGE)])
    
    x_test = np.array(x_data[int(len(x_data) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_data[int(len(y_data) * TRAINING_PERCENTAGE):])

    y_train = categorize(y_train)
    y_test = categorize(y_test)
 
    model = Sequential([
        Dense(len(x_train[0]), input_dim=len(x_train[0])),
        LeakyReLU(alpha=0.3),
        Dropout(0.2),
        Dense(200),
        LeakyReLU(alpha=0.3),
        Dense(200),
        LeakyReLU(alpha=0.3),
        Dense(200),
        LeakyReLU(alpha=0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
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

    #Evaluating the model
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("Accuracy FFNN: %.2f%%" % (scores[1]*100))
