import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input, BatchNormalization

from eval import crps
from export import export
from graphing import graph

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
EPOCHS = int(os.environ.get('FFNN_EPOCHS', 100))
TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', 0.8))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 200))


def categorize(data):
    categorized = []
    for item in data:
        sub_arr = [0]*NUM_CLASSES
        sub_arr[100 + item] = 1
        categorized.append(np.array(sub_arr))
    return np.array(categorized)


def flatten(data):
    final_play_data = []
    for array in data:
        final_play_data.append(array.flatten())
    final_play_data = np.asarray(final_play_data)
    return final_play_data


def ffnn(x_data, y_data):

    x_data = flatten(x_data)
    numnodes = len(x_data[0])
    
    x_train = np.array(x_data[: int(len(x_data) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_data[: int(len(y_data) * TRAINING_PERCENTAGE)])
    
    x_test = np.array(x_data[int(len(x_data) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_data[int(len(y_data) * TRAINING_PERCENTAGE):])
 
    model = Sequential([
        Dense(numnodes, input_dim=len(x_train[0])),
        LeakyReLU(alpha=0.3),
        BatchNormalization(),
        Dropout(0.3),

        Dense(numnodes*2),
        LeakyReLU(alpha=0.3),
        BatchNormalization(),
        Dropout(0.3),

        Dense(numnodes*2),
        LeakyReLU(alpha=0.3),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1),
        LeakyReLU(alpha=0.3)
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=[crps]
    )

    model.summary()

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=2
    )

    # graph(history, to_file='images/ffnn.png')

    #Evaluating the model
    scores = model.evaluate(x_test, y_test, verbose=2)
    print(f'CRPS FFNN: {scores[1]}')

    df = export(x_train, x_test, model)

    print(df)
