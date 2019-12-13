import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, LSTM

from eval import crps
from export import export
from graphing import graph

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
EPOCHS = int(os.environ.get('RNN_EPOCHS', 10))
TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', 0.8))


def rnn(x_data, y_data):

    x_train = np.array(x_data[: int(len(x_data) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_data[: int(len(y_data) * TRAINING_PERCENTAGE)])
    
    x_test = np.array(x_data[int(len(x_data) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_data[int(len(y_data) * TRAINING_PERCENTAGE):])

    maxFeatures = 15500
    
    model = Sequential()
    model.add(LSTM(200))
    model.add(Dense(1, activation=LeakyReLU(alpha=0.3)))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=[crps]
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test)
    )

    graph(history, to_file='images/rnn.png')

    scores = model.evaluate(x_test, y_test)
    print(f'CRPS RNN: {scores[1]}')

    df = export(x_train, x_test, model)

    print(df)
