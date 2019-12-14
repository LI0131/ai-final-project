import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input

from export import export
from eval import crps
from graphing import graph

TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', 0.8))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
EPOCHS = int(os.environ.get('AE_EPOCHS', 100))


def flatten(data):
    final_play_data = []
    for array in data:
        final_play_data.append(array.flatten())
    final_play_data = np.asarray(final_play_data)
    return final_play_data


def ae(x_data, y_data):
        
    x_train = np.array(x_data[: int(len(x_data) * TRAINING_PERCENTAGE)])
    y_train = np.array(y_data[: int(len(y_data) * TRAINING_PERCENTAGE)])
    
    x_test = np.array(x_data[int(len(x_data) * TRAINING_PERCENTAGE):])
    y_test = np.array(y_data[int(len(y_data) * TRAINING_PERCENTAGE):])

    x_train = flatten(x_train)
    x_test = flatten(x_test)

    autoencoder = Sequential()

    # Encoder Layers
    autoencoder.add(Dense(len(x_train[0]), input_dim=len(x_train[0]), activation=LeakyReLU(alpha=0.3)))
    autoencoder.add(Dense(23, activation=LeakyReLU(alpha=0.3)))
    autoencoder.add(Dense(11, activation=LeakyReLU(alpha=0.3)))

    # Decoder Layers
    autoencoder.add(Dense(11, activation=LeakyReLU(alpha=0.3)))
    autoencoder.add(Dense(23, activation=LeakyReLU(alpha=0.3)))
    autoencoder.add(Dense(len(x_train[0]), activation=LeakyReLU(alpha=0.3)))

    autoencoder.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[crps]
    )

    history = autoencoder.fit(
        x_train, x_train,
        epochs=100,
        batch_size=128,
        validation_data=(x_test, x_test),
        verbose=2
    )

    graph(history, to_file='images/ae.png')

    inputs = Input(shape=(len(x_train[0]),))
    encoder_layer1 = autoencoder.layers[0](inputs)
    encoder_layer2 = autoencoder.layers[1](encoder_layer1)
    encoder_layer3 = autoencoder.layers[2](encoder_layer2)
    encoder = Model(inputs, encoder_layer3)

    encoded_x_train = encoder.predict(x_train, verbose = 2)
    print(len(encoded_x_train[0]))
    print(encoded_x_train.shape)
    encoded_x_test = encoder.predict(x_test, verbose = 2)
    
    model = Sequential([
        Dense(45, input_dim=len(encoded_x_train[0])),
        LeakyReLU(alpha=0.3),
        Dense(22),
        LeakyReLU(alpha=0.3),
        Dense(1),
        LeakyReLU(alpha=0.3)
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=[crps]
    )

    history = model.fit(
        encoded_x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(encoded_x_test, y_test)
    )

    # graph(history, to_file='images/ae-ffnn.png')

    #Evaluating the model
    scores = model.evaluate(encoded_x_test, y_test)
    print(f'CRPS AE: {scores[1]}')
