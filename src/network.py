from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import LSTM, Dense, Activation, Dropout

import numpy as np



"""
Builds and trains a sequential LSTM network.
Each LSTM layer is followed by a dropout - 0.2 between the two LSTMs, 0.5 between the second LSTM and Dense layer
Layers use softmax activation and the loss is a categorical cross-entropy function.

Network hyperparameters:
    LSTM layer neurons  : 128
    Dense layer neurons : 3

Training hyperparameters:
    Batch size      : 300
    Training epochs : 500

X: training data
y: labels

@:return the trained neural network

"""
def build_trained_network(X, y):
    print("Building network...")

    model = Sequential()
    model.add(LSTM(128, input_shape=(30, 3), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, input_shape=(30, 3), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print("Network built.")
    print("Training network...")
    model.fit(X, y, batch_size=300, epochs=500, verbose=1)
    print("Training complete.")
    return model

"""
Uses the network model to create a seed prediction.
The notes in the seed are rescaled to musical values.

model: the neural network model
seed: the seed used for prediction
max_t: the longest note time span (used for scaling)
@:return the seed prediction
"""
def predict(model, seed, max_t):
    prediction = list()
    _seed = np.expand_dims(seed, axis=0)
    for i in range(3000):
        predictions = model.predict(_seed)
        _seed = np.squeeze(_seed)
        _seed = np.concatenate((_seed, predictions))
        _seed = _seed[1:]
        _seed = np.expand_dims(_seed, axis=0)
        predictions = np.squeeze(predictions)
        prediction.append(predictions)
    for note in prediction:
        note[0] = int(88 * note[0] + 24)
        note[1] = int(127 * note[1])
        note[2] *= max_t
        if note[0] < 24:
            note[0] = 24
        elif note[0] > 102:
            note[0] = 102
        if note[1] < 0:
            note[1] = 0
        elif note[1] > 127:
            note[1] = 127
        if note[2] < 0:
            note[2] = 0

    return prediction