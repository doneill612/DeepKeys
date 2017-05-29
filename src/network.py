from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import LSTM, Dense, Activation, Dropout
import numpy as np



"""
Builds, trains, and saves a sequential LSTM network.
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
def build_trained_network(X, y, n_epochs=25):
    print("Building network...")

    model = Sequential()
    model.add(LSTM(128, input_shape=(30, 3), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, input_shape=(30, 3), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('linear'))

    optimizer = RMSprop(lr=0.005)
    model.compile(loss='mse', optimizer=optimizer)
    print("Network built.")
    print("Training network...")
    model.fit(X, y, batch_size=128, epochs=n_epochs, verbose=1)
    print("Network trained. Saving...")
    model.save('savedmodels/model_{}_epochs_111.h5'.format(n_epochs))
    print("Saved.")
    return model

"""
Loads a trained model with the name model_name

model_name: the name of the h5 model (not including path - path is constructed)
@:return the saved model
"""
def get_trained_network(model_name):
    print("Loading model...")
    path = "savedmodels/{}.h5".format(model_name)
    model = load_model(path)
    print("Model loaded!")
    return model

"""
Uses the network model to create a seed prediction.
The notes in the seed are rescaled to musical values.

model: the neural network model
seed: the seed used for prediction
max_t: the longest note time span (used for scaling)
@:return the seed prediction
"""
def predict(model, seed, max_t, n_epochs = 25):

    print("Generating...")
    prediction = list()
    _seed = seed
    _seed = np.expand_dims(_seed, axis=0)
    for i in range(int(n_epochs * 7.5) if int(n_epochs * 7.5) < 5000 else 5000):
        predictions = model.predict(_seed)
        print(predictions)
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

    print("Done!")
    return prediction
