import utils
import network
import os
# Suppress some Keras warnings ...
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from random import randint


NOTE_MEMORY = 30

def __pre_process():
    songs = utils.get_songs('samples')
    notes = utils.get_all_notes(songs)
    max_t = utils.scale_note_data(notes)
    X = []
    y = []

    rand_start = randint(0, len(notes) - NOTE_MEMORY)

    for i in range(len(notes) - NOTE_MEMORY):
        X.append(notes[i:i + NOTE_MEMORY])
        y.append(notes[i + NOTE_MEMORY])

    seed = notes[rand_start: rand_start + NOTE_MEMORY]

    return songs, notes, max_t, X, y, seed

def generate_from_trained(output_fn, model_loc='savedmodels/model_3600_epochs_111.h5', epochs=int(2000/7.5)):

    songs, notes, max_t, _, __, seed = __pre_process()

    model = network.get_trained_network(model_loc)

    prediction = network.predict(model, seed, max_t, n_epochs=epochs)

    utils.save_new_song(prediction, output_fn)

def ground_up(epochs, output_fn):

    songs, notes, max_t, X, y, seed = __pre_process()

    model = network.build_trained_network(X, y, n_epochs=epochs)

    prediction = network.predict(model, seed, max_t, n_epochs=epochs)

    utils.save_new_song(prediction, output_fn)
