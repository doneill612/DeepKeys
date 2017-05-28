import utils
import network

import os
# Suppress some Keras warnings ...
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

songs = utils.get_songs('samples')
notes = utils.get_all_notes(songs)

max_t = utils.scale_note_data(notes)

X = []
y = []

len_prev = 30

for i in range(len(notes) - len_prev):
    X.append(notes[i:i + len_prev])
    y.append(notes[i + len_prev])

seed = notes[0:len_prev]

model = network.build_trained_network(X, y)

prediction = network.predict(model, seed, max_t)

utils.save_new_song(prediction, 0)