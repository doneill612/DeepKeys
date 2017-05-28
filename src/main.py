import utils
import network

import os
# Suppress some Keras warnings ...
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NOTE_MEMORY = 30

songs = utils.get_songs('samples')
notes = utils.get_all_notes(songs)

max_t = utils.scale_note_data(notes)

X = []
y = []

for i in range(len(notes) - NOTE_MEMORY):
    X.append(notes[i:i + NOTE_MEMORY])
    y.append(notes[i + NOTE_MEMORY])

seed = notes[0:NOTE_MEMORY]

model = network.build_trained_network(X, y, n_epochs=400)

prediction = network.predict(model, seed, max_t, n_epochs=400)

utils.save_new_song(prediction, 1)