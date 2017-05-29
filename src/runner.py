import utils
import network

from random import randint

NOTE_MEMORY = 30

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

model = network.load_model('savedmodels/model_3600_epochs_111.h5')

prediction = network.predict(model, seed, max_t, n_epochs=int(2000/7.5))

file_name = 'quick_2'

utils.save_new_song(prediction, file_name)