import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import os
import numpy as np
import random

TRAIN_DIR = "training2"

# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56243777
# save np load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class Model:
    def __init__(self):
        self.model = Sequential()

        # hidden conv net layers
        num_layers = [32, 64, 128]
        specify_shape = True
        for num_layer in num_layers:
            if specify_shape:
                self.model.add(Conv2D(num_layer, 
                                      (3, 3), 
                                      padding='same', 
                                      input_shape=(184, 152, 3), 
                                      activation='relu'))
                specify_shape = False
            else:
                self.model.add(
                    Conv2D(num_layer, (3, 3), padding='same', activation='relu'))

            self.model.add(Conv2D(num_layer, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.2))

        # fully connected dense layer
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))

        # output layer
        self.model.add(Dense(4, activation='softmax'))

        # compilation settings
        self.alpha = 1e-4
        opt = keras.optimizers.adam(lr=self.alpha, decay=1e-6)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        # log everything via tensorboard
        self.tensorboard = TensorBoard(log_dir="logs/v0.1")

    def fit(self, epochs):
        index = 0

        files = os.listdir(TRAIN_DIR)
        size = len(files)

        step_size = 5
        test_size = 100
        batch_size = 128

        for _ in range(epochs):
            random.shuffle(files)

            while index < size:
                attack_none = []
                attack_enemy_structures = []
                attack_enemy_units = []
                attack_enemy_start = []

                lengths = [0, 0, 0, 0]

                for file in files[index:index+step_size]:
                    path = os.path.join(TRAIN_DIR, file)
                    data = list(np.load(path))
                    for d in data:
                        choice = np.argmax(d[0])
                        lengths[choice] += 1
                        if choice == 0:
                            attack_none.append(d)
                        elif choice == 1:
                            attack_enemy_structures.append(d)
                        elif choice == 2:
                            attack_enemy_units.append(d)
                        elif choice == 3:
                            attack_enemy_start.append(d)

                random.shuffle(attack_none)
                random.shuffle(attack_enemy_structures)
                random.shuffle(attack_enemy_units)
                random.shuffle(attack_enemy_start)

                min_choice = min(lengths)

                attack_none = attack_none[:min_choice]
                attack_enemy_structures = attack_enemy_structures[:min_choice]
                attack_enemy_units = attack_enemy_units[:min_choice]
                attack_enemy_start = attack_enemy_start[:min_choice]

                training = attack_none + attack_enemy_structures \
                + attack_enemy_units + attack_enemy_start

                random.shuffle(training)

                training_x = np.array([i[1] for i in training[:-test_size]]).reshape(-1, 184, 152, 3)
                training_y = np.array([i[0] for i in training[:-test_size]])

                testing_x = np.array([i[1] for i in training[-test_size:]]).reshape(-1, 184, 152, 3)
                testing_y = np.array([i[0] for i in training[-test_size:]])

                self.model.fit(training_x, 
                               training_y,
                               batch_size=batch_size,
                               validation_data=(testing_x, testing_y),
                               shuffle=True,
                               verbose=1, 
                               callbacks=[self.tensorboard])
                self.model.save(f'CNN-{epochs}-epoch-{self.alpha}-alpha')
                index += step_size

model = Model()
model.fit(10)