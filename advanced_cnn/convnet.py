import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

import os
import numpy as np
import random

from collections import deque

class DQNModel:
    def __init__(self, action_space, gamma=0.98, eps=1.0, eps_min=0.01, eps_decay=0.995):
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        self.actions = action_space
        self.num_actions = len(action_space)

        self.build_neural_network_model()
        
    def build_neural_network_model(self):
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
        self.model.add(Dense(self.num_actions, activation='softmax'))

        # compilation settings
        self.alpha = 1e-4
        opt = keras.optimizers.adam(lr=self.alpha, decay=1e-6)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        self.model.summary()

        # log everything via tensorboard
        self.tensorboard = TensorBoard(log_dir="log")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) <= batch_size:
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    # def fit(self, epochs):
    #     index = 0

    #     files = os.listdir(TRAIN_DIR)
    #     size = len(files)

    #     step_size = 5
    #     test_size = 100
    #     batch_size = 128

    #     for _ in range(epochs):
    #         random.shuffle(files)

    #         while index < size:
    #             attack_none = []
    #             attack_enemy_structures = []
    #             attack_enemy_units = []
    #             attack_enemy_start = []

    #             lengths = [0 for x in range(25)]
    #             actions = {}
    #             for x in range(25):
    #                 actions[x] = []

    #             for file in files[index:index+step_size]:
    #                 path = os.path.join(TRAIN_DIR, file)
    #                 data = list(np.load(path))
    #                 for d in data:
    #                     choice = np.argmax(d[0])
    #                     lengths[choice] += 1
    #                     actions[choice].append(d)

    #             min_choice = min(lengths)
    #             training = []

    #             for _, data in actions.items():
    #                 random.shuffle(data)
    #                 training += data[:min_choice]

    #             random.shuffle(training)

    #             training_x = np.array([i[1] for i in training[:-test_size]]).reshape(-1, 184, 152, 3)
    #             training_y = np.array([i[0] for i in training[:-test_size]])

    #             testing_x = np.array([i[1] for i in training[-test_size:]]).reshape(-1, 184, 152, 3)
    #             testing_y = np.array([i[0] for i in training[-test_size:]])

    #             self.model.fit(training_x, 
    #                            training_y,
    #                            batch_size=batch_size,
    #                            validation_data=(testing_x, testing_y),
    #                            shuffle=True,
    #                            epochs=1,
    #                            verbose=1, 
    #                            callbacks=[self.tensorboard])
    #             self.model.save(f'CNN-{epochs}-epoch-{self.alpha}-alpha')
    #             index += step_size

# model = DQNModel(25)
# model.fit(50)