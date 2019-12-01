import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

import os
import numpy as np
import random

from collections import deque

class DQNModel:
    def __init__(self, action_space, gamma=0.99, eps=1.0, eps_min=0.01, eps_decay=0.9998):
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
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))

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
