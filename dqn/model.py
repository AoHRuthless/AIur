import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

import os
import numpy as np
import random

from collections import deque

LOAD = True

class DQNModel:
    def __init__(self, action_space, gamma=0.99, eps=1.0, eps_min=0.01, eps_decay=0.9998):
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        self.actions = action_space
        self.num_actions = len(action_space)

        self.model = self.build_neural_network_model()
        self.target_model = self.build_neural_network_model(print_summary=False)

        if LOAD:
            self.load("training/terran-dqn.h5")
        
    def build_neural_network_model(self, print_summary=True):
        model = Sequential()

        # hidden conv net layers
        num_layers = [32, 64, 128]
        specify_shape = True
        for num_layer in num_layers:
            if specify_shape:
                model.add(Conv2D(num_layer, 
                                 (3, 3), 
                                 padding='same', 
                                 input_shape=(184, 152, 3), 
                                 activation='relu'))
                specify_shape = False
            else:
                model.add(
                    Conv2D(num_layer, (3, 3), padding='same', activation='relu'))

            model.add(Conv2D(num_layer, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

        # fully connected dense layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        # output layer
        model.add(Dense(self.num_actions, activation='softmax'))

        # compilation settings
        self.alpha = 1e-4
        opt = keras.optimizers.adam(lr=self.alpha, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        if print_summary:
            model.summary()

        # log everything via tensorboard
        self.tensorboard = TensorBoard(log_dir="log")
        return model

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
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_next = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + q_next * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
