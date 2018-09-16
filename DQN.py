#import snakeClass
import pygame
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
import random
import numpy as np
import pandas as pd
import pygame
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
import random
import numpy as np
import pandas as pd
import keras
import keras.backend as K
import copy

pd.set_option('display.max_columns', 500)

class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.model = self.network()
        #self.model = self.network("weights.hdf5")
        self.epsilon = 3
        self.actual = []



    def get_state(self, game, player, food):
        state = [
                    player.position[-1][0] - 20 in player.position or player.position[-1][0] - 20 < 0,                    # danger left
                    player.position[-1][0] - 40 in player.position or player.position[-1][0] - 40 < 0,                    # danger 2 left
                    player.position[-1][0] + 20 in player.position or player.position[-1][0] + 20 > game.display_width,   # danger right
                    player.position[-1][0] + 40 in player.position or player.position[-1][0] + 40 > game.display_width,   # danger 2 right
                    player.position[-1][1] - 20 in player.position or player.position[-1][1] - 20 < 0,                    # danger up
                    player.position[-1][-1] - 40 in player.position or player.position[-1][-1] - 40 < 0,                  # danger 2 up
                    player.position[-1][1] + 20 in player.position or player.position[-1][1] + 20 > game.display_height,  # danger down
                    player.position[-1][-1] + 40 in player.position or player.position[-1][-1] + 40 > game.display_height,# danger 2 down
                    player.x_change == -20,           # move left
                    player.x_change == 20,            # move right
                    player.y_change == -20,           # move up
                    player.y_change == 20,            # move down
                    food.x_food < player.x,           # food left
                    food.x_food > player.x,           # food right
                    food.y_food < player.y,           # food up
                    food.y_food > player.y            # food down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
        return np.asarray(state)


    def set_reward(self, game, player, food):
        if game.crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        elif (player.x_change < 0 and food.x_food < player.x) or (player.x_change > 0 and food.x_food > player.x) or (player.y_change < 0 and food.y_food < player.y) or (player.y_change > 0 and food.y_food > player.y):
            self.reward = 2
        else:
            self.reward = -1
        return self.reward

    def possible_moves(self, player):
        if player.x_change == -20:
            return [0,2,3]
        elif player.x_change == 20:
            return [1,2,3]
        elif player.y_change == -20:
            return [0,1,2]
        elif player.y_change == 20:
            return [0,1,3]

    def replay(self, game, player, food, actual):
        player.position = copy.deepcopy(actual[0])
        player.x, player.y, player.x_change, player.y_change, food.x_food, food.y_food, game.crash, player.eaten, player.food = actual[1:]

    '''
    def next_state(self, game, player, food, i):
        actual = [player.position, player.x, player.y, player.x_change, player.y_change, food.x_food, food.y_food, game.crash, player.eaten]
        original_state = self.get_state(game, player, food)
        player.do_move(i, player.x, player.y, game, food)
        player.display_player(player.x, player.y,player.food,game,player)
        array = [original_state, i, self.set_reward(game, player), self.get_state(game, player, food)]
        pygame.time.wait(500)
        self.replay(game, player, food, actual)
        player.display_player(player.x, player.y, player.food, game, player)
        return array
    '''

    def next_state(self, game, player, food, i):
        actual = [player.position, player.x, player.y, player.x_change, player.y_change, food.x_food, food.y_food, game.crash, player.eaten, player.food]
        original_state = self.get_state(game, player, food)
        player.do_move(i, player.x, player.y, game, food)
        player.display_player(player.x, player.y,player.food,game,player)
        array = [original_state, i, self.set_reward(game, player), self.get_state(game, player, food)]
        pygame.time.wait(500)
        self.replay(game, player, food, actual)
        player.display_player(player.x, player.y, player.food, game, player)
        return array

    def loss(self, target, state, action):
        return K.mean(K.square(target - self.predict_q(self.model, state, action)), axis=-1)

    def network(self,weights=None):
        model = Sequential()
        model.add(Dense(output_dim=50, activation='relu', input_dim=17))
        model.add(Dense(output_dim=50, activation='relu'))
        model.add(Dense(output_dim=1, activation='softmax'))
        opt = RMSprop(lr=0.025)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        # [self.loss(self.agent_target, self.agent_predict)]
        return model

    def act(self, state):
        if random.random(0, 1) < self.epsilon:
            return random.randint(0, 4)
        else:
            return np.argmax(self.brain.predictOne(state))

    def observe(self, sequence):  # in (s, a, r, s_) format
        self.memory.add(sequence)

    def q_parameter(self):
        q = self.reward + self.gamma * np.argmax(self.fit_q())

    def predict_q(self, model, state, action):
        predictor = np.array([np.hstack(np.array([state, action]))])
        q = model.predict(predictor)
        return q

    def train_q(self, storage, state, action):
        train = np.array([storage[:17]])
        test = np.array([storage[17]])
        self.model.compile(loss='mse', optimizer=RMSprop(lr=0.025))
        self.model.fit(train, test, epochs=1)

    def train2_q(self,training, test):
        training = training.values
        test = test.values
        self.model.fit(training, test, epochs=1)


    def initialize_dataframe(self):
        state = [0]*16
        for i in range(16):
            state[i]= random.choice([0, 1])
        move = random.randint(1,4)
        reward = random.choice([-1, -10, 10])
        future_state = [0]*16
        for i in range(16):
            future_state[i] = random.choice([True, False])
        Q = 1
        array = [state, move, reward, future_state, Q]
        self.dataframe = self.dataframe.append([array])

    def store_memory(self, state, action, q):
        self.short_memory = np.hstack(np.array([state, action, q]))
        #print(self.short_memory)





'''
def run():
    pygame.init()
    while True:
        game = Game(400, 400)
        player1 = game.player
        food1 = game.food
        display(player1, food1, game)
        initial_move(player1,game,food1)
        display(player1, food1, game)
        pygame.time.wait(game.speed)
        while not game.crash:
            loop(player1, food1, game)
            display(player1, food1, game)
            pygame.time.wait(game.speed)

run()
'''




