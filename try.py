import pygame
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
import random
import numpy as np
import pandas as pd
from DQN import DQNAgent
from snakeClass import *
import copy

class Agent(object):
    def __init__(self):
        self.actual = []

class Play(object):

    def __init__(self):
        self.x = 0.45 * 400
        self.y = 0.5 * 400
        self.position = []
        self.position.append([self.x, self.y])
        self.x_change = 20
        self.y_change = 0

    def do_move(self, move, x, y):
        move_array = [self.x_change, self.y_change]

        if move == 0 and self.x_change == 0:  # left
            move_array = [-20, 0]
        elif move == 1 and self.x_change == 0:  # right
            move_array = [20, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        self.update_position(self.x, self.y)

    def update_position(self, x, y):
        self.position[-1][0] = x
        self.position[-1][1] = y


def run():
    agent = Agent()
    player1 = Play()
    agent.actual = copy.deepcopy([player1.position, player1.x])
    print(agent.actual[0])
    i = 1
    player1.do_move(i, player1.x, player1.y)
    print(agent.actual[0])

run()
