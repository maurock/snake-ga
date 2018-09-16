import pygame
from random import randint
from DQN import DQNAgent
import numpy as np
import copy
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import linear_model

class Game:

    def __init__(self, display_width, display_height):
        pygame.display.set_caption('SnakeGen')
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = pygame.display.set_mode((display_width, display_height))
        self.crash = False
        self.player = Player(self)
        self.food = Food(self, self.player)
        self.speed = 50
        self.score = 0


class Player(object):

    def __init__(self, game):
        x = 0.45 * game.display_width
        y = 0.5 * game.display_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('snakeBody.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y,agent):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    # categorical_array = [straight, right, left]
    def do_move(self, move, x, y, game, food,agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:

            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move ,[1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move,[0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move,[0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        #print(self.x_change, self.y_change, self.x, self.y, self.position)
        if self.x < 0 or self.x == game.display_width or self.y < 0 or self.y == game.display_height or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y,agent)




    def display_player(self, x, y, food, game, player):
        self.position[-1][0] = x
        self.position[-1][1] = y

        for i in range(food):
            x_temp, y_temp = self.position[len(self.position) - 1 - i]
            game.gameDisplay.blit(player.image, (x_temp, y_temp))
        update_screen()


class Food(object):

    def __init__(self, game, player):
        #self.x_food, self.y_food = self.food_coord(game, player)
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('food.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.display_width - 20)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.display_height - 20)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game,player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def display(player, food, game):
    game.gameDisplay.fill((255, 255, 255))
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game, player)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()

def initial_move(player, game, food,agent):
    player.do_move(1, player.x, player.y, game, food,agent)

# def loop(player, food, game,agent):
#     move = 0
#     if food.x_food < player.x and [(player.x - 20), player.y] not in player.position and (player.x - 20) > 0:
#         move = 0
#
#     elif food.x_food > player.x and [(player.x + 20), player.y] not in player.position and (
#             player.x + 20) < game.display_width:
#         move=1
#
#     elif food.y_food < player.y and [player.x, (player.y - 20)] not in player.position and (player.y - 20) > 0:
#         move =2
#
#     elif food.y_food > player.y and [player.x, (player.y + 20)] not in player.position and (
#             player.y + 20) < game.display_height:
#         move = 3
#
#     elif [(player.x - 20), player.y] not in player.position and player.x_change == 0 and (player.x - 20) > 0:
#         move = 0
#
#     elif [(player.x + 20), player.y] not in player.position and player.x_change == 0 and (
#             player.x + 20) < game.display_width:
#         move = 1
#
#     elif [player.x, (player.y - 20)] not in player.position and player.y_change == 0 and (player.y - 20) > 0:
#         move = 2
#
#     elif [player.x, (player.y + 20)] not in player.position and player.y_change == 0 and (
#             player.y + 20) < game.display_height:
#         move = 3
#
#     else:
#         move = randint(1, 4)
#
#     player.do_move(move, player.x, player.y, game, food,agent)


def run():
    pygame.init()
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot =[]
    while counter_games < 100:
        #Initialize game
        game = Game(400, 400)
        player1 = game.player
        food1 = game.food
        #agent.reward = 0
        #Initialize storage to train first network
        state_init1 = agent.get_state(game,player1,food1)    #[0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
        action = [1, 0, 0]
        player1.do_move(action, player1.x, player1.y, game, food1, agent)
        state_init2 = agent.get_state(game, player1, food1)
        reward1 = agent.set_reward(game,player1,food1,game.crash)
        agent.remember(state_init1,action, reward1, state_init2, game.crash)
        agent.replay_new(agent.memory)
        #Performn first move
        #display(player1, food1, game)
        while not game.crash:
            if counter_games < 15:
                agent.epsilon = 3
            elif counter_games < 30:
                agent.epsilon = 2
            elif counter_games >= 30:
                agent.epsilon = 0
            state_old = agent.get_state(game, player1, food1)
            if randint(0, 10) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)[0]
            else:
                prediction = agent.model.predict(state_old.reshape((1,16)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)[0]
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            state_new = agent.get_state(game, player1, food1)
            reward = agent.set_reward(game, player1, food1, game.crash)
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            #display(player1, food1, game)
            #pygame.time.wait(game.speed)


        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    agent.model.save_weights('weights_new2.hdf5')

    fit = np.polyfit(counter_plot, score_plot, 1)
    fit_fn = np.poly1d(fit)
    plt.plot(counter_plot, score_plot, 'yo', counter_plot, fit_fn(counter_plot), '--k')
    plt.show()

run()