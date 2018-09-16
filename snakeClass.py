import pygame
from random import randint
from DQN import DQNAgent
import numpy as np
import copy


class Game:

    def __init__(self, display_width, display_height):
        pygame.display.set_caption('SnakeGen')
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = pygame.display.set_mode((display_width, display_height))
        self.crash = False
        self.player = Player(self)
        self.food = Food(self, self.player)
        self.speed = 1


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

    def do_move(self, move, x, y, game, food,agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if move == 0 and self.x_change == 0:  # left
            move_array = [-20, 0]
        elif move == 1 and self.x_change == 0:  # right
            move_array = [20, 0]
        elif move == 2 and self.y_change == 0:  # top
            move_array = [0, -20]
        elif move == 3 and self.y_change == 0:  # bottom
            move_array = [0, 20]
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
while counter_games < 10:
    #Initialize game
    game = Game(400, 400)
    player1 = game.player
    food1 = game.food
    #agent.reward = 0
    #Initialize storage to train first network
    state1 = agent.get_state(game,player1,food1)
    action = 1
    agent.store_memory(state1, action, 1)
    agent.dataframe = agent.dataframe.append([np.hstack(np.array([state1, action, 1]))])
    # agent.dataframe = agent.dataframe.append([np.hstack(np.array([state1, 2, 3]))])
    # agent.dataframe = agent.dataframe.append([np.hstack(np.array([state1, 3, 5]))])
    agent.train2_q(agent.dataframe[agent.dataframe.columns[:17]], agent.dataframe[agent.dataframe.columns[17]])
    #Performn first move
    display(player1, food1, game)
    initial_move(player1, game, food1,agent)
    display(player1, food1, game)
    while not game.crash:
        state2 = agent.get_state(game,player1,food1)
        agent.actual = copy.deepcopy([player1.position, player1.x, player1.y, player1.x_change, player1.y_change, food1.x_food,
                   food1.y_food,game.crash, player1.eaten, player1.food])
        primary_q_array = []
        if randint(0,10)<agent.epsilon:
            final_move=randint(0,3)
        else:
            for i in agent.possible_moves(player1):
                player1.eaten = False
                player1.do_move(i, player1.x, player1.y, game, food1,agent)
                if game.crash==True:
                    primary_q_array.append(agent.set_reward(game,player1, food1))
                else:
                    primary_q = agent.predict_q(agent.model, state2, i)
                    primary_q_array.append(primary_q)
                agent.replay(game,player1,food1,agent.actual)
            predicted_q =max(primary_q_array)
            max_index = primary_q_array.index(predicted_q)
            final_move = agent.possible_moves(player1)[max_index]
        player1.do_move(final_move, player1.x, player1.y, game, food1,agent)
        temp_reward = agent.reward + agent.set_reward(game, player1, food1)
        print('REWARD: ', temp_reward)
        temp_state = agent.get_state(game, player1, food1)
        agent.actual = copy.deepcopy([player1.position, player1.x, player1.y, player1.x_change, player1.y_change, food1.x_food,
                   food1.y_food,game.crash, player1.eaten, player1.food])
        secondary_q_array = []

        if not game.crash:
            for j in agent.possible_moves(player1):
                player1.eaten = False
                player1.do_move(j, player1.x, player1.y, game, food1, agent)

                if game.crash == True:
                    secondary_q_array.append(agent.reward)
                    print('CRASH', secondary_q_array)

                else:

                    secondary_q = agent.predict_q(agent.model, temp_state, j)
                    secondary_q_array.append(secondary_q)
                agent.replay(game, player1, food1, agent.actual)
            max_target_q = max(secondary_q_array)
            target_q = [temp_reward + agent.gamma * max_target_q]
            print('1', target_q)

        else:
            target_q = temp_reward
            print('2', target_q)
        #agent.agent_target = target_q
        #agent.agent_predict = predicted_q
        #agent.store_memory(state2, final_move, target_q[0])
        print('3', state2)
        print('3', final_move)
        print('3', target_q[0])
        print('3', [np.hstack(np.array([state2, final_move, target_q[0][0]]))])
        agent.dataframe = agent.dataframe.append([np.hstack(np.array([state2, final_move, target_q[0][0]]))])
        agent.train2_q(agent.dataframe[agent.dataframe.columns[:17]], agent.dataframe[agent.dataframe.columns[17]])
        display(player1, food1, game)
        pygame.time.wait(game.speed)
    print('GAME: ',counter_games)
    counter_games = counter_games + 1

agent.model.save_weights('weights.hdf5')



# pygame.init()
# agent = DQNAgent()
# counter_games = 0
# while counter_games < 10:
#     #Initialize game
#     game = Game(400, 400)
#     player1 = game.player
#     food1 = game.food
#     agent.reward = 0
#     #Initialize storage to train first network
#     state1 = agent.get_state(game,player1,food1)
#     action = 1
#     agent.store_memory(state1, action, 1)
#     agent.dataframe = agent.dataframe.append([np.hstack(np.array([state1, action, 1]))])
#     # agent.dataframe = agent.dataframe.append([np.hstack(np.array([state1, 2, 3]))])
#     # agent.dataframe = agent.dataframe.append([np.hstack(np.array([state1, 3, 5]))])
#     agent.train2_q(agent.dataframe[agent.dataframe.columns[:17]], agent.dataframe[agent.dataframe.columns[17]])
#     #Performn first move
#     display(player1, food1, game)
#     initial_move(player1, game, food1,agent)
#     display(player1, food1, game)
#     while not game.crash:
#         state2 = agent.get_state(game,player1,food1)
#         agent.actual = copy.deepcopy([player1.position, player1.x, player1.y, player1.x_change, player1.y_change, food1.x_food,
#                    food1.y_food,game.crash, player1.eaten, player1.food])
#         primary_q_array = []
#         if randint(0,10)<agent.epsilon:
#             final_move=randint(0,3)
#         else:
#             for i in agent.possible_moves(player1):
#                 player1.eaten = False
#                 player1.do_move(i, player1.x, player1.y, game, food1,agent)
#                 if game.crash==True:
#                     primary_q_array.append(agent.set_reward(game,player1, food1))
#                 else:
#                     primary_q = agent.predict_q(agent.model, state2, i)
#                     primary_q_array.append(primary_q)
#                 agent.replay(game,player1,food1,agent.actual)
#             predicted_q =max(primary_q_array)
#             max_index = primary_q_array.index(predicted_q)
#             final_move = agent.possible_moves(player1)[max_index]
#         player1.do_move(final_move, player1.x, player1.y, game, food1,agent)
#         temp_reward = agent.reward + agent.set_reward(game, player1, food1)
#         print('REWARD: ', temp_reward)
#         temp_state = agent.get_state(game, player1, food1)
#         agent.actual = copy.deepcopy([player1.position, player1.x, player1.y, player1.x_change, player1.y_change, food1.x_food,
#                    food1.y_food,game.crash, player1.eaten, player1.food])
#         secondary_q_array = []
#
#         if not game.crash:
#             for j in agent.possible_moves(player1):
#                 player1.eaten = False
#                 player1.do_move(j, player1.x, player1.y, game, food1, agent)
#
#                 if game.crash == True:
#                     secondary_q_array.append(agent.reward)
#                     print('CRASH', secondary_q_array)
#
#                 else:
#
#                     secondary_q = agent.predict_q(agent.model, temp_state, j)
#                     secondary_q_array.append(secondary_q)
#                 agent.replay(game, player1, food1, agent.actual)
#             max_target_q = max(secondary_q_array)
#             target_q = [temp_reward + agent.gamma * max_target_q]
#             print('1', target_q)
#
#         else:
#             target_q = temp_reward
#             print('2', target_q)
#         #agent.agent_target = target_q
#         #agent.agent_predict = predicted_q
#         #agent.store_memory(state2, final_move, target_q[0])
#         print('3', state2)
#         print('3', final_move)
#         print('3', target_q[0])
#         print('3', [np.hstack(np.array([state2, final_move, target_q[0][0]]))])
#         agent.dataframe = agent.dataframe.append([np.hstack(np.array([state2, final_move, target_q[0][0]]))])
#         agent.train2_q(agent.dataframe[agent.dataframe.columns[:17]], agent.dataframe[agent.dataframe.columns[17]])
#         display(player1, food1, game)
#         pygame.time.wait(game.speed)
#     print('GAME: ',counter_games)
#     counter_games = counter_games + 1
#
# agent.model.save_weights('weights.hdf5')

run()