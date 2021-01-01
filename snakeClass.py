import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch 
from GPyOpt.methods import BayesianOptimization
from bayesOpt import *
import datetime
import distutils.util
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = False
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


class Game:
    """ Initialize PyGAME """
    
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):
    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food(object):
    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent, batch_size):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev


def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent, params['batch_size'])
        if params['display']:
            display(player1, food1, game, record)
        
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (steps < 100):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, player1, food1)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(3)[randint(0,2)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            state_new = agent.get_state(game, player1, food1)

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            
            # if food is eaten, steps is set to 0
            if reward > 0:
                steps = 0
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)
            if params['display']:
                display(player1, food1, game, record)
                pygame.time.wait(params['speed'])
            steps+=1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
    return total_score, mean, stdev

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed
    if args.bayesianopt:
        bayesOpt = BayesianOptimizer(params)
        bayesOpt.optimize_RL()
    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params)