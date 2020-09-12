# Deep Reinforcement Learning
## Project: Train AI to play Snake

## Introduction
The goal of this project is to develop an AI Bot able to learn how to play the popular game Snake from scratch. In order to do it, I implemented a Deep Reinforcement Learning algorithm. This approach consists in giving the system parameters related to its state, and a positive or negative reward based on its actions. No rules about the game are given, and initially the Bot has no information on what it needs to do. The goal for the system is to figure it out and elaborate a strategy to maximize the score - or the reward.
We are going to see how a Deep Q-Learning algorithm learns how to play snake, scoring up to 50 points and showing a solid strategy after only 5 minutes of training.

## Install
This project requires Python 3.6 with the pygame library installed, as well as Keras with Tensorflow backend.
```bash
git clone git@github.com:maurock/snake-ga.git
```

## Run
To run the game, executes in the snake-ga folder:

```python
python snakeClass.py --display=True --speed=50
```
Arguments description:

- --display - Type bool, default True, display or not game view
- --speed - Type integer, default 50, game speed

This will run and show the agent. The default configuration loads the file *weights/weights.hdf5* and runs a test.
The Deep neural network can be customized in the file snakeClass.py modifying the dictionary *params* in the function *define_parameters()*

To train the agent, set in the file snakeClass.py:
- params['load_weights'] = False
- params['train'] = True

In snakeClass.py you can set argument *--display*=False and *--speed*=0, if you do not want to see the game running. This speeds up the training phase.

## For Mac users
It seems there is a OSX specific problem, since many users cannot see the game running.
To fix this problem, in update_screen(), add this line.

```                              
def update_screen():
    pygame.display.update() <br>
    pygame.event.get() # <--- Add this line ###
```
