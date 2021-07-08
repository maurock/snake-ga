# Deep Reinforcement Learning
## Project: Train AI to play Snake
*UPDATE:*

This project has been recently updated and improved:
- It is now possible to optimize the Deep RL approach using Bayesian Optimization.
- The code of Deep Reinforcement Learning was ported from Keras/TF to Pytorch. To see the old version of the code in Keras/TF, please refer to this repository: [snake-ga-tf](https://github.com/maurock/snake-ga-tf).

## Introduction
The goal of this project is to develop an AI Bot able to learn how to play the popular game Snake from scratch. In order to do it, I implemented a Deep Reinforcement Learning algorithm. This approach consists in giving the system parameters related to its state, and a positive or negative reward based on its actions. No rules about the game are given, and initially the Bot has no information on what it needs to do. The goal for the system is to figure it out and elaborate a strategy to maximize the score - or the reward. \
We are going to see how a Deep Q-Learning algorithm learns how to play Snake, scoring up to 50 points and showing a solid strategy after only 5 minutes of training. \
Additionally, it is possible to run the Bayesian Optimization method to find the optimal parameters of the Deep neural network, as well as some parameters of the Deep RL approach.

## Install
This project requires Python 3.6 with the pygame library installed, as well as Pytorch. If you encounter any error with `torch=1.7.1`, you might need to install Visual C++ 2015-2019 (or simply downgrade your pytorch version, it should be fine). \
The full list of requirements is in `requirements.txt`.
```bash
git clone git@github.com:maurock/snake-ga.git
```

#### Windows (quick installation):
1) Install Anaconda with Python 3 (64bit)
2) Run "Anaconda Prompt (Anaconda3)" via Start menu
3) Install modules: pygame, torch and GPyOpt:
```
pip install pygame torch GPyOpt
```

## Run
To run and show the game, executes in the snake-ga folder:

```python
python snakeClass.py
```
Arguments description:

- --display - Type bool, default True, display or not game view
- --speed - Type integer, default 50, game speed

The default configuration loads the file *weights/weights.h5* and runs a test.

To train the agent, set in the file snakeClass.py:
- `params['train'] = True`
The parameters of the Deep neural network can be changed in *snakeClass.py* by modifying the dictionary `params` in the function `define_parameters()`

If you run snakeClass.py from the command line, you can set the arguments `--display=False` and `--speed=0`. This way, the game display is not shown and the training phase is faster.

## Optimize Deep RL with Bayesian Optimization
To optimize the Deep neural network and additional parameters, run:

```python
python snakeClass.py --bayesianopt=True
```

This method uses Bayesian optimization to optimize some parameters of Deep RL. The parameters and the features' search space can be modified in *bayesOpt.py* by editing the `optim_params` dictionary in `optimize_RL`.

## For Mac users
It seems there is a OSX specific problem, since many users cannot see the game running.
To fix this problem, in update_screen(), add this line.

```
def update_screen():
    pygame.display.update() <br>
    pygame.event.get() # <--- Add this line ###
```
