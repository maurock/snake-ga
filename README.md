# Deep Reinforcement Learning
## Project: Train a AI how to play Snake

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
python snakeClass.py --episodes=150 --display=True --speed=0 --load-weights=True
```
Arguments description:

- --episodes - Type integer, default 150, number of games
- --display - Type bool, default False, display or not game view
- --speed - Type integer, default 0, game speed
- --load-weight - Type bool, default False, loads pre-trained weights of agent for the neural network

This will run the agent. The Deep neural network can be customized in the file DQN.py.

In snakeClass.py you can set argument *--display*=True and *--speed*=50, if you want to see the game running.
