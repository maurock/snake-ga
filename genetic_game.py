import snakeGenetic
import keras
import pygame
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import load_model
import pandas as pd
import numpy as np

modello = load_model('snake8.h5')

df = pd.DataFrame()
t=0
while t<50:
    food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = snakeGenetic.initialize()
    horizontal=1
    vertical=0
    if x_food < x:
        right_food = 0
        left_food = 1
    else:
        right_food = 1
        left_food = 0
    if y_food < y:
        up_food = 1
        down_food = 0
    else:
        up_food = 0
        down_food = 1
    if [(x + 20), y] not in position and (x + 20) < snakeGenetic.display_width:
        right_free = 1
    else:
        right_free = 0
    if [(x - 20), y] not in position and (x - 20) > 0:
        left_free = 1
    else:
        left_free = 0
    if [x, (y - 20)] not in position and (y - 20) > 0:
        up_free = 1
    else:
        up_free = 0
    if [x, (y + 20)] not in position and (y + 20) < snakeGenetic.display_height:
        down_free = 1
    else:
        down_free = 0
    food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = snakeGenetic.loop_move(
        food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 1)

    array2 = [right_food, left_food, up_food, down_food, right_free, left_free, up_free, down_free, vertical, horizontal]
    move=1
    while gameExit == False:
        move = [move]
        array1 = np.concatenate([array2,move], axis=0)
        if x_food < x:
            right_food = 0
            left_food = 1
        else:
            right_food = 1
            left_food = 0
        if y_food < y:
            up_food = 1
            down_food = 0
        else:
            up_food = 0
            down_food = 1
        if [(x + 20), y] not in position and (x + 20) < snakeGenetic.display_width:
            right_free = 1
        else:
            right_free = 0
        if [(x - 20), y] not in position and (x - 20) > 0:
            left_free = 1
        else:
            left_free = 0
        if [x, (y - 20)] not in position and (y - 20) > 0:
            up_free = 1
        else:
            up_free = 0
        if [x, (y + 20)] not in position and (y + 20) < snakeGenetic.display_height:
            down_free = 1
        else:
            down_free = 0


        array2 = [right_food, left_food, up_food, down_food, right_free, left_free, up_free, down_free, vertical, horizontal]

        array = np.concatenate([array1,array2], axis = 0)
        array = np.array([array])
        pred = modello.predict(array)
        pos = np.where(np.amax(pred[0])==pred[0])
        print(pred)
        move_attempt =1+ pos[0][0]
        print(move_attempt)
        food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = snakeGenetic.loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, move_attempt)
        if x_change == 20:
            move = 1
        elif x_change == -20:
            move = 2
        elif y_change == -20:
            move = 3
        elif y_change == 20:
            move = 4

    t=t+1
