import pygame
import cv2
from random import randint
import csv
import numpy as np
import pandas as pd


pygame.init()
display_width= 400
display_height = 400

black = (0,0,0)
white = (255,255,255)

position = []

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('SnakeGen')
clock = pygame.time.Clock()


carImg = pygame.image.load('ball_sized2.png')
pizzaImg = pygame.image.load('pizza_sized.png')


def car(x,y):
    gameDisplay.blit(carImg,(x,y))


def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    #pygame.display.update()
    #pygame.time.wait(100)


def crash():
    message_display('OPS SEI MORTO!')

def food_coord():
    x_rand = randint(20, display_width - 20)
    x_rand = x_rand - x_rand % 20
    y_rand = randint(20, display_width - 20)
    y_rand = y_rand - y_rand % 20
    return x_rand, y_rand

def display_food(x,y):
    gameDisplay.blit(pizzaImg, (x, y))


def initialize():
    position=[]
    x = 0.45 * display_width
    y = 0.5 * display_height
    x = x - x % 20
    y = y - y % 20

    position.append([x,y])

    food_coord()
    x_food = food_coord()[0]
    y_food = food_coord()[1]
    return 1, False, 0, 0, False, 0, position, x, y, x_food, y_food, 0 ,0
    '''
    food = 1

    x_change = 0
    y_change = 0
    gameExit = False
    horizontal = False
    vertical = False
    eaten = True
    cont = 0
    '''

def loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,move):
    if move == 1 and horizontal == 0:       #left
        x_change = -20
        y_change = 0
        vertical = 0
        horizontal = 1
    elif move == 2 and horizontal == 0:        #right
        x_change = 20
        y_change = 0
        vertical = 0
        horizontal = 1
    elif move == 3 and vertical == 0:           #top
        y_change = - 20
        x_change = 0
        vertical = 1
        horizontal = 0
    elif move == 4 and vertical == 0:           #bottom
        y_change = 20
        x_change = 0
        vertical = 1
        horizontal = 0

    x = x + x_change
    y = y + y_change
    if (position[-1][0] != x or position[-1][1] != y):
        if [x,y] in position:
            crash()
            pygame.time.wait(100)
            gameExit = True

        if food>1:
            for i in range(0,food-1):
                position[i][0]= position[i+1][0]
                position[i][1] = position[i+1][1]
        position[-1][0] = x
        position[-1][1] = y

    if eaten == True:
        food_coord()
        x_food = food_coord()[0]
        y_food = food_coord()[1]
        eaten = False

    if x == x_food and y == y_food:
        eaten= True
        food = food + 1
        position.append([x,y])

    gameDisplay.fill(white)
    display_food(x_food, y_food)

    for i in range(food):
        #print(len(position))
        x_temp=position[len(position)-1-i][0]
        y_temp=position[len(position)-1-i][1]
        car(x_temp, y_temp)
    pygame.time.wait(100)
    pygame.display.update()


    if x < 0 or x > display_width-20 or y<0 or y > display_height-20:
        crash()
        gameExit = True

    clock.tick(1000)
    cont = cont+1

    return food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change

def fitness(x_food, x, y_food, y, food):
    score = (1-(np.sqrt((x_food-x)**2 + (y_food - y)**2))/566 + food)
    return score


#Execute game
#food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = initialize()
#l r t b
'''
dataset=pd.DataFrame()
t=1
while(t<100):
    gameExit=False

    food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = initialize()

    while gameExit==False:
        #print(position)
        if x_food < x and [(x-20),y] not in position and (x-20)>0:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,1)
            move=1
        elif x_food > x and [(x+20),y] not in position and (x+20)<display_width:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 2)
            move=2
        elif y_food < y and [x,(y-20)] not in position and (y-20)>0:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 3)
            move=3
        elif y_food > y and [x,(y+20)] not in position and (y+20) < display_height:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 4)
            move=4
        elif [(x-20),y] not in position and horizontal==False and (x-20)>0:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 1)
            move=1
        elif [(x+20),y] not in position and horizontal==False and (x+20)<display_width:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 2)
            move=2
        elif [x,(y-20)] not in position and vertical==False and (y-20)>0:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 3)
            move=3
        elif [x,(y+20)] not in position and vertical==False and (y+20) < display_height:
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, 4)
            move=4
        else:
            move = randint(1,4)
            food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change = loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change, move)


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
        if [(x+20),y] not in position and (x+20)<display_width:
            right_free = 1
        else:
            right_free = 0
        if [(x-20),y] not in position and (x-20)>0:
            left_free = 1
        else:
            left_free=0
        if [x,(y-20)] not in position and (y-20)>0:
            up_free = 1
        else:
            up_free = 0
        if [x,(y+20)] not in position and (y+20)<display_height:
            down_free = 1
        else:
            down_free=0



        array = [right_food,left_food,up_food,down_food,right_free,left_free,up_free,down_free, vertical, horizontal, move]
        #for i in range(len(grid)):
            #array.append(grid[i])
        dataset = dataset.append([array])

    print(t)
    t=t+1

dataset.to_csv(path_or_buf='snakemove6.csv',sep = ',',index=False, header=False)

pygame.quit()
quit()
'''