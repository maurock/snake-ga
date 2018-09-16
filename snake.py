import pygame
import cv2
from random import randint
import pandas as pd


pygame.init()
display_width = 400
display_height = 400

black = (0, 0, 0)
white = (255, 255, 255)

position = []

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('SnakeGen')
clock = pygame.time.Clock()

img = cv2.imread('ball.png')
img = cv2.resize(img, (20, 20))
cv2.imwrite('ball_sized.png',img)
carImg = pygame.image.load('ball_sized.png')
img = cv2.imread('pizza.png')
img = cv2.resize(img, (20, 20))
cv2.imwrite('pizza_sized.png',img)
pizzaImg = pygame.image.load('pizza_sized.png')


def car(x,y):
    gameDisplay.blit(carImg,(x, y))


def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()
    pygame.time.wait(1000)


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
    return 1, False, True, False, False, 0, position, x, y, x_food, y_food, -20 ,0 , 1

def loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,move):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and horizontal == False:
                x_change = -20
                y_change = 0
                vertical = False
                horizontal = True
                move=1
            elif event.key == pygame.K_RIGHT and horizontal == False:
                x_change = 20
                y_change = 0
                vertical = False
                horizontal = True
                move=2
            elif event.key == pygame.K_UP and vertical == False:
                y_change = - 20
                x_change = 0
                vertical = True
                horizontal = False
                move=3
            elif event.key == pygame.K_DOWN and vertical == False:
                y_change = 20
                x_change = 0
                vertical = True
                horizontal = False
                move=4

    x = x + x_change
    y = y + y_change
    if (position[-1][0] != x or position[-1][1] != y):
        if [x,y] in position:
            crash()
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
        print(len(position))
        x_temp=position[len(position)-1-i][0]
        y_temp=position[len(position)-1-i][1]
        car(x_temp, y_temp)
    pygame.time.wait(100)
    pygame.display.update()


    if x < 0 or x > display_width-20 or y<0 or y > display_height-20:
        crash()
        gameExit = True

    clock.tick(100)
    return food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,move



dataset=pd.DataFrame()
t=0
while t <10:
    food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,move = initialize()
    while gameExit==False:
        #print(position)
        print('1',food)
        print('2',gameExit)
        print('3',horizontal)
        print('4',vertical)
        print('5',eaten)
        print('6',cont)
        print('7',position)
        print('8',x)
        print('9',y)
        print('10',x_food)
        print('11',x_change)
        print('12',y_change)

        food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,move= loop_move(food, gameExit, horizontal, vertical, eaten, cont, position, x, y, x_food, y_food, x_change, y_change,move)

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
        array = [right_food,left_food,up_food,down_food,right_free,left_free,up_free,down_free,move]
        print(array)

        dataset = dataset.append([array])
    t=t+1

colnames=['right_food','left_food','up_food','down_food','right_free','left_free','up_free','down_free','move']
dataset.to_csv(path_or_buf='snakemove2.csv',sep = ',',index=False,header=colnames)

pygame.quit()
quit()

'''
#Execute game
game_loop()
pygame.quit()
quit()
'''