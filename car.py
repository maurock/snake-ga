import pygame

pygame.init()
display_width= 800
display_height = 800

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('SnakeGen')
clock = pygame.time.Clock()

img = cv2.imread('racecar.png')
carImg = pygame.image.load(img)


def car(x,y):
    gameDisplay.blit(carImg,(x,y))



def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',115)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)

def crash():
    message_display('You Crashed')


def game_loop():
    x = display_width * 0.45
    y = display_height * 0.5

    x_change = 0
    y_change = 0
    gameExit = False

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -25
                elif event.key == pygame.K_RIGHT:
                    x_change = 25
                elif event.key == pygame.K_UP:
                    y_change = - 25
                elif event.key == pygame.K_DOWN:
                    y_change = 25

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    x_change = 0
                    y_change = 0
            print(event)
        x = x + x_change
        y = y + y_change
        gameDisplay.fill(white)
        car(x,y)

        if x < 0 or x > display_width:
            gameExit = True

        if y == 50:
            crash()

        pygame.display.update()
        clock.tick(60)

game_loop()
pygame.quit()
quit()