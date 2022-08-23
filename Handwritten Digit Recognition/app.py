import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
from numpy import testing
from numpy.lib.type_check import imag
from pygame import image

WINDOWSIZEX = 640
WINDOWSIZEY = 480

#Initialize pygame
pygame.init()

FONT = pygame.font.Font(pygame.font.get_default_font(), 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit Board")

BOUNDARYIMC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

IMAGE_SAVE = True

MODEL = load_model("bestmodel.h5")

LABELS = {0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Siz",7:"Seven",8:"Eight",9:"Nine"}

isWriting = False

number_xcord = []
number_ycord = []

imagecnt = 0

PREDICT = True

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            isWriting = True

        if event.type == MOUSEMOTION and isWriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord,ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONUP:
            isWriting = False
            number_ycord = sorted(number_ycord)
            number_xcord = sorted(number_xcord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYIMC, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYIMC)
            rect_min_y, rect_max_y = number_ycord[0]-BOUNDARYIMC, min(number_ycord[-1]+BOUNDARYIMC, WINDOWSIZEX)

            number_xcord=[]
            number_ycord=[]

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            print(img_arr)

            if IMAGE_SAVE:
                cv2.imwrite("image.png",img_arr)
                imagecnt+=1

            if PREDICT:

                #textRecObj = image.get_rect()
                image = cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10), 'constant', constant_values= 0)
                image = cv2.resize(image,(28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                print(label)

                textSurface = FONT.render(label, True, RED, WHITE)
                #textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, (rect_min_x, rect_max_y))

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()

            

             