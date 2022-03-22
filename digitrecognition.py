import pygame
from pygame import FULLSCREEN
import random
import copy
import math
import os


#window setup
import ctypes
user32 = ctypes.windll.user32
global scr_width
global scr_height
scr_width = user32.GetSystemMetrics(0)
scr_height = user32.GetSystemMetrics(1)
window = pygame.display.set_mode((scr_width,scr_height),FULLSCREEN)
pygame.display.set_caption("POPLAR")
pygame.font.init()
from pygame.locals import *
pygame.init()

#BOARD ======================
class Board:
    def __init__(self):
        self.RUN = True

        #colours
        self.backC = (33,44,48)
        self.baseC = (40,62,63)
        self.textC = (247,216,148)
        self.highC = (60,82,83)
        self.subC = (130,51,60)
        self.deadC = (193,193,193)
        self.trackC = (130,51,60)

        #options
        OpFont = pygame.font.SysFont('', 25)
        save = [scr_width-200, 0, 200, 100]
        load = [scr_width-200, 100, 200, 100]
        self.optionplacement = [save,load]
        self.optionstitles = [OpFont.render("SAVE", False, self.textC), OpFont.render("LOAD", False, self.textC)]
        self.textbox = False
        self.textboxchoice = -1
        self.textboxtext = ""

        #tracking
        self.TrackFont = pygame.font.SysFont('', 80)
        self.PredictFont = pygame.font.SysFont('', 100)
        self.tracktitle = self.TrackFont.render("NEURAL NETS OUTPUT:", False, self.trackC)
        self.trackcoord = [700, 250]
        tl = [self.trackcoord[0]+250, self.trackcoord[1]+200]
        self.trackline = [tl, [tl[0]+150, tl[1]]]

        #subs
        self.SubFont = pygame.font.SysFont('', 35)
        self.subtexttitles = ["CORRECT ANSWER: ", "TOTAL TESTS: ", "TOTAL CORRECT: ", "% CORRECT: "]
        self.subcoord = [550, 10]

        #data
        mnist = tf.keras.datasets.mnist #28x28 images of hand-written digits 0-9
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.rawdata = [x_train, y_train]

        #performance
        self.totaltests = 0
        self.totalcorrect = 0

        #stall
        self.stall = False

        #net
        self.netstart = [550,350]

    def Show(self):
        self.ShowOptions()
        self.ShowSubs()
        self.ShowData()
        self.ShowTracking()

    def ShowOptions(self):
        #options=======
        M.highlight = -1
        #base
        for z in range(len(self.optionplacement)):
            i = self.optionplacement[z]
            
            #mouse over
            if (i[0] < M.coord[0] < i[0] + i[2]) and (i[1] < M.coord[1] < i[1] + i[3]):
                M.highlight = z
                pygame.draw.rect(window, self.highC, (i))
            else:
                pygame.draw.rect(window, self.baseC, (i))
        
            #text
            window.blit(self.optionstitles[z], (i[0]+i[2]//2-10, i[1]+i[3]//2))
        
        #texbox
        if self.textbox:
            #line
            l = self.optionplacement[0]
            coords = [[l[0]-200, l[1]+50], [l[0]-25, l[1]+50]]
            pygame.draw.line(window, self.baseC, coords[0], coords[1])

            #text
            BoxFont = pygame.font.SysFont('', 25)
            Text = BoxFont.render(self.textboxtext, False, self.baseC)
            window.blit(Text, (coords[0][0], coords[0][1]-17))

    def Textbox(self):
        act = False
        for event in pygame.event.get():
            #keys
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RETURN]:
                    act = True
                elif keys[pygame.K_BACKSPACE]:
                    self.textboxtext = self.textboxtext[:-1]
                else:
                    self.textboxtext += event.unicode

        #save + load
        if act:
            if self.textboxchoice == 0:
                neu.Write(self.gen, R.Nn, self.textboxtext)
            elif self.textboxchoice == 1:
                try:
                    newnet, self.gen = neu.Read(self.textboxtext)
                    R = Reader()
                    R.Nn = newnet
                except:
                    FileNotFoundError
            
            self.textbox = False

    def ShowSubs(self):
        X, Y = copy.deepcopy(self.subcoord)

        data = [self.correctvalue, self.totaltests, self.totalcorrect]
        if self.totaltests > 0:
            percent = str(round((self.totalcorrect/self.totaltests)*100, 2)) + "%"
        else:
            percent = ""
        data.append(percent)

        for i in range(len(self.subtexttitles)):
            text = self.SubFont.render(self.subtexttitles[i]+str(data[i]), False, self.subC)
            window.blit(text, (X, Y))
            Y += 25

    def ShowData(self):
        for i in self.pixels:
            i.Show()

    def FindData(self):
        index = random.randint(0,59999)

        #picture
        self.pixels = []
        start = [100, 350]
        dim = 15
        for x in range(28):
            for y in range(28):
                coord = [start[0] + (dim*x), start[1] + (dim*y)]
                self.pixels.append(DataPixels(coord, self.rawdata[0][index][y][x], dim))

        #data
        self.normaliseddata = []
        self.correctvalue = self.rawdata[1][index]
        for x in range(28):
            for y in range(28):
                normalise = self.rawdata[0][index][y][x]/255
                self.normaliseddata.append(normalise)

    def FindTracking(self, prediction):
        self.tracktext = self.PredictFont.render(str(prediction), False, self.trackC)

    def ShowTracking(self):
        #title
        window.blit(self.tracktitle, self.trackcoord)
        
        #prediction
        t = self.trackline
        pygame.draw.line(window, self.subC, t[0], t[1], 5)
        window.blit(self.tracktext, (t[0][0]+50, t[0][1]-self.tracktext.get_height()) )

        #past prediction
        R.ShowPastPrediction()
        R.ShowCostData()
        self.ShowSubs()
        ##self.ShowNet()
    
    def FindNet(self):
        X, Y = self.netstart
        net = R.Nn
        record = []
        
        for l in range(net.layernum): #cycles through layers
            record.append([])
            for n in range(net.neuronnum[l]): #cycles through neurons
                if l != 0:
                    X = self.netstart[0] + (75 * l)
                    Y = self.netstart[1]  + (25 * n)
                elif 402 > n > 382:
                    X = self.netstart[0]
                    Y = self.netstart[1]  + (20 * (n-382))
                record[-1].append([[X, Y], net.layers[l][n]])

        det = []
        for l in range(len(record)): #cycles through layers
            det.append([])
            for n in range(len(record[l])): #cycles through neurons
                unit = record[l][n][-1]
                coord = record[l][n][0]
                a = unit.activation
                c = round(255*a)
                s = [c,c,c]

                det[-1].append([coord, s, []])
                for w in range(len(unit.weight)): #cycles through weights
                    if (l == 2 and 402 > w > 382) or l != 2:
                        det[-1][-1][-1].append(unit.weight[w])
                #end result index with: [[X, Y], activation shade, [values of all weights]]
        self.netdetails = det

    def ShowNet(self):
        prevlayer = []
        det = self.netdetails
        for l in det: #cycles through layers
            thislayer = []
            for n in l: #cycles through neurons
                thislayer.append(n[0])
                #print(n[0])
                pygame.draw.circle(window, n[1], n[0], 15) #show shaded unit

                if len(prevlayer) > 0:
                    for w in range(len(n[2])):
                        if (l == 2 and 402 > n[2][w] > 382) or l != 2:
                            #colour
                            if n[2][w] < 0:
                                colour = (63,72,204) #blue
                            else:
                                colour = (255,0,0) #red

                            #print(prevlayer)
                            try:
                                p = prevlayer[w]
                            except IndexError:
                                print(w)
                            pygame.draw.line(window, colour, (n[0]), p, round(abs(n[2][w]*10)))
                prevlayer = copy.deepcopy(thislayer)

#MOUSE ======================
class Mouse:
    def __init__(self):
        self.coord = [0, 0]
        self.coord[0], self.coord[1] = pygame.mouse.get_pos()
        
        #buttons
        self.leftclick = False
        self.highlight = -1

    def Input(self):
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            #mouse
            if event.type == pygame.MOUSEMOTION:
                self.coord[0], self.coord[1] = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    self.LClickDOWN()
            if event.type == pygame.MOUSEBUTTONUP:
                if self.leftclick:
                    self.LClickUP()

            #keys
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    if B.stall:
                        B.stall = False
                    else:
                        B.stall = True
        if keys[pygame.K_k]:#kill switch
            B.RUN = False

    def LClickDOWN(self):
        if self.highlight != -1:
            self.leftclick = True
        else:
            if B.stall:
                R.Ai()
                R.NextRound()

        #exit
        if self.coord[0] > scr_width-50 and self.coord[0] < scr_width and self.coord[1] < scr_height and self.coord[1] > scr_height-50:
            B.RUN = False

    def LClickUP(self):
        if self.highlight != -1:
            B.textbox = True
            B.textboxchoice = copy.deepcopy(self.highlight)
            B.textboxtext = ""

        self.leftclick = False

#ARTIFICIAL INTELLIGENCE ======================
class Reader:
    def __init__(self):
        self.Nn = neu.NeuralNet()
        self.currenttest = -1

        #past predictions ===
        #declare
        self.hindsight = 150
        self.length = 1000//self.hindsight
        height = 25
        self.depth = 10 # also batch size
        startcoord = [600, 750]

        #commit
        self.pastpredictions = []
        for x in range(self.hindsight):
            self.pastpredictions.append([])
            coord = [startcoord[0] + (self.length*x), startcoord[1]]
            for d in range(self.depth):
                self.pastpredictions[-1].append(PredictionPixels(coord, height, self.length, d*25))

        #costplotter
        self.batchcost = []
        self.cost = 0
        self.startplot = [startcoord[0]+self.length/2, startcoord[1]]
        self.costplots = []
        for x in range(self.hindsight):
            coord = [self.startplot[0] + (self.length*x), self.startplot[1]]
            self.costplots.append(coord)

    def Ai(self):
        #gather inputs
        inputs = copy.deepcopy(B.normaliseddata)
        result = self.Nn.Forward(inputs)
        B.FindNet()

        #prediction
        prediction = result.index(max(result))
        B.FindTracking(prediction)

        #batchwork
        if self.currenttest == self.depth-1: #next batch
            self.PastPredictionNudge()
            self.currenttest = 0
            self.FindCostData()
            self.Nn.BackPropAdjust() #change weights and biases
        else: #continue batch
            self.batchcost.append(self.Nn.cost)
            self.currenttest += 1

        #prediction check
        if prediction == B.correctvalue: #correct
            self.pastpredictions[0][self.currenttest].colour = (0,255,0)
            B.totalcorrect += 1
        else: #wrong
            self.pastpredictions[0][self.currenttest].colour = (255,0,0)
        B.totaltests += 1

    def NextRound(self):
        B.FindData()
        self.Nn.BackPropRecord(B.correctvalue)
    
    def PastPredictionNudge(self):
        #record
        colours = []
        for z in self.pastpredictions:
            colours.append([])
            for i in z:
                colours[-1].append(i.colour)
        
        #organise
        colours.pop()
        new = []
        for i in range(self.depth):
            new.append((70,70,70))
        colours.insert(0, new)

        #commit
        for z in range(self.hindsight):
            for i in range(self.depth):
                self.pastpredictions[z][i].colour = colours[z][i]

    def ShowPastPrediction(self):
        for z in self.pastpredictions:
            for i in z:
                i.Show()

    def FindCostData(self):
        #new
        self.cost = sum(self.batchcost)/len(self.batchcost)

        #nudge
        self.costplots.pop()
        for i in self.costplots:
            i[0] += self.length

        #add
        coord = [self.startplot[0], self.startplot[1] - self.cost*10]
        self.costplots.insert(0, coord)

        self.batchcost = []

    def ShowCostData(self):
        pygame.draw.lines(window, B.textC, False,  self.costplots, 2)

#EFFECTS ======================
class DataPixels:
    def __init__(self, coord, intensity, dim):
        self.coord = copy.deepcopy(coord)
        self.colour = (intensity, intensity, intensity)
        self.dim = dim
    
    def Show(self):
        pygame.draw.rect(window, self.colour, (self.coord[0], self.coord[1], self.dim, self.dim))

class PredictionPixels:
    def __init__(self, coord, height, length, depth):
        self.colour = (70,70,70)
        self.coord = [coord[0], coord[1] + depth]
        self.height = height
        self.length = length
    
    def Show(self):
        pygame.draw.rect(window, self.colour, (self.coord[0], self.coord[1], self.length, self.height))


import neural5 as neu
import tensorflow as tf

B = Board()
M = Mouse()
R = Reader()

B.FindData()

while B.RUN:
    pygame.time.delay(1)
    window.fill(B.backC)
    
    #input
    if B.textbox:
        B.Textbox()
    else:
        M.Input()
    if not B.stall:
        R.Ai()
        data = R.Nn.cost

    #show
    B.Show()

    #reset
    if not B.stall:
        R.NextRound()

    pygame.display.update()