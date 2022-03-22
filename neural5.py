#=========================wall of Richard Burgins hard work, may it stand eternal and unto it I pledge my life======================================================
#===================================================================================================================================================================
#===================================================================================================================================================================

import copy
import math
import random
import numpy


#Objects==========
class NeuralNet:
    def __init__(self):
        self.layernum = 4 #number of layers
        self.neuronnum = [784,16,16,10] #neurons for each layer
        self.inputnum = [1,784,16,16] #inputs for each layer

        self.layers = []
        for l in range(self.layernum): #cycles through the layers
            self.layers.append([]) #adds new layer
            for _ in range(self.neuronnum[l]): #cycles through the neurons
                if l == 0:##
                    sig = "INPUT"##
                elif l == self.layernum-1:##
                    sig = "OUTPUT"##
                else:##
                    sig = "HIDDEN"##
                self.layers[l].append(Neuron(self.inputnum[l], sig)) #adds new neuron ##sig

        self.cost = 0

    def Forward(self, inputs):
        outputs = []
        for l in range(self.layernum): #cycles through layers
            if l != 0: #input layer
                thislayer = copy.deepcopy(nextlayer)
            nextlayer = []
            
            for n in range(self.neuronnum[l]): #cycles through neurons
                if l == 0:#if input nodes
                    nextlayer.append(self.layers[l][n].ActivationFunction([inputs[n]]))
                elif l == self.layernum-1: #if output nodes
                    outputs.append(self.layers[l][n].ActivationFunction(thislayer))
                else: #the rest
                    nextlayer.append(self.layers[l][n].ActivationFunction(thislayer))
        return outputs

    def CostFinder(self, correctoutput):
        #real output
        realoutput = []
        for i in self.layers[-1]:
            realoutput.append(i.activation)

        #crunch
        a, b = realoutput, correctoutput
        total = 0
        for i in range(10):
            total += (a[i] - b[i])**2

        #commit
        self.cost = total

    def BackPropRecord(self, correctvalue):
        #find cost
        correctoutput = [0,0,0,0,0,0,0,0,0,0]
        correctoutput[correctvalue] = 1
        self.CostFinder(correctoutput)

        #reset units
        for l in self.layers:
            for u in l:
                u.activationnudges = []

        #start recursion
        for i in range(1, len(self.layers)+1): #cycles through layers
            for u in self.layers[-i]: #cycles through units
                if i == len(self.layers):
                    prevlayer = 0
                else:
                    prevlayer = self.layers[-i-1]

                if i == 1: #discriminate output layer
                    feed = correctoutput[self.layers[-1].index(u)] - u.activation

                    u.activationnudges.append(feed)
                u.Record(prevlayer)

    def BackPropAdjust(self):
        #print("=====================================")
        #print(self.layers[0][0].weightrecord)
        for l in self.layers:
            for u in l:
                u.Adjust()

#======= 
class Neuron:
    def __init__(self, inputnumber, sig):##sig
        self.inputnumber = inputnumber
        self.bias = random.uniform(0,1)
        self.weight = []
        for _ in range(self.inputnumber):
            self.weight.append(random.uniform(-1,1))
        self.weightnum = len(self.weight)

        #backprop
        self.biasrecord = []
        self.weightrecord = []
        for _ in range(self.inputnumber):
            self.weightrecord.append([])

        ##diag
        self.sig = sig
    
    def ActivationFunction(self, processinginput):
        total = sum(numpy.multiply(processinginput,self.weight))
        self.activation = self.sigmoid(total + self.bias)
        return self.activation

    def sigmoid(self, x):
        result = 1/(1+math.exp(-x))

        return result

    def Record(self, prevlayer):
        #find desired change to activation
        dc = sum(self.activationnudges)/len(self.activationnudges)

        #adjust bias according to desired change
        self.biasrecord.append(dc)

        #cycle through units connected via weights
        if prevlayer:
            l = len(prevlayer)
            for i in range(l):
                
                #adjust the weight according to the connected unit's activation and the sign of the desired change
                c = 0.01 #weight change coefficient
                ua = prevlayer[i].activation #units activation
                v = c * ua #committed value
                if dc <= 0:
                    v = -v
                self.weightrecord[i].append(v)

                #relay a change to the connected unit's activation: relative to the sign of dc, the sign of the connected weight, and proportional to the magnitude of the weight
                w = self.weight[i]
                v = (abs(self.weight[i])+1)/2  #proportion found
                if dc > 0: #relative to the sign of the dc
                    rc = v #increase a
                else:
                    rc = -v #decrease a
                if w > 0: #relative to the sign of the weight
                    rc = rc #increase a
                else:
                    rc = -rc #decrease a

                prevlayer[i].activationnudges.append(rc) #relay the change
        else:
            #since a of input is +ve, use dc
            v = dc
            self.weightrecord[0].append(v)

    def Adjust(self):
        #bias
        x = sum(self.biasrecord)/len(self.biasrecord)
        self.bias += x
        if self.bias > 1:
            self.bias = 1
        elif self.bias < 0:
            self.bias = 0

        #weights
        for i in range(self.inputnumber):
            w = self.weight[i]
            x = sum(self.weightrecord[i])/len(self.weightrecord[i])
            w += x
            if w > 1:
                w = 1
            elif w < -1:
                w = -1

        #reset
        self.biasrecord = []
        self.weightrecord = []
        for _ in range(self.inputnumber):
            self.weightrecord.append([])


#Functions==========
def Write(gennumber, net, name):
    f = open("files\\nets\\"+name+".txt", "w")

    #encodes the data
    f.write((str(gennumber)+","))
    for l in t: #cycles through layers
        f.write("L,")
        for n in l: #cycles through neurons
            f.write("n,")
            for w in n.weight: #cycles through weights
                f.write(("w,"+str(w)+","))
            f.write(("b,"+str(n.bias)+","))
    f.close()

def Read(name):
    f = open("files\\savefiles\\"+name+".txt", "r")

    #nets
    newnet = NeuralNet()

    #decodes the data
    block = f.readline()
    chunks = block.split(",")

    #declaring indexes
    gennumber = int(chunks[0])
    for i in range(len(chunks)):
        if chunks[i] == "L":#cycles through the layers
            Li += 1
            ni = -1
        if chunks[i] == "n":#cycles through neurons
            ni += 1
            wi = -1
        if chunks[i] == "w":#cycles through weights
            wi += 1
            newnet.layers[Li][ni].weight[wi] = float(chunks[i+1])
        if chunks[i] == "b":
            newnet.layers[Li][ni].bias = float(chunks[i+1])
    f.close()


    return newnet, gennumber
