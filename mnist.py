import numpy as np
import sys
import math
import random
import ast
from copy import deepcopy
import os
import numpy as np
import function

bestscores = []
bestgens = []
population_size = 6
saved = 5
board_size = 28
struct = [board_size**2, 128, 10]
generation = max([0]+[int(x[4:]) for x in os.listdir() if "genf" in x])
networks = []
muts = 100
mut_am = 7.5
simd = {0:set(), 1: set([7]), 2: set(), 3:{8,9}, 4:set(), 5:set([6]), 6:{5, 9}, 7:set([1]), 8:set([3]), 9:{3,6}}
np.set_printoptions(threshold=sys.maxsize)

#pix = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
#with open("mnist_pics.txt", "w+") as writer:
#    writer.write(str([int(''.join(["1" if i>127 else "0" for i in np.concatenate(j).ravel()]), 2) for j in pix]))

with open("mnist_pics.txt", "r") as reader:
    pics = ast.literal_eval(reader.read())

pics = [[int(b) for b in str(bin(a))[2:].zfill(28*28)] for a in pics]

#print(len(pics[0]))
#[print(pics[0][x*28: x*28+28]) for x in range(28)]

with open("mnist_labels.txt", "r") as reader:
    labs = ast.literal_eval(reader.read())
#file2 = 'data/train-labels-idx1-ubyte'
#labs = idx2numpy.convert_from_file(file2)
#print(labs[:20])

print("Loaded dataset")

class Network:
    def __init__(self, we, bi):
        self.weights = we
        self.biases = bi

    def __init__(self):
        self.weights = [np.random.normal(0, 1, [struct[0], struct[1]]), np.random.normal(0, 1, [struct[1], struct[2]])]
        self.biases = [np.zeros((1, struct[1])), np.zeros((1, struct[2]))]
        #self.eva = np.vectorize(self.evaluate)
        #print(len(self.weight1[0]), len(self.weight1[1]))

    def ev(self, board):
        return [self.evaluate(i) for i in board]

    def evaluate(self, board):

        z1 = np.dot(board, self.weights[0]) + self.biases[0]
        a1 = function.relu(z1)
        z2 = np.dot(a1, self.weights[1]) + self.biases[1]
        last_layer = function.softmax(z2)[0]
        #print("Evaluated")
        if sum(last_layer)==0: per = 0
        else: per = max(last_layer)/sum(last_layer)
        if per == 1: per = .999
        if per >= 1: print("Error in evaluation")
        if per<0: print("Negative throughput")
        #print(np.argmax(last_layer)+per)
        return np.argmax(last_layer)+per

def printer(x):
    print(x)
    return 0

def test():
    nettup = make_net()
    n = Network(nettup[0], nettup[1])
    n.evaluate([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 1)

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=0)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return x * (x>0)

def negative(x):
    return -x

sm = np.vectorize(softmax)
rel = np.vectorize(relu)
sig = np.vectorize(sigmoid)
neg = np.vectorize(negative)


def make_net():
    weight1 = []
    bias1 = []
    for e in range(len(struct)):
        if e>0:
            b = [random.uniform(-1, 1) for _ in range(struct[e])]
            bias1.append(b)
        if e<len(struct)-1:
            w = [random.uniform(-1, 1) for _ in range(struct[e]*struct[e+1])]
            weight1.append(w)

    return Network(weight1, bias1)

def save():
    with open("genf"+str(generation), 'w+') as outfile:
        outfile.write(str([(n.weights, n.biases) for n in networks[:saved]]))

def load(gen_num):
    with open("genf"+str(gen_num), 'r') as infile:
        arr = infile.read()
    return ast.literal_eval(arr)

def game_over(board):
    b = np.array(sum(board, []))
    check = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for i in check:
        if b[i[0]]==b[i[1]] and b[i[1]]==b[i[2]] and b[1]!=0:
            return b[i[0]]
    if list(b).count(0)==0: return -5 #Tie
    return 0

def train():
    global generation
    global networks
    visit2 = False
    counterb = random.randint(0, 149)
    if generation > 0:
            networks = [Network(i[0], i[1]) for i in load(generation)]
            print("Loaded generation", generation)
    else: networks = [Network() for _ in range(population_size*2)]
    while True:
        if counterb == 75: counterb = 0
        networks +=  [Network() for _ in range(population_size-len(networks))]
        print("Made networks")
        print("Batch", counterb)
        #Mutation
        for index in list(range(len(networks)))+list(range(len(networks)//2)):
            net_cop = deepcopy(networks[index])
            for i in range(random.randint(1, muts-1)):
                if random.random()<.5:
                    p = random.randint(0,1)
                    q = random.randint(0, len(net_cop.biases[p])-1)
                    r = random.randint(0, len(net_cop.biases[p][q])-1)
                    net_cop.biases[p][q][r] += random.uniform(-mut_am, mut_am)
                else:
                    p = random.randint(0,1)
                    q = random.randint(0, len(net_cop.weights[p])-1)
                    r = random.randint(0, len(net_cop.weights[p][q])-1)
                    net_cop.weights[p][q][r] += random.uniform(-mut_am, mut_am)

            networks.append(net_cop)

        #Crossover
        cross_list = [(random.randint(0, (saved-1)//2), random.randint(0, (saved-1)//2)) for _ in range(saved)]*2
        cross_list += [(random.randint(0, saved-1), random.randint(0, len(networks)-1)) for _ in range(saved) if random.random()<.1]
        cross_list += [(random.randint(0, len(networks)-1), random.randint(0, len(networks)-1)) for _ in range(saved) if random.random()<.1]
        cross_list = [t for t in cross_list if t[0]!=t[1]]
        for pair in cross_list:
            out1 = deepcopy(networks[pair[0]])
            out2 = deepcopy(networks[pair[1]])
            if random.random()<.5:
                p = random.randint(0,1)
                q = random.randint(0, len(net_cop.biases[p])-1)
                col_start = random.randint(0, len(out1.biases[p][q])-2)
                col_end = random.randint(col_start+1, len(out1.biases[p][q])-1)
                temp1 = out1.biases[p][q][col_start:col_end]
                for a in range(col_start, col_end):
                    out1.biases[p][q][a] = out2.biases[p][q][a]
                for z in range(col_start, col_end):
                    out2.biases[p][q][z] = temp1[z-col_start]
            else:
                p = random.randint(0,1)
                q = random.randint(0, len(net_cop.weights[p])-1)
                col_start = random.randint(0, len(out1.weights[p][q])-2)
                col_end = random.randint(col_start+1, len(out1.weights[p][q])-1)
                temp1 = out1.weights[p][q][col_start:col_end]
                for a in range(col_start, col_end):
                    out1.weights[p][q][a] = out2.weights[p][q][a]
                for z in range(col_start, col_end):
                    out2.weights[p][q][z] = temp1[z-col_start]

            networks.append(out1)
            networks.append(out2)

        #Battle
        fit_dict = {}
        r  = counterb
        for net in networks:
            ans_list = net.ev(pics[r*400:(r+1)*400])
            outcome = 0
            right = 0
            for i in range(len(ans_list)):
                #print(ans_list[i])
                if int(ans_list[i])==labs[r*400:(r+1)*400][i]:
                    right += 1
                    outcome += .9 + .1*(ans_list[i]%1)
                else: outcome -= .1
            fit_dict[net] = outcome
            print("Finished batch with accuracy", right/400)
        #counterb += 1
        sorted_nets = sorted(fit_dict.items(), reverse=True, key = lambda x: x[1])
        temp_networks = []
        for i in range(saved):
            temp_networks.append(sorted_nets[i][0])
        networks = temp_networks
        generation += 1
        bestf = sorted_nets[0][1]/400
        if visit2: visit2=False
        elif bestf<.15: visit2 = True
        if not visit2: counterb+=1
        with open("scores.txt", "a+") as sc:
            sc.write(" " + str(bestf))
        bestscores.append(bestf)
        bestgens.append(generation)
        print("Generation", generation, "best fitness", bestf)
        print("Population size", len(sorted_nets))
        if generation % 50 == 0:
            #plt.plot(bestgens, bestscores)
            #plt.savefig("fitness.png")
            save()

if __name__ == "__main__":
    train()
