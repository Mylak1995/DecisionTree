import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

#Class Tree

class Tree:
    def __init__(self, op=None, kids=[], classification=None, pos=None, neg=None):
        self.op = op
        self.kids = kids
        self.classification = classification   # tree.class from the specs, but class is a keyword
        self.pos = pos
        self.neg = neg
    def is_leaf(self):
        return (self.classification != None)
    def count_leaves(self):
        if self.is_leaf():
            return 1
        else:
            return self.kids[0].count_leaves() + self.kids[1].count_leaves()

#Loading data
def load_data(filename):
    data = sio.loadmat(filename)
    return data['x'], data['y']

#Emotions handling

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def label_to_string(label):
    if label == 1:
        return 'anger'
    elif label == 2:
        return 'disgust'
    elif label == 3:
        return 'fear'
    elif label == 4:
        return 'happiness'
    elif label == 5:
        return 'sadness'
    elif label == 6:
        return 'surprise'
    else:
        return 'Unknown emotion label'

def string_to_label(s):
    if s == 'anger':
        return 1
    elif s == 'disgust':
        return 2
    elif s == 'fear':
        return 3
    elif s == 'happiness':
        return 4
    elif s == 'sadness':
        return 5
    elif s == 'surprise':
        return 6
    else:
        return -1

#Loading tree from file

#Loading tree from saved mat file
def load_from_mat(file):
    load=sio.loadmat(file)
    t_mat=load['tree']
    return(load_from_mat_aux(t_mat))

def load_from_mat_aux(t_mat):
    if t_mat['op'][0][0].size==0:
        return(Tree(classification=t_mat['class'][0][0][0][0]))
    else:
        return(Tree(op=t_mat['op'][0][0][0][0],kids=[load_from_mat_aux(t_mat['kids'][0][0][0][0]),load_from_mat_aux(t_mat['kids'][0][0][0][1])]))
#Tests one tree, returns depth of classification and classification
def classify(tree, features, depth=0):
    if tree.classification != None:
        return [tree.classification, depth+1]
    if features[tree.op-1]==1:
        return classify(tree.kids[0], features, depth+1)
    else:
        return classify(tree.kids[1], features, depth+1)

# testTrees(T,x2) from the specs, but named test_trees for style
#Tests set of trained trees T on x2
def test_trees(T, x2, random=0):
    L=[]
    for index in range(0,x2.shape[0]):
        depths_pos = []
        out_pos = []
        depths_neg = []
        out_neg = []
        for t_num in range(0,T.size):
            test = classify(T[t_num],x2[index])

            if test[0]==1:
                depths_pos.append(test[1])
                out_pos.append(t_num+1)
            else:
                depths_neg.append(test[1])
                out_neg.append(t_num+1)
        if random:
            import random
            if len(out_pos)==0:
                output=random.randint(1, 6)
            else:
                output=out_pos[random.randint(0,len(out_pos)-1)]
        else:
            if len(out_pos)==0:
                output = out_neg[depths_neg.index(max(depths_neg))]
            else:
                output = out_pos[depths_pos.index(min(depths_pos))]
        L.append(output)
    return L


if __name__=='main':
    #Change TEST DATA to filename of the test data
    x, y = load_data(TEST DATA)

    T=[]
    for i in range(1,7):
        T.append(load_from_mat('tree_'+label_to_string(i)+'.mat'))
    np.array(T)

    #returns Python list of prediction labels
    Results = test_trees(T,x)
