import numpy as np
import scipy.io as sio

my_data1=sio.loadmat('cleandata_students.mat')

data1=my_data1['x']
classification1=my_data1['y']

my_data2=sio.loadmat('noisydata_students.mat')

data2=my_data2['x']
classification2=my_data2['y']


class tree:
    def __init__(self,op=None,kids=[],classification=None):
        self.op=op
        self.kids=kids
        self.classification=classification

    def unique(bin_targets):
        if len(bin_targets)==np.sum(bin_targets) or sum(bin_targets)==0:
            return True
        return False

    def majority_value(bin_targets):
        if np.sum(bin_targets)>(len(bin_targets)/2):
            return 1
        return 0


    def choose_best_decision_attribute(examples,attributes,binary_targets):
        #for i in attributes:
            
        return true

    def entropy(p,n):
        from math import log
        a=float(p)/(p+n)
        b=float(n)/(p+n)
        log2 = lambda x: log(x)/log(2)
        return (-a*log2(a)-b*log2(b))

    #input of p and n?
    #define p0,n0,p1,n1
    def remainder(attributes):
        #something p0,p1 =
        a=float(p0+n0)/(p+n)
        b=float(p1+n1)/(p+n)
        return a*entropy(p0,n0)+b*entropy(p1,n1)

    def gain(attributes,p,n):
        return entropy(p,n)-remainder(attributes)

    def decision_tree_learning(examples,attributes,bin_targets):
        if is_unique(bin_targets):
            return tree(classification=bin_targets[0])
        elif examples.size==0:
            return tree(classification=majority_value(bin_targets))
        else:
            return True


    def to_binary_classifier(classifier,number):
        list=[]
        for x in range(0,len(classifier)):
            if classifier[x][0] == number:
                list.append(1)
            else:
                list.append(0)
        return list
