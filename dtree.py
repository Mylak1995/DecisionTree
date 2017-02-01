import scipy.io as sio
import numpy as np

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

#Transformation of classifier to binary classifier for given emotion number
def to_binary_classifier(classifier,number):
    list=[]
    for x in range(0,len(classifier)):
        if classifier[x][0] == number:
            list.append(1)
        else:
            list.append(0)
    return list

#Checking if binary targets contain of one value (point uniquely to answer)
def is_unique(bin_targets):
    if bin_targets.size==np.sum(bin_targets) or np.sum(bin_targets)==0:
        return True
    return False

#Returning majority value of binary targets
def majority_value(bin_targets):
    if np.sum(bin_targets)>(bin_targets.size/2):
        return 1
    return 0


def choose_best_decision_attribute(examples,attributes,binary_targets):
    max = attribute_calculation(examples,0,binary_targets)
    index = 0
    for x in range (1,attributes.size):
        value = attribute_calculation(examples,x,binary_targets)
        if max<value:
            max = value
            index = x
    return attributes[x]

def attribute_calculation(examples,index,binary_targets):
    p0, n0, p1, n1=0, 0, 0, 0
    for x in range(0,binary_targets.size):
        if binary_targets[x]==0:
            if examples[x][index]==1:
                n1 += 1
            else:
                n0 += 1
        else:
            if examples[x][index] == 1:
                p1 += 1
            else:
                p0 += 1

    p = np.sum(binary_targets)
    n = binary_targets.size-p
    e1 = entropy(p,n)
    e2 = entropy(p0,n0)
    e3 = entropy(p1,n1)
    remainder = (p0+n0)*e2/(p+n)+(p1+n1)*e3/(p+n)
    return e1 - remainder

#Calculating entropy
def entropy(p,n):
    from math import log
    a=float(p)/(p+n)
    b=float(n)/(p+n)
    log2 = lambda x: log(x)/log(2)
    return (-a*log2(a)-b*log2(b))


def decision_tree_learning(examples,attributes,bin_targets):
    if is_unique(bin_targets):
        return tree(classification=bin_targets[0])
    elif examples.size==0 or attributes.size==0:
        return tree(classification=majority_value(bin_targets))
    else:
        best_attribute=choose_best_decision_attribute(examples,attributes,bin_targets)
        index=np.argwhere(attributes==best_attribute)
        attributes=np.delete(attributes,index)
        examples=np.delete(examples,index,axis=1)
        ex1=examples[examples[:,index]==1]
        ex0=examples[examples[:,index]==0]
        bt1=bin_targets[examples[:,index]==1]
        bt0=bin_targets[examples[:,index]==0]
        return tree(best_attribute,[decision_tree_learning(ex1,attributes,bt1),decision_tree_learning(ex0,attributes,bt0)])







