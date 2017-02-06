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
    bc=np.array([])
    for x in range(0,classifier.size):
        if classifier[x] == number:
            bc=np.append(bc,1)
        else:
            bc = np.append(bc, 0)
    return bc.astype(int)

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

#Calculating entropy
def entropy(p,n):
    from math import log
    a=p/(p+n)
    b=n/(p+n)
    log2 = lambda x: log(x)/log(2)
    if a==0 or b==0:
        return 0
    return (-a*log2(a)-b*log2(b))


#Performing IG calculation for given attribute
def attribute_calculation(examples,index,binary_targets):
    p0, n0, p1, n1 = 0, 0, 0, 0
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
    print(n1, p1, n0, p0, n, p)

    if p+n==0:
        e1 = 0
    else:
        e1 = entropy(p,n)

    if p0+n0==0:
        e2 = 0
    else:
        e2 = entropy(p0,n0)

    if p1+n1==0:
        e3 = 0
    else:
        e3 = entropy(p1,n1)

    remainder = (p0+n0)*e2/(p+n)+(p1+n1)*e3/(p+n)
    return e1 - remainder

#Choosing maximum IG
def choose_best_decision_attribute(examples,attributes,binary_targets):
    index=0
    max = attribute_calculation(examples,index,binary_targets)
    for x in range (0,attributes.size):
        value = attribute_calculation(examples,x,binary_targets)
        if max < value:
            max=value
            index = x
    return attributes[index]

#Learning

def decision_tree_learning(examples,attributes,bin_targets):
    if examples.size==0 or attributes.size==0:
        return tree(classification=majority_value(bin_targets))
    elif is_unique(bin_targets):
        return tree(classification=bin_targets[0])
    else:
        best_attribute=choose_best_decision_attribute(examples,attributes,bin_targets)
        index=np.where(attributes==best_attribute)
        index=index[0][0]
        attributes=np.delete(attributes,index)

        ex1=examples[examples[:,index]==1]
        ex0=examples[examples[:,index]==0]

        bt1=bin_targets[examples[:,index]==1]
        bt0=bin_targets[examples[:,index]==0]

        ex1=np.delete(ex1,index,axis=1)
        ex0=np.delete(ex0,index,axis=1)

        t1=decision_tree_learning(ex1, attributes, bt1)

        t0=decision_tree_learning(ex0, attributes, bt0)

        return tree(best_attribute,[t1,t0])

#Writing tree to newick format

def to_newick(tree):
    if tree.op==None:
        if tree.classification==1:
            return "yes"
        else:
            return "no"
    else:
        newick="AU"+str(tree.op)
        return "("+to_newick(tree.kids[0])+","+to_newick(tree.kids[1])+")"+newick


decision_tree_learning(data1,np.arange(1,46),to_binary_classifier(classification1,1))

