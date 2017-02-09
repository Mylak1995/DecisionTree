import scipy.io as sio
import numpy as np

my_data1=sio.loadmat('cleandata_students.mat')
data1=my_data1['x']
classification1=my_data1['y']

my_data2=sio.loadmat('noisydata_students.mat')
data2=my_data2['x']
classification2=my_data2['y']


class tree:
    def __init__(self,op=None,kids=[],classification=None, pos=None, neg=None):
        self.op=op
        self.kids=kids
        self.classification=classification
        self.pos=pos
        self.neg=neg
    def pruneTree(self,validation_x,binary_classifier):
        L=findAllParents(self)
        non_changed=0
        for node in L:
            initial_recall=calculateRecall(self,validation_x,binary_classifier)

            temp=node.kids
            tempop=node.op
            node.op=None
            node.kids=[]

            if node.pos>node.neg:
                node.classification=1
            else:
                node.classification=0
            end_recall=calculateRecall(self,validation_x,binary_classifier)

            if end_recall<initial_recall:
                node.classification=None
                node.kids=temp
                node.op=tempop
                non_changed+=1
        print(non_changed,len(L))
        if non_changed==len(L):
            return False
        return self.pruneTree(validation_x,binary_classifier)


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
    if p+n==0: return 0
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

    e1 = entropy(p,n)
    e2 = entropy(p0,n0)
    e3 = entropy(p1,n1)

    remainder = (p0+n0)*e2/(p+n)+(p1+n1)*e3/(p+n)
    return e1 - remainder

#Choosing maximum IG
def choose_best_decision_attribute(examples,attributes,binary_targets,threshold):
    index = 0
    max = attribute_calculation(examples,index,binary_targets)
    for x in range (0,attributes.size):
        value = attribute_calculation(examples,x,binary_targets)
        if max < value:
            max = value
            index = x
    if max < threshold:
        return 0
    return attributes[index]

#Learning

def decision_tree_learning(examples,attributes,bin_targets,threshold):
    if examples.size==0 or attributes.size==0:
        return tree(classification=majority_value(bin_targets),pos=np.sum(bin_targets),neg=bin_targets.size-np.sum(bin_targets))
    elif is_unique(bin_targets):
        return tree(classification=bin_targets[0],pos=np.sum(bin_targets),neg=bin_targets.size-np.sum(bin_targets))
    else:
        best_attribute=choose_best_decision_attribute(examples,attributes,bin_targets,threshold)
        if best_attribute==0:
            return tree(classification=majority_value(bin_targets),pos=np.sum(bin_targets),neg=bin_targets.size-np.sum(bin_targets))
        index=np.where(attributes==best_attribute)
        index=index[0][0]
        attributes=np.delete(attributes,index)

        ex1=examples[examples[:,index]==1]
        ex0=examples[examples[:,index]==0]

        bt1=bin_targets[examples[:,index]==1]
        bt0=bin_targets[examples[:,index]==0]

        ex1=np.delete(ex1,index,axis=1)
        ex0=np.delete(ex0,index,axis=1)

        t1=decision_tree_learning(ex1, attributes, bt1, threshold)

        t0=decision_tree_learning(ex0, attributes, bt0, threshold)

        return tree(best_attribute,[t1,t0],pos=np.sum(bin_targets),neg=bin_targets.size-np.sum(bin_targets))

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


#Counting leafs

def isLeaf(tree):
    if tree.classification!=None:
        return True
    return False

def countLeaves(tree):
    if isLeaf(tree):
        return 1
    else:
        return countLeaves(tree.kids[0])+countLeaves(tree.kids[1])

'''
attributes=np.arange(1,46)
examples=data1
bin_targets=to_binary_classifier(classification1,5)
trained = decision_tree_learning(examples,attributes,bin_targets,0)

new=to_newick(trained)+";"

from ete3 import Tree as Tr
from ete3 import TreeStyle, Tree, TextFace, add_face_to_node
t = Tr(new,format=8)

ts = TreeStyle()
ts.rotation = 90
ts.show_leaf_name = False
def my_layout(node):
        F = TextFace(node.name, tight_text=True)
        add_face_to_node(F, node, column=0, position="branch-right")
ts.layout_fn = my_layout
t.show(tree_style=ts)
'''

#Tests one tree, returns depth of classification and classification
def test_one_tree(tree,features,depth):
    if tree.classification!=None:
        return [tree.classification,depth+1]
    if features[tree.op-1]==1:
        return test_one_tree(tree.kids[0],features,depth+1)
    else:
        return test_one_tree(tree.kids[1],features,depth+1)

#Tests set of trained trees T on x2
def testTrees(T,x2):
    L=[]
    for index in range(0,x2.shape[0]):
        output = 7
        depth = 100
        for t_num in range(0,T.size):
            test=test_one_tree(T[t_num],x2[index],0)
            if test[0]==1 and test[1]<depth:
                output = t_num+1
        #in case none of the trees predict the thing - randomize
        #import random
        #if output==7:
        #    output=random.randint(1, 6)

        L.append(output)
    return L

#Creating confusion matrix

def confusion_matrix_10_cross(data,classification,threshold):
    ''' Creating slices '''
    x_slices = np.vsplit(data[:900, :], 9)
    x_l_slice = data[900:, :]
    x_slices.append(x_l_slice)

    y_slices = np.vsplit(classification[:900], 9)
    y_l_slice = classification[900:]
    y_slices.append(y_l_slice)

    confusion_matrix = np.zeros((6,7), dtype=np.int)

    ''' i = ith cross-validation run '''
    for i in range(0,10):
        train_x = np.empty((0, 45), int)
        train_y = np.empty((0,1), int)

        for j in range (0,10):
            if j==i:
                test_x = x_slices[j]
                test_y = y_slices[j]
            else:
                train_x = np.concatenate((train_x, x_slices[j]), axis=0)
                train_y = np.concatenate((train_y, y_slices[j]), axis=0)

        attributes = np.arange(1, 46)

        ''' Training on train data '''
        L=[]

        for i in range(1,7):
            L.append(decision_tree_learning(train_x,attributes,to_binary_classifier(train_y,i),threshold))

        T = np.array(L)

        results = testTrees(T,test_x)

        for i in range(0, test_y.size):
            confusion_matrix[test_y[i] - 1, results[i] - 1]+=1

    return confusion_matrix
'''
cm=confusion_matrix_10_cross(data1,classification1,0)
print(cm)
'''

'''
for i in range(0,16):
    print(i*0.005)
    cm=confusion_matrix_10_cross(data1,classification1,i*0.01)
    print((cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5])/np.sum(cm))
'''






#Finds all nodes that are parents to two leaves
def findAllParents(tree):
    L=[]
    if isLeaf(tree):
        return L
    elif isLeavesParent(tree):
        L.append(tree)
        return L
    else:
        K=findAllParents(tree.kids[0])
        M=findAllParents(tree.kids[1])
        L.extend(K)
        L.extend(M)
        return L

#Checks if a node is a parent to two leaves
def isLeavesParent(tree):
    if isLeaf(tree.kids[0]) and isLeaf(tree.kids[1]):
        return True
    return False


'''
train_x = data1[:700,:]
validation_x = data1[700:900,:]
test_x = data1[900:,:]

train_y = classification1[:700,:]
validation_y = classification1[700:900,:]
test_y = classification1[900:,:]


L=[]
for i in range(1,7):
    L.append(decision_tree_learning(train_x,attributes,to_binary_classifier(train_y,i),0))

L[i-1].pruneTree(validation_x,to_binary_classifier(train_y,i))
'''
#Calculates recall on single tree
def calculateRecall(tree,data,labels):
    counter=0
    for index in range(0,data.shape[0]):
        if test_one_tree(tree,data[index],0)[0]==labels[index]:
            counter+=1
    return counter/data.shape[0]
