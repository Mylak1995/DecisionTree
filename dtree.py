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
    for x in range(0,classifier.size):
        if classifier[x][0] == number:
            list.append(1)
        else:
            list.append(0)
    return np.asarray(list)

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
    for x in range (0,attributes.size):
        value = attribute_calculation(examples,x,binary_targets)
        if max<value:
            max = value
            index = x
    return attributes[index]

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
    if p+n==0: return 0
    a=p/(p+n)
    b=n/(p+n)
    log2 = lambda x: log(x)/log(2)
    if a==0 or b==0:
        return 0
    return (-a*log2(a)-b*log2(b))


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
        return tree(best_attribute,[decision_tree_learning(ex1,attributes,bt1),decision_tree_learning(ex0,attributes,bt0)])


def getwidth(tree):
    if tree.kids==[]: return 1
    return getwidth(tree.kids[0])+getwidth(tree.kids[1])

def getdepth(tree):
    if tree.kids==[0]: return 0
    return max(getdepth(tree.kids[0]),getdepth(tree.kids[1]))+1

def to_newick(tree):
    if tree.op==None:
        if tree.classification==1:
            return "yes"
        else:
            return "no"
    else:
        newick="AU"+str(tree.op)
        return "("+to_newick(tree.kids[0])+","+to_newick(tree.kids[1])+")"+newick

attributes=np.asarray(list(range(1,46)))
examples=data1
bin_targets=to_binary_classifier(classification1,6)
trained = decision_tree_learning(examples,attributes,bin_targets)
print(trained.op)
#new=to_newick(trained)+";"
#print(new)
#from ete3 import Tree as Tr
#from ete3 import TreeStyle, Tree, TextFace, add_face_to_node
#t = Tr(new,format=8)

#ts = TreeStyle()
#ts.rotation = 90
#ts.show_leaf_name = False
#def my_layout(node):
#        F = TextFace(node.name, tight_text=True)
#        add_face_to_node(F, node, column=0, position="branch-right")
#ts.layout_fn = my_layout
#t.show(tree_style=ts)




import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def treeWidth(node):
    if node.isLeaf():
        return 1
    return treeWidth(node.trueBranch) + treeWidth(node.falseBranch)

def treeHeight(node):
    if not node:
        return 0
    return max(treeHeight(node.trueBranch), treeHeight(node.falseBranch)) + 1

def drawNode(draw, node, x, y):
    if node.isLeaf():
        draw.text((x-20,y), str(node.result), (0,0,0))
    else:
        # false is the left branch
        wt = treeWidth(node.trueBranch) * 100
        wf = treeWidth(node.falseBranch) * 100
        left = x - (wt + wf)/2
        right = x + (wt + wf)/2
        draw.text((x-20,y-10), str(node.testedAttribute), (0,0,0))
        draw.line((x,y,left+wf/2,y+100), fill=(255,0,0))
        draw.line((x,y,right-wt/2,y+100), fill=(255,0,0))
        drawNode(draw, node.falseBranch, left+wf/2, y+100)
        drawNode(draw, node.trueBranch, right-wt/2, y+100)

def visualizeTree(rootNode):
    w = treeWidth(rootNode) * 100
    h = treeHeight(rootNode) * 100 + 50
    print(w, h)
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)
    drawNode(draw, rootNode, w/2, 50)
    img.save('tree.jpg','JPEG')
    plt.imshow(img)
    plt.show()

    # Tree Node class
class TreeNode:
       def __init__(self, testedAttribute=None, result=None):
           self.testedAttribute = testedAttribute  # attribute that the node is testing
           self.trueBranch = None
           self.falseBranch = None
           self.result = result  # tree.class in the manual, 0 or 1
        def isLeaf(self):
           return not (self.trueBranch or self.falseBranch)