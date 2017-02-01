import scipy.io as sio

my_data1=sio.loadmat('cleandata_students.mat')

data1=my_data1['x']
classification1=my_data1['y']

my_data2=sio.loadmat('noisydata_students.mat')

data2=my_data2['x']
classification2=my_data2['y']


class tree:
    def __init__(self,op,kids=[],classification=None):
        self.op=op
        self.kids=kids
        self.classification=classification


def to_binary_classifier(classifier,number):
    list=[]
    for x in range(0,len(classifier)):
        if classifier[x][0] == number:
            list.append(1)
        else:
            list.append(0)
    return list

print(to_binary_classifier(classification1,1))
print(classification1)









