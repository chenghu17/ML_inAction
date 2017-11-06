# author:hucheng

from numpy import *

class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(filename):
    dataset = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #在linear regression中也能使用这种方式将每行的值转换为float
        #而不需要进行for循环
        fltLine = map(float,curLine)
        dataset.append(fltLine)
    return dataset