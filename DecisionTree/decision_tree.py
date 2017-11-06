# coding: utf-8
from math import log
import operator

#创建数据集和标签
def creatDataSet():
    dataSet = [ [1,1,'yes'],
                [1,1,'yes'],
                [1,0,'no'],
                [0,1,'no'],
                [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#计算信息熵
#首先对测试数据集中的类别进行划分，并分类存储其出现次数
def caluShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
#计算每个label出现的概率，使用信息熵公式进行计算
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

#按照给定的特征及其特征值对数据集进行划分，返回满足条件的数据集
#此处只能返回包含指定特征值特征的数据集
#所以此方法会被多次调用，即对于相同的axis，value不同
#取出划分后的多个数据子集并做后续的信息熵计算
def splitDataSet(dataset,axis,value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最划分特征
#通过计算各个信息熵，选择出最好的特特征划分方式
#首先获取数据集的特征种类
#计算数据集的信息熵
#初始化最好信息增益和最有特征
#循环遍历数据集中的每个特征
#在对每个特征进行遍历的过程中，获取单个特征的取值集合
#对单个特征值的取值集合进行遍历，即进行数据集划分，取出相应的子集，计算其信息熵和概率的乘积
#得到划分后的信息熵，计算信息增益
#判断当前信息增益与原信息增益的大小，取最大值，并将此时的特征作为最优划分特征
#直到遍历完整个特征集，返回最优划分
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0])-1
    baseShannonEnt = caluShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDateSet = splitDataSet(dataset,i,value)
            prob = len(subDateSet)/float(len(dataset))
            newEntropy+=prob*caluShannonEnt(subDateSet)
        infoGain = baseShannonEnt-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#如果递归到最后一类，还存在多个结果
#则对该类的结果集进行排序，选出出现次数最多的项作为最终的结果
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classList.key():
            classCount[vote] = 0
        classCount+=1
    sortedClassCount = sorted(classCount.item(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#生成决策树
#首先计算数据集中的类别，如果结果集属于同一类，则直接返回该类别，即叶子节点
#若数据集中指含有一类，即只有一列，此时选择数据中出现此处最多的象作为结果，即叶子节点
#使用选择最优划分特征方法，得到最优特征并获取该特征的label，将该lable放入myTree元组中国
#获取该特征所有的取值情况，将该特征不同取值下的不同子集进行递归的构建决策树，即 将最优特征的分支数据集重复上述操作
#最终生成决策树myTree
#其中需要主要的是，如何使得递归结束，即如何将节点归为叶子节点！！！

def creatTree(dataset,lables):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataset)
    bestFeatureLable = lables[bestFeature]
    myTree = {bestFeatureLable: {}}
    del(lables[bestFeature])
    featValue = [example[bestFeature] for example in dataset]
    values = set(featValue)
    for value in values:
        sublables = lables[:]
        myTree[bestFeatureLable][value]=creatTree(splitDataSet(dataset,bestFeature,value),sublables)
    return myTree

if __name__ == '__main__':
    myDat,labels= creatDataSet()
    print(creatTree(myDat,labels))
