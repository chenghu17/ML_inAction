#基于概率论的分类方法：朴素贝叶斯
from numpy import *
#数据集，label为人工标记
def loadDataSet():
    postingList = [['my','dog','has','flea','problem','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#从文档中收集并创建单词列表
def createVocabList(dataset):
    vocabSet = set([])
    for doucument in dataset:
        vocabSet = vocabSet | set(doucument)
    return list(vocabSet)
#将目标文档转换为词向量的形式
def setOfwords2Vec(vocabList,inputList):
    returnVec = [0]*len(vocabList)
    for word in inputList:
        if word in vocabList:
            #returnVec[vocabList.index(word)] = 1
            #此处为词袋模型(bag-of-words model)
            #原式只是记录词是否出现过，但是多次出现所内藏的含义将不能用是否出现替代
            #所以此处更新为每次出现，即增加一次
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word %s is not in vocabList' % word)
    return returnVec

def trainNBO(trainMatrix,trainCatagory):
    #文档的个数
    numTrainDocs = len(trainMatrix)
    #词向量的维度
    numWords = len(trainMatrix[0])
    #侮辱性文档（class=1）所占比例
    pAbusive = sum(trainCatagory)/float(numTrainDocs)
    #class=1/class=0的词向量初始化,为避免一个不出现为0，在后续的多值概率乘积时影响到其他词的概率，初始值设为1
    #class=0/class=1的总词量（含重复）
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    #计算不同类别文档中单词的出现情况，集中到各类统一的词向量中,对于重复单词，其对应的值从1-->2-->...
    #计算不同类别文档中单词的总量，记录到各类对应的变量中
    p0Denom = 2.0;p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCatagory[i] == 1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #计算每个词在该类文档 所有词中出现的频率
    #p1Vect = p1Num/p1Denom
    p1Vect = log(p1Num/p1Denom)
    #p0Vect = p0Num/p0Denom
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pclass1):
    #注意此处的乘法，为对应位置的数相乘，然后使用sum方法对各个数字进行求和
    #并加上对应类别概率取对数值
    #由于在trainNBO中对每个词出现的频率取来对数，所以在次对类别概率取对数然后相加
    #其本质其实是将测试数据中词出现的频率乘该词在训练集中出现的频率乘该类别出现的频率，其积取对数
    p1 = sum(vec2Classify*p1Vec)+log(pclass1)
    p0 = sum(vec2Classify*p0Vec)+log(1-pclass1)
    if p1>p0:
        return 1
    else:
        return 0

def testNB():
    postingList, classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    # 训练数据对应的词向量矩阵
    trainMat = []
    for posting in postingList:
        trainMat.append(setOfwords2Vec(vocabList, posting))
    p0V, p1V, pAb = trainNBO(trainMat, classVec)
    testEntry = ['love','my','dalmation']
    #将测试实例转换为词向量,setOfword2Vec返回值为list属性，所以需要array来进行类型转换
    thisDOC = array(setOfwords2Vec(vocabList,testEntry))
    print("doc:",testEntry,"is classify:",classifyNB(thisDOC,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDOC = array(setOfwords2Vec(vocabList,testEntry))
    print("doc:",testEntry,"is classify:",classifyNB(thisDOC,p0V,p1V,pAb))

if __name__=='__main__':

    testNB()

