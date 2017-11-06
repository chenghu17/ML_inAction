#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
import random
import operator


#收集数据
#处理数据
#分析数据
#训练算法
#测试数据
#使用算法

#对数据集进行处理，使用矩阵表示数据属性、使用列表表示标签
#其中数据最后一列为int型，即label列
#实验数据来自于网上的150组数据集
#三种不同的鸢尾属植物setosa(1)、versicolor(2)和virginica(3)的花朵样本
#属性：萼片长度(sepal length)、萼片宽度sepalwidth)、花瓣长度(petal length)和花瓣宽度(petal width)
#注意python中file读取的 read()、readline()、readlines()的区别

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLine = len(arrayOfLines)
    returnMat = zeros((numberOfLine,4))
    classLableVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        # 此处的划分依据根据数据集中的实际表示来定
        listFromLine = line.split(',')
        returnMat[index,:]=listFromLine[0:4]
        classLableVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLableVector

#归一化数值
#由于每个属性是等权的，但是由于属性本身的数值差异会影响到数据的结果
#所以使用归一化数值法，将所有列的数值映射到(0,1)区间
#采用的方法是选择属性的最小值与最大值，计算其差值A
#计算具体属性与最小属性的差B，使用B/A的商，作为该属性值的映射值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    rangeVals = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(rangeVals,(m,1))
    return normDataSet,rangeVals,minVals

#对目标对象进行分类，同样采用kNN_base.py中的比较方法
#即对矩阵进行减法操作，然后求每行的平方的和，之后再开平平方。即最小二乘法的推广
#对得到的结果以从小到大的顺序进行排序，注：此处的排序是根据值的大小对index进行排序并返回相应的index list
#对前k个结果所对应的label进行降序归类(投票)，归类的(voteLabel,counts)，返回[0][0]即count最大的votelabel
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#validation
def datingClassTest():
    hoRatio = 0.1
    #从文件中读取数据并进行处理
    returnMat, classLabels = file2matrix('dataset.txt')
    #对数据进行数值归一化
    normDataSet, rangeVals, minVals = autoNorm(returnMat)
    m = normDataSet.shape[0]
    #print('m=',m)
    numTestVecs = int(m*hoRatio)
    #print('numTestVecs= %d' %numTestVecs)
    errorCount = 0.0
    #从m行中随机取numTestVecs个数，即随机取出测试数据集的行数
    testVecsIndex = random.sample([x for x in range(m)],numTestVecs)
    #此处有一个大坑！！！
    #list存在sorted()和sort()两个排序方法，而sort()对list进行排序后，实际是改变了list中数据的存储位置
    #没有进行备份，而且该方法返回值为none，但是原list已经被排序了！！！！
    testVecsIndex.sort()
    #训练数据，即已知类型数据
    #normDataSet_train = []
    #classLabels_test = []
    #针对于python自带的list对象，可以使用append和extend
    #但是对于numpy创建的array对象，不存在这两个方法，得使用其自带的方法，
    #对array进行merge操作，此处使用了vstack方法
    #code中normDataSet_train为从dataset中选出的testdataset
    #而normDataSet则为整个dataset，留出来的数据集做验证操作，此处称为留出法
    #在这个实验中因为样本的排序是存在规则的，所以validation set是随机选取的，testVecsIndex即为验证集的index set
    for i in range(numTestVecs):
        if i != 0:
            normDataSet_train = vstack((normDataSet_train,normDataSet[testVecsIndex[i-1]+1:testVecsIndex[i],:]))
            classLabels_test.extend(classLabels[testVecsIndex[i-1]+1:testVecsIndex[i]])
        else:
            normDataSet_train = normDataSet[0:testVecsIndex[i],:]
            classLabels_test = classLabels[0:testVecsIndex[i]]
    for i in range(numTestVecs):

        classifierResult = classify0(normDataSet[testVecsIndex[i],:],
                                     normDataSet_train, classLabels, 10)
        print("the classifier came back with %d,the real answer is: %d" %(classifierResult,classLabels[testVecsIndex[i]]))
        if classifierResult != classLabels[testVecsIndex[i]]:
            errorCount+=1
    print("the error rate is %f" % (errorCount/numTestVecs))

def realDataPredict():
    resultList = ['setosa','versicolor','virginica']
    #萼片长度(sepallength)、萼片宽度(sepalwidth)、花瓣长度(petallength)和花瓣宽度(petalwidth)
    sepal_length = float(input("萼片长度(sepallength) = "))
    sepal_width = float(input("萼片宽度(sepalwidth) = "))
    petal_length = float(input("花瓣长度(petallength) = "))
    petal_width = float(input("花瓣宽度(petalwidth) = "))
    objectVec = array([sepal_length,sepal_width,petal_length,petal_width])
    returnMat, classLabels = file2matrix('dataset.txt')
    normDataSet, rangeVals, minVals = autoNorm(returnMat)
    classifierResult = classify0((objectVec-minVals)/rangeVals,normDataSet,classLabels,10)
    print('this plant belongs to %s' %(resultList[classifierResult-1]))


if __name__=='__main__':
    #returnmat,classlabel = file2matrix('dataset.txt')
    #normDataSet, rangeVals, minVals = autoNorm(returnmat)
    #print(normDataSet)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(returnmat[:,0],returnmat[:,0],
    #           10.0*array(classlabel),10.0*array(classlabel))
    #plt.show()

    datingClassTest()
    realDataPredict()

