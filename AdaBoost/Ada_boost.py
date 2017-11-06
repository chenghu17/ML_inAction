# author:hucheng
# ensemble method : bagging,random forest,boosting
# this part is about AdaBoost
# they use the same classifier in either bagging or boosting
# boosting通过集中关注被已有分类器错分的数据，通过这些数据来训练新的classifier，所以它是基于错误的classifier
# AdaBoosting（adaptive boosting）和SVM一样，预测两个类别中的一个。如果需要应用到多个类别中，需要进行修改
from numpy import *

#初始化数据集
#得到数据集后，最重要的是要将它们可视化！
#可视化是数据挖掘前期很有必要而且有效的步骤
def loadSimpData():
    dataMat = matrix([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1.,1.],
                     [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#是否存在某个小于或大于正在测试的阈值
#通过阈值对数据进行分类，在阈值一侧的数据归类为-1，另一侧的数据归类为+1
def stumpClassift(dataMatrix,dimen,threshVal,threshIneq):
    #和数据集拥有相同行数的列向量，初始值都为1
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#基于加权输入值进行决策的分类器
#在加权数据集中进行循环，并找到具有最低错误率的单层决策树
#此处的单层决策树 是指 它指依赖单个特征来做决策，因此这棵树只有一次分裂
#D为数据的权重向量

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    #对每列（所有特征）进行遍历
    for i in range(n):
        #计算第i+1行的最大／最小值，用于确定步长的大小
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        #步长，此处为范围的1/10
        stepSize = (rangeMax-rangeMin)/numSteps
        #对每一步进行遍历
        for j in range(-1,int(numSteps)+1):

            for inequal in ['lt','gt']:
                #threshVal为阈值，该值会从rangeMin---->rangeMax
                #在增长的过程中，
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassift(dataMatrix,i,threshVal,inequal)
                #初始化误差，每一项为1
                errArr = mat(ones((m,1)))
                #对于预测值与实际值相同的项，其误差值赋值为0
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print("split:dim %d, thresh %.2f,thresh ineqal: %s,the "
                      "weighted error is %.3f" %(i,threshVal,inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = 1
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m =shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print('D:',D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggClassEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print('total error:',errorRate,'\n')
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataMatrix =  mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassift(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


if __name__=='__main__':
    data,label = loadSimpData()
    weakClassArr = adaBoostTrainDS(data,label)
    print(adaClassify([[5,5],[0,0]],weakClassArr))