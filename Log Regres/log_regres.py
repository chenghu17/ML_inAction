# author:hucheng
from numpy import *
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #1.0表示线性回归方差中的常数项，先初识化为1.0，后续再对所有参数进行调节
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1/(1+exp(-inX))

#梯度上升
def gradAscent(dataMatIn,classLabels):
    #将数据集转换为numpy矩阵
    dataMatrix = mat(dataMatIn)
    #将label转换为numpy矩阵，并且转置为列向量
    labelMat = mat(classLabels).transpose()
    #m为样本个数，n为特征个数
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    #将参数矩阵设置为weights，默认值为1，n为样本数据的特征数
    weights = ones((n,1))
    for k in range(maxCycles):
        #得到每个样本映射到(0,1)区间的估计值
        #由于样本的真实值为0／1，所以在对样本进行预测时，要将其原本的结果值映射到(0，1)区间中，这样才有可比较性
        #h即为样本的预测值
        h = sigmoid(dataMatrix*weights)
        #error为样本真实值与预测值的差
        error = (labelMat-h)
        #为什么这里用实际值与预测值的差值 * 原数据集 = 梯度？
        #需要继续考虑
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.001
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights)
        #以当前值作为误差值，对参数进行变动
        error = classLabels[i]-h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

#改进的随笔梯度上升
def stocGradAscent1(dataMatIn,classLabels,numIter=150):
    dataMatrix = array(dataMatIn)
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0,len(dataIndex)))
            theChonsenOne = dataIndex[randIndex]
            h = sigmoid((sum(dataMatrix[theChonsenOne]*weights)))
            error = classLabels[theChonsenOne]-h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights,dataMat, labelMat):
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #arange为numpy中的一个方法，其中前两个参数为输出点对的范围，包前不包后
    #第三个参数为诶步长，即每隔多远取一个点，最终返回一个array对象
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



if __name__=='__main__':
    dataMat, labelMat = loadDataSet()
    #weights = gradAscent(dataMat, labelMat)    #梯度下降
    #weights = stocGradAscent0(dataMat, labelMat)    #随机梯度下降
    weights = stocGradAscent1(dataMat, labelMat)    #改进随机梯度下降
    #getA()是将weights矩阵以数组的形式进行返回
    #plotBestFit(weights.getA(),dataMat,labelMat)    #梯度下降
    plotBestFit(weights,dataMat,labelMat)    #随机梯度下降


