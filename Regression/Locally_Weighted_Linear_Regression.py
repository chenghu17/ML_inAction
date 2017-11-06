# author:hucheng

from numpy import *
import time
import matplotlib.pyplot as plt

def loadDataSet():
    #数据集最后一列为目标值
    numFeat = len(open('data.txt').readline().split())-1
    dataArr = []
    labelArr = []
    fr = open('data.txt')
    for line in fr.readlines():
        #获取当前行，转换为列表
        curLine = line.strip().split()
        #将列表除最后一列外，加入到dataMat中
        #将列表最后一列加入到labelMat中
        #原《机器学习实战》课本中使用的方法重复，更改后如下
        #更改后发现时我理解错来，因为现在读出来的特征值为str，需要转换为float
        lineArr = []
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))
    return dataArr,labelArr

def stanRegres(xArr,yArr):
    #xArr即为输入数据列表，yArr即为输出数据列表
    xMat = mat(xArr)
    #注意这里需要转置，因为yMat为行向量，需要转换为列向量才能进行矩阵内积
    yMat = mat(yArr).T
    #xMat的转置乘上xMat
    xTx = xMat.T * xMat
    #根据公式，需要判断xTx的逆是否存在
    #这里使用矩阵所对应的行列式是否为0来进行判断
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

#局部加权线形回归预测时，需要先选出对应的数据子集
#而计算得到的参数公式中包含权重矩阵
#权重举证的获取方法使用的是类似于支持向量机中的kernel method
#一般使用的核为 高斯核
def lwlr(testPoint,xArr,yArr,k=0.1):
    xMat = mat(xArr)

    yMat = mat(yArr).T
    m = shape(xMat)[0]
    #权重为对角矩阵
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint-xMat[j,:]
        #高斯核,对每一行的数据，与测试数据计算高斯期望，期望值即为权重值
        #构成对角权重矩阵
        weights[j,j]= exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*weights*xMat
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular,cannot do inverse")
        return
    #参数计算公式
    ws = xTx.I * (xMat.T * weights *yMat)
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        #对每一组测试数据，计算局部加权回归后的预测值
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

if __name__ == '__main__':
    #time1 = time.time()
    dataset,labelset = loadDataSet()
    #time2 = time.time()
    xMat = mat(dataset)
    yMat = mat(labelset)
    #对于预测的回归曲线如何绘制，就是计算所有的预测值
    #将提供的x，与预测值一一对应，即可绘制出来！！！
    yHat = lwlrTest(dataset,dataset,labelset,0.28)
    #对数据进行排序，所有的绘图函数都需要将数据点按序排列
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T.flatten().A[0],s=2,c='red')
    plt.show()
