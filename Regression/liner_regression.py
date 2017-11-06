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


if __name__ == '__main__':
    time1 = time.time()
    dataset,labelset = loadDataSet()
    w = stanRegres(dataset,labelset)
    print(w)
    time2 = time.time()
    xMat = mat(dataset)
    yMat = mat(labelset)
    yHat = xMat*w
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #绘制原始点
    ax.scatter(xMat[:,1].flatten().A[0],
               yMat.T[:,0].flatten().A[0])
    #对原始点进行排序，绘制最佳拟合直线
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * w
    ax.plot(xCopy[:,1],yHat)
    plt.show()

#此处每组数据只能是两位，一个偏移量，一个实际的输入x
#因为如果存在多个数据，就存在两种解释：
#1、如果绘制在平面内，则模型由多项式构成，含有x的多次幂
#2、如果模型不是有多项式构成，规定都为一次，那么为多元方程，不能给在二维平面中进行绘制
#如果又不是多项式，又不是多元，又在二维平面中绘制，则最终的拟合线将以多项式的形式进行绘制