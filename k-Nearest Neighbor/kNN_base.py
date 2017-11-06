# coding:utf-8

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],
                   [1.0,1.0],
                   [0,0],
                   [0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#首先获取数据集的第一维的维数
#使用numpy中tile方法将输入inX向量copy成为dataset相同维度
#使用矩阵相减，得到inX与dataset各item的差值
#对相减得到的结果平方
#对每一行采用numpy中的sum(axis=1)计算每一行向量相加的和
#对每行的和开平方，得到测试点与dataset中各点的距离
#再使用numpy中的argsort()对距离进行排序，返回排序后的index(索引值)
#循环k次，对排序后的index对应的label进行归类，以label:count的字典形式
#对字典使用sorted(cmp,key,reverse)排序
#输出排序后的[0][0]值，即为测试数据所对应的label
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
        print(sortedDistIndicies[i], voteLabel)
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__=='__main__':
    group,labels = createDataSet()
    print(classify0([0,0],group,labels,3))