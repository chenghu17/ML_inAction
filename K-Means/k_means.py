# author:hucheng

from numpy import *

def loadDataSet(filePath):
    fr = open(filePath)
    dataArr = []
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        arrLine = []
        for i in curLine:
            arrLine.append(float(i))
        dataArr.append(arrLine)
    return dataArr

def calDistance(vec1,vec2):
    return sqrt(sum(power((vec1-vec2),2)))

def randCent(dataSet,k):
    #获取数据集的列数
    n = shape(dataSet)[1]
    #注册中心质点的矩阵
    centroids = mat(zeros((k,n)))
    #随机取值，生成质点矩阵
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float((maxJ-minJ))
        #加号后面是使用每列中的数值间距乘上列数个[0,1)的随机数，即保证质点的取值都在数据集类
        #这里是针对于每一列！因为每一列为一个属性！
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataset,k,distMeas = calDistance,createCent = randCent):
    m = shape(dataset)[0]
    #数据集划分矩阵，第一列保存对应行的划分类别{0,1...,k},第二列记录距离的平方
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataset,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                #对数据集中的每一组数据，计算其与k个质点的距离，取距离最小进行归类
                distJI = distMeas(centroids[j,:],dataset[i,:])
                if distJI<minDist:
                    minDist = distJI;minIndex = j
            #这个判断条件??
            #感觉可以不用while循环，直接for循环
            #但是有点不对的地方在于，对于每一次归类后，需要重新更新质点，然后对每个数据再重新进行划分
            if clusterAssment[i,0] !=  minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):
            ptsInClust = dataset[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment


if __name__=='__main__':
    data = mat(loadDataSet('testSet_2.txt'))
    centroids,clusterassment = kMeans(data,4)
    print(centroids)