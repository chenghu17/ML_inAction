# author:hucheng
# association analysis 、association rule learning
# frequent item sets
# association rules
# support 、confidence
# Apriori原理：如果某个项集是频繁的，那么它的所有子集也是频繁的，反过来也成立！
# 例如短板原理，一个桶的容量由最短的那块板决定。一个项集出现的频率由出现最少的那个子集来决定
# 关联分析的目标：发现频繁项集、发现关联规则



def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

#构建大小为1的候选项集的集合
def createC1(dataSet):
    C1 = []
    #对dataSet中的每个单项建立一个集合，从单项开始计算频繁项集
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #注这里使用frozenset，为不可改变的集合
    #由于之后需要作为字典键值使用，所以不能使用set类型
    return map(frozenset,C1)

#用于将C1生成L1，此处的L1是用于生成C2的
#D为数据集、Ck为候选项集列表Ck、感兴趣项集的最小支持度minSupport
def scanD(D,Ck,minSupport):
    #记录Ck中单项在dataSet出现的次数，以字典的形式存储
    ssCnt = {}
    #对于dataSet中的每组数据
    for tid in D:
        #判断Ck中的项是否为该组数据的子集
        for can in Ck:
            #如果是其子集，则++
            if can.issubset(tid):
                if can in ssCnt:
                    ssCnt[can] += 1
                else:
                    ssCnt[can] = 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    #retList为满足最小支持度的项
    #supportData为频繁项集的支持度集
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

# create Ck
# Lk为频繁项集，k为项集元素个数
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2])>0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

#未完，待续......