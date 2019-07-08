#  if---then过程  决策的过程
#  决策树学习通常有三个步骤：特征选择   决策树的生成    修剪

#  分类决策树模型是一种对实例进行分类的树形结构，决策树由结点和有向边组成。结点有两种类型： 内部结点(特征或者属性)和叶结点(具体的类别)，最终将实例分配到叶结点的类中。

#  信息熵：熵指的体系的混乱程度。 香农熵：信息的度量方式，信息的混乱程度。信息特别有序：熵值越低。

#  信息增益：划分数据集前后信息发生的变化

#  决策树的工作原理（迭代思想）

#  决策树的开发流程： 收集数据  准备数据(预处理)   分析数据   训练数据   测试数据   使用算法

#  决策树本身的算法特性：优点 ： 计算复杂度不高  数据有缺失也能跑  可以处理一些不相关的特征   缺点： 容易过拟合

#   例子1：判断是不是鱼类或者非鱼类   1. 不浮出水面是否可以生存  2. 是否有脚蹼

from math import log
import operator

def createDataSet():
    """
    DataSet  数据集
    Args: 没有参数
    :return: 返回数据集和对应的label标签
    """
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'],[1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calShannonEnt(dataSet):
    #求list长度，表示计算参与训咯的数据量
    numEntries = len(dataSet)
    #分裂标签出现的次数
    lableCounts = {}
    for feat in dataSet:
        currentLabel = feat[-1]
        #为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典 把当前的键值加进去，每个键值记录了当前类别出现的次数
        if currentLabel not in lableCounts.keys():
            lableCounts[currentLabel] = 0
        lableCounts[currentLabel] += 1
    #根据label标签的占比情况，求出label标签的香农熵
    shannonEnt = 0
    for key in lableCounts:
        prob = float(lableCounts[key]) / numEntries
        #计算香农熵， 以2为底  求对数
        shannonEnt -= prob * log(prob , 2)
    return  shannonEnt

def splitDataSet(dataSet, index, value):
    """

    :param dataSet: 数据集         待划分的数据集
    :param index: 表示每一行的index列     划分数据集的特征
    :param value: 表示index对应的value值        需要返回的特征的值
    :return:
        index 列为value的数据集
    """
    retDataSet = []
    for feat in dataSet:
        #index列为value的数据集
        #判断index列的值是否为value
        if feat[index] == value:
            #[:index]
            reducedFeatVec = feat[:index]
            '''
            extend  append
            
            '''
            reducedFeatVec.extend(feat[index+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureSplit(dataSet):
    """
    选取最好的特征
    :param dataSet:  数据集
    :return: 返回最优的特征列
    """
    #求第一行有多少列的特征， 最后一列是label
    numFeatures = len(dataSet[0]) - 1
    #label的信息熵
    baseEntropy = calShannonEnt(dataSet)
    #最优的信息增益值，和最优的feature编号
    bestInfoGain, bestFeature = 0, -1
    #迭代所有的feature
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        #创建一个临时的信息熵
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        #信息增益
        infoGain =  baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature= ', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最优的列，得到最优列对应的label
    bestFeat = chooseBestFeatureSplit(dataSet)
    #获得label的名称
    bestFeatLabel = labels[bestFeat]
    #初始化树
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    #取出最优列
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value), subLabels)
    return myTree

def majorityCnt(classList):
    """

    :param classList: label列的集合
    :return: 最有特征列
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

