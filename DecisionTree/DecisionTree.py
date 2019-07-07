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
import matplotlib.pyplot as plt

def createDataSet():
    """
    DataSet  数据集
    Args: 没有参数
    :return: 返回数据集和对应的label标签
    """
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'],[1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']   # 不出露出水面   脚蹼
    return dataSet, labels

#计算信息增益，为了寻找最优特征
def calShannonEnt(dataSet):
    #求list长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    #分类标签出现的次数
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
    return shannonEnt

def splitDataSet(dataSet, index, value):
    """

    :param dataSet: 数据集         待划分的数据集
    :param index: 表示每一行的index列     划分数据集的特征
    :param value: 表示index对应的value值        需要返回的特征的值
    :return:
        index 列为value的数据集
    """
    #append:  添加一个对象  整体打包
    #extend: 向内容添加到列表中  合并 merge后面
    retDataSet = []
    for featVec in dataSet:
        #index列为value的数据集
        #判断index列的值是否为value
        if featVec[index] == value:
            #[:index]
            reducedFeatVec = featVec[:index]
            '''
            extend  append
            
            '''
            reducedFeatVec.extend(featVec[index+1:])
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
        infoGain = baseEntropy - newEntropy
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

def classify(inputTree, featLabels, testVec):
    """

    :param inputTree:   决策树模型
    :param featLabels:  feature标签对应的名称
    :param testVec:  测试输入的数据
    :return:   分类的结果，需要映射到label才能知道名称
    """

    #获取树的根节点对应的key值
    firstStr = list(inputTree.keys())[0]
    print('key值', firstStr)
    #通过key知道根节点对应的value
    secondDict = inputTree[firstStr]
    print('根节点对应的value', secondDict)
    #判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    #测试数据，找到根节点对应的label的位置，也就知道从输入的数据的第几位开始进行分类
    key = testVec[featIndex]
    valueOFFeat = secondDict[key]
    print('+++++', firstStr,  'xxxxx', secondDict, '-----', key, '<<<<<', valueOFFeat)
    #判断分支是否结束，判断valueOFFeat是否是dict类型
    if isinstance(valueOFFeat, dict):
        classLabel = classify(valueOFFeat,featLabels,testVec)
    else:
        classLabel = valueOFFeat
    return classLabel

def get_tree_height(tree):
    """
    递归获取获取输的高度

    :param tree: tree
    :return: 高度
    """

    if not isinstance(tree, dict):
        return 1
    child_trees = list(tree.values())[0].values()

    #遍历子树，获得字数的最大高度

    max_height = 0

    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height
    return max_height + 1

def fishTest():
    # 1.创建数据和结果的标签
    myData, labels = createDataSet()
    #打印myData  labels

    #计算label的分类标签的香农熵

    #calShannonEnt(myData)

    import copy
    myTree = createTree(myData, copy.deepcopy(labels))
    print(myTree)
    #[1,1]
    print(classify(myTree, labels, [1, 1]))

    #获取树的高度
    print(get_tree_height(myTree))

def ContactLensesTest():
    """
    眼镜类型的测试
    :return:
    """
    #加载文本文件  数据集
    fr = open('lenses.txt')
    #解析数据， 获得features数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    #得到数据对应的labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 构建决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)

if __name__ == "__main__":
    fishTest()
    ContactLensesTest()













