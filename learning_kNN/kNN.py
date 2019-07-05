from numpy import *
from os import listdir
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return  group, labels
def classify0(inX, dataSet, labels, k):
    """
    :param inX: [1,2,3]
    :param dataSet: 输入的训练样本集 [[1,2,3],[1,2,0]]
    :param labels: 训练集标签 labels元素数目要与dataset中行数相同  程序使用欧式距离（曼哈顿距离）
    :param k: 选择最近邻的数目
    :return: maxClassCount   最大分类个数
    """
    # -------------实现classify0()第一种实现方式----------------------
    #1. 距离计算
    dataSetSize = dataSet.shape[0]   #训练集的行数
    #使用tile生成和训练样本对应的矩阵，并与训练样本求差

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    """
    欧式距离：点到点的距离
        第一行：同一个点到dataSet的第一个点的距离
        第二行：同二个点到dataSet的第二个点的距离
        .
        .
        .
        第N行：同一个点到dataSet的第N个点的距离
        [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
        
    """
    #求平方
    sqDiffMat = diffMat ** 2


    #将矩阵每一行相加
    sqDistances = sqDiffMat.sum(axis =1)

    #开方

    distances = sqDistances ** 0.5

    #根据距离排序  从大到小的排序  ， 返回对应的索引位置

    #argsort()  是将x中的元素从小到大排序， 提取其对应的index(索引), 然后输出到y。

    #for example: y = array([3, 0, 2, 1, 4, 5])  >>>  [1, 3, 2, 0, 4, 5]

    sortedDisIndicies = distances.argsort()

    #print(dataSetSize)
    print(sortedDisIndicies)
    # 2. 选取距离最小的k个点

    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDisIndicies[i]]
        # 例如： list.get(k,d) get相当于一个if else语句
        #  l = {5:2, 3:4}   l.get(3, 0)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount
# -------------实现classify0()第二种实现方式----------------------
"""
    欧式距离：点到点的距离
        第一行：同一个点到dataSet的第一个点的距离
        第二行：同二个点到dataSet的第二个点的距离
        .
        .
        .
        第N行：同一个点到dataSet的第N个点的距离
        [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]

"""
#dist = np.sum((inX - dataset) ** 2, axis = 1) ** 0.5

def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件的路径
    :return: 返回对应的类别
    """
    fr = open(filename)
    #获得文件中数据行的行数
    number0fLines = len(fr.readlines())
    #生成对应的空矩阵
    #例如：zeros(2, 3)
    returnMat = zeros((number0fLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        #label标签
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
def autoNorm(dataset):
    """

    :param dataset: 数据集
    :return: 归一化后的数据集normDataSet
    归一化公式：
         Y = (X - Xmin) / (Xmax - Xmin)
    """
    minVals = dataset.min(0)
    maxVals = dataset.max(0)

    #极差
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataset))
    m = dataset.shape[0]

    normDataSet = dataset - tile(minVals, (m, 1))

    normDataSet = normDataSet / tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

def datingClassTest():
    """
      对约会数据进行分析
    :return: 错误数
    """
    #设置一个测试数据的比例
    hoRatio = 0.1
    #从文件中加载数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)

    #m 表示数据的行数，矩阵的第一维
    m = normMat.shape[0]

    #设置测试的样本数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs = ', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        #对测试数据
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"% (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    """
    将图像转化为向量
    :param filename: 图片文件  32*32=1024
    :return:一维矩阵

    首先创建一个1*1024的0数组，然后打开给定的文件， 循环读出文件的前32行
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    # 导入训练数据
    hwLabels = []
    trainingFileList = listdir('trainDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainDigits/%s' % fileNameStr)
    #print(trainingMat)
    # 导入测试数据
        # 2. 导入测试数据
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

def test1():
    """
    第一个例子演示
    """
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 1))

if __name__ == '__main__':
    #test1()
    #handwritingClassTest()
    datingClassTest()
