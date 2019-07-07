import os
import operator
def splitDataSet(dataSet, index, value):
    """

    :param dataSet: 数据集         待划分的数据集
    :param index: 表示每一行的index列     划分数据集的特征
    :param value: 表示index对应的value值        需要返回的特征的值
    :return:
        index 列为value的数据集
    """
    # append:  添加一个对象  整体打包
    # extend: 向内容添加到列表中  合并 merge后面
    retDataSet = []
    print(len(dataSet))
    for featVec in dataSet:
        # index列为value的数据集
        # 判断index列的值是否为value
        if featVec[index] == value:
            # [:index]
            reducedFeatVec = featVec[:index]
            print(reducedFeatVec)
            reducedFeatVec.extend(featVec[index + 1:])
            print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)

    print(retDataSet)
    print(len(retDataSet))

if __name__ == "__main__":
    datasetnew = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in datasetnew.readlines()]
    splitDataSet(lenses, 2, 'yes')
