
def splitDataSet(index):
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    retDataSet = []
    test2 = len(dataSet[0])
    print(test2)
    for feat in dataSet:
        reducedFeatVec = feat[:index]
        reducedFeatVec.extend(feat[index+1:])
        retDataSet.append(reducedFeatVec)

    print(retDataSet)

if __name__ == '__main__':
    splitDataSet(2)

result = []
result.extend([1,2,3])
print(result)

result.append([4,5,6])
print(result)
result.extend([7,8,9])

print(result)

