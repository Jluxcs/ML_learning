from numpy import *
import os

returnVect = zeros((1, 1024))
fr = open('test.txt')
for i in range(1):
    lineStr = fr.readline()
    print(lineStr)
    for j in range(32):
        print(int(lineStr[j]))
        returnVect[0, 32 * i + j] = int(lineStr[j])

        print(returnVect)