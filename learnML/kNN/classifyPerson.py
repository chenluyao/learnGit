from numpy import *
import operator
import kNNFile2Matrix
import kNNAlgo

def classifyPerson():
    resultList= ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    normDataMat, datingLabels, ranges, mins = kNNFile2Matrix.file2NormMatrix('datingTestSet2.txt')
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = kNNAlgo.classify((inArr - mins)/ranges, normDataMat, datingLabels, 4)
    print "You will probably like this person: ",resultList[classifierResult - 1]