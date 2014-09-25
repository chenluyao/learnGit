from numpy import *
import operator

def file2NormMatrix(filename):
    datingDataMat, datingLabels = file2Matrix(filename)
    normMat, ranges, minValues = autoNorm(datingDataMat) 
    return normMat, datingLabels, ranges, minValues # ranges, minValues will be used in the norm of new input

def file2Matrix(filename):
    fr = open(filename)
    arrayOLine = fr.readlines()
    numberOfLines = len(arrayOLine)
    returnMat = zeros((numberOfLines, 3)) #why 3? coz 3 features for one person; why 2 pairs of ()? coz it's the 1st parameter as a whole
    classLabelVector = []
    index = 0
    for line in arrayOLine:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,: ] = listFromLine[0:3] #  [index,:] = [index, 0:ALL]
        classLabelVector.append(int(listFromLine[-1])) # number <- int
        index += 1
    return returnMat, classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 0 -> pick min from columns    (3,1)
    maxVals = dataSet.max(0) # same as above
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1)) # extends to matrix
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
    