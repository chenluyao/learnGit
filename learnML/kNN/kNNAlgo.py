from numpy import *
import operator

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # tile() entends inX into (dataSetSize,1)big matrix
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    sortedDistIndices = distances.argsort() # index list is returned after argsort()
    classCount = {} # {} defines a dictionary
    for i  in range(k): # indice all start from 0, range(k) = [0,1,...,k-1]
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # 2nd arg of get() defines default value if not found
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True) # sort of dictionary, in reversed order of quantity
    return sortedClassCount[0][0]
    
    