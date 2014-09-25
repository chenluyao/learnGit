from numpy import *
import operator
import kNNFile2Matrix
import kNNAlgo

import matplotlib
import matplotlib.pyplot as plt

def kNNTest(filename, hoRatio, k):
    normDataMat, datingLabels, ranges, mins = kNNFile2Matrix.file2NormMatrix(filename)
    m = normDataMat.shape[0]
    numTest = int(m * hoRatio) #need to be integer for later range(numTest)
    errorCount = 0.0
    errorRatePerc = 0.0
    for i in range(numTest):
        classifierResult = kNNAlgo.classify(normDataMat[i,:],normDataMat[numTest:m,:], datingLabels[numTest:m], k)
        #print "No. %d test: Classify-%d vs Real-%d" % (i+1, classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    errorRatePerc = 100 * errorCount/float(numTest)
    #print "DONE! Total Error Rate: %f %%" % errorRatePerc
    return errorRatePerc

###non-reusable##
def drawKNNPerformAsKChages(startK, endK):
    filename = 'datingTestSet2.txt'
    hoRatio = 0.1
    kErrorRateMat = zeros((endK-startK,2))
    index = 0
    errorRate = 0.0
    for k in range(startK,endK):
        errorRate = kNNTest(filename, hoRatio, k)
        kErrorRateMat[index,:] = [k,errorRate]
        index += 1
        print "k: %d, errorRate: %f %%" % (k, errorRate)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('k')
    ax.set_ylabel('ErrorRate%')
    l=ax.plot(kErrorRateMat[:,0], kErrorRateMat[:,1])
    ax.legend(l,('ErrorRate %',),'upper right') # 2nd para has to be tuple!!
    plt.title('k-ErrorRate')
    plt.show()
    
     

