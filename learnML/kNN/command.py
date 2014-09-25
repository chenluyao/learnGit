from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt

def draw(datingDataMat, xInd, yInd, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,xInd], datingDataMat[:,yInd], 15.0*array(datingLabels),15.0*array(datingLabels)) #size, color
    plt.show()
    