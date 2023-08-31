#!/usr/bin/python3

import numpy as np
import pickle as pkl

class ExperimentResult:

    #
    #
    #
    def __init__(self, trainCompressorTime, compressTime, classifierTimeTrain,classifierTimeTest, avgClassifierAccuracy, stddevClassifierAccuracy,rhoTime,outputVertices):
        self.trainCompressorTime = trainCompressorTime
        self.compressTime = compressTime
        self.classifierTimeTrain = classifierTimeTrain
        self.classifierTimeTest = classifierTimeTest
        self.avgClassifierAccuracy = avgClassifierAccuracy
        self.stddevClassifierAccuracy = stddevClassifierAccuracy
        self.outputVertices=outputVertices
        self.rhoTime=rhoTime



    #
    #
    #
    def __str__(self):
        retval = "Compressor training time, Compression time, classification training time, classification testing time, average accuracy, stddev accuracy, Vertices\n"
        retval += str(self.trainCompressorTime) + ", " + str(self.compressTime) + ", " + str(self.classifierTimeTrain) + ", "+ str(self.classifierTimeTest) + ", " + str(self.avgClassifierAccuracy) + ", " + str(self.stddevClassifierAccuracy)+ ", " +str(self.rhoTime)+"\n"+str(self.outputVertices)+"\n"
        return(retval)

    #
    #
    #
    def toList(self):
        return([self.trainCompressorTime, self.compressTime,self.classifierTimeTrain, self.classifierTimeTest, self.avgClassifierAccuracy, self.stddevClassifierAccuracy,self.rhoTime,self.outputVertices])


    #
    #
    #
    # def writeToFile(self, filePrefix, run, numRuns):
    def writeToFile(self, filePrefix):#, run, numRuns):
        filename = filePrefix# + str(run) + "-" + str(numRuns)
        with open(filename, 'wb') as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL);



#
# Read the given experimental result from the specified file.
#
def readResultsFromFile(filePrefix, compressionRatio, run, numRuns):
    filename = filePrefix + str(compressionRatio) +"." + str(run) + "-" + str(numRuns)
    retval = None
    with open(filename, "rb") as handle:
        retval = pkl.load(handle)
    return(retval)


#
#
#
def readResultsFromFiles(filePrefix, compressionRatios, numRuns, runs=None):
    results = []
    if type(runs) == type(None):
        runs = range(1, numRuns+1)
    for ratio in compressionRatios:
        for j in runs:
            result = readResultsFromFile(filePrefix, ratio, j, numRuns)
            results.append(result)
    return(results)

#
# Convert a list of ExperimentResults to a single numpy array.
# The shape is # of runs x # of result values (i.e., 5)
#
def resultsToNumpyArray(resultList):
    arrayLst = []
    for res in resultList:
        arr = np.array(res.toList())
        arrayLst.append(arr)
    retval = np.stack(arrayLst)
    print("retval.shape:" + str(retval.shape))
    return(retval)



#
#
#
def computeStatistics(resultList):
    resultsNp = resultsToNumpyArray(resultList)
    mean = resultsNp.mean(axis=0)
    stddev = resultsNp.std(axis=0)
    print("Shape of resultsNp: " + str(resultsNp.shape))
    print("Mean: " + str(mean))
    print("Stddev: " + str(stddev))
    return(mean, stddev)




###################################################################
# MAIN
#
# Experimental analysis routines.
#
if __name__ == "__main__":
    #result = readResultsFromFile("./results/result-gaussian-dependent.pkl", 0.1, 3, 5)
    results = readResultsFromFiles("./results/result-gaussian-dependent.pklvikas", [0.2], numRuns=5, runs=[1, 2, 3, 4])
    statistics = computeStatistics(results)
