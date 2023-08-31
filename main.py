#!/usr/bin/python3
#
#
# Here we implement our optimal transport compression framework.
# This notebook additionally contains tests of our compression method.
#
# Given:
# - A common set $V$ of n nodes.
# - N samples, where each sample is an independent random graph
#   A^{(j)} on the vertex set $V$, with node data $X_v^{(j)}$.
#
# Output:
# - A subset $\hat{V}$ of nodes.  The subgraph induced by $\hat{V}$
# is the output of every graph in the dataset.

#
# Method: We use the optimal transport framework of that one kid from MIT.
# The inputs to his method are the initial distribution $p_0$ on the
# vertex set, plus the cost function value for each pair of vertices.
# Here, we are actually applying his method to a complete weighted graph.
#
# Our job in the present code is to determine the initial distribution and
# the cost function.
import sys
import pickle
from math import *;
import random;

import numpy as np
from grakel.utils import graph_from_networkx;
import networkx as nx;
from sklearn.feature_selection import mutual_info_classif

from ExperimentResult import *
from GenerateDatasets import *;
from ComputeAccuracy import *;
# from time import time;
import time
from OTC import *;
from gcn_mi_estimator_final import *;
# import Non_fixed2fixed as scm;
import SensitivityCompression as scm;
from kaggle import load_data
from MI_Estimator import MI
from readpkl import data2txt
###############################################################
#
# OT compression routines.
#

#
#
#
class OTParameters:

    #
    #
    #
    def __init__(self, initialDistribution, costMatrix):
        self.__initialDistribution = initialDistribution;
        self.__costMatrix = costMatrix;
        
    #
    #
    #
    def getParameters(self):
        return( (self.__initialDistribution, self.__costMatrix));
###############################################################

#
# Given two numpy arrays, compute the mutual information
# between X and C.
# X is required to have shape
# # of samples x size of feature x 1
#
# C has shape # of samples
#
#
def mutualInfo(X, C):
    Xreshaped = X
    if (len(Xreshaped.shape) == 3):
        Xreshaped = X[:,:,0]
    print("Xreshaped: " + str(Xreshaped.shape) + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    retval = mutual_info_classif(Xreshaped, C)
    return(retval)


#
# Get the initial distribution for our compression method.
# The argument is a dataset (i.e., a Dataset object).
#
# Recall that the probability assigned to a vertex v is
# given by
# I(X_v; C) / \sum_{w} I(X_w; C).
#
def getInitialDistribution(dataset):
    denominator = 0.0;
    numNodes = dataset.getGraphSize();
    numClasses = dataset.getNumClasses();
    featureSize = dataset.getFeatureSize();
    features = dataset.getFeatures();
    classLabels = dataset.getClassLabels();
    numSamples = dataset.getDatasetSize();

    print("Features shape:" + str(features.shape))
    retval = np.zeros(numNodes);
    for v in range(numNodes):
        X = features[:,v];      # Samples for node v.  Should be 100 x 1 for our synthetic datasets.
        X = np.stack([X], 1)    # Stack for silly reasons.
        print("Shape of X: " + str(X.shape))
        print("NumSamples: " + str(numSamples))
        print("numClasses: " + str(numClasses))
        print("feature size:" + str(featureSize))
        #IXvC = MI_Xv(X=X, L=classLabels, N=numSamples, V=1, D=featureSize, C=numClasses, F=5);
        IXvC = mutualInfo(X, classLabels)
        print("denominator: " + str(denominator))
        print("IXvC:" + str(IXvC))
        retval[v] = IXvC;
        denominator += IXvC;
    
    retval /= denominator;
    #for v in range(numNodes):
    #    retval[v] /= denominator;
    return(retval);    


#
# Given two numpy arrays encoding probability mass functions,
# return the KL divergence of the two.
#
def KLDivergence(dist0, dist1):
    retval = 0;

    d0Support = dist0.shape[0];
    d1Support = dist1.shape[0];

    
    for x in range(d0Support):
        if x >= d1Support:
            d1x = 0;
        else:
            d1x = dist1[x];
        #print(dist0[x])
        #print(d1x)
        if dist0[x] > 0:
            retval += dist0[x] * log(dist0[x]/d1x);
    return(retval);        

   


#
# Compute D_{KL}(E_{v,w} | C=0 || E_{v,w} | C=1).
# The return value is an n x n numpy matrix.
#
def getDkl(dataset):
    retval = 0;
    

    # Compute the empirical distribution of E_{v,w} | C=0.  Same for E_{v,w} | C=1.
    adj0 = dataset.getAdjacencies(classLabel = 0);
    adj1 = dataset.getAdjacencies(classLabel = 1);
    numVtcs = adj0.shape[1];
    numGraphs0 = adj0.shape[0];
    numGraphs1 = adj1.shape[0];
    #totalNumGraphs = numGraphs0 + numGraphs1;

    # Sum all adjacency matrices for adj0 together.  Do the same for adj1.
    # These will give us two matrices that give empirical estimates of
    # the probability of an edge between v, w, conditioned on the value of C.
    empiricalP0 = np.sum(adj0, 0) / numGraphs0;
    empiricalP1 = np.sum(adj1, 0) / numGraphs1;

    empiricalDist0 = np.stack((1 - empiricalP0, empiricalP0), 2)
    empiricalDist1 = np.stack((1 - empiricalP1, empiricalP1), 2)
    retval = np.zeros((numVtcs, numVtcs));
    for i in range(numVtcs):
        for j in range(numVtcs):
            retval[i][j] = KLDivergence(empiricalDist0[i][j], empiricalDist1[i][j]);
    return(retval);

#
# Compute I(X_v; C | X_w).
#
def getIXvCGivenXw(dataset, v, w):
    features = dataset.getFeatures();
    classLabels = dataset.getClassLabels();
#    featureSize = dataset.getFeatureSize();
#    numClasses = dataset.getNumClasses();
#    numNodes = dataset.getGraphSize();
    numSamples = dataset.getDatasetSize();

    # Get I(X_v, X_w; C).
    # Get I(X_w; C).
    jointFeatures = []
    for samp in range(numSamples):
        jointFeatures.append( (features[samp, v]+1) * (-1)**(features[samp, w]))
    print("Jointfeatures: " + str(jointFeatures))
    IXvwC = mutual_info_classif(jointFeatures, classLabels)
    IXwC = mutualInfo(features[:, w], classLabels)
    retval = IXvwC - IXwC
    print("This estimator is broken!")
    return(retval)



#
# R_{v,w} = I(X_v; C | X_w) + I(X_w; C | X_v).
# Use MI_X(X, L, N, V, D, C, F) for this.
#
def getR(dataset, attributed=False):
    if not attributed:
        return(0.0)
    # features is a numpy array: # of data points x # of nodes x size of a feature vector.
    features = dataset.getFeatures();      
    classLabels = dataset.getClassLabels();
    featureSize = dataset.getFeatureSize();
    numClasses = dataset.getNumClasses();
    numNodes = dataset.getGraphSize();
    numSamples = dataset.getDatasetSize();

    infoMatrix = np.zeros((numNodes, numNodes));
    retval = np.zeros((numNodes, numNodes));
    
    for v in range(numNodes):
        for w in range(numNodes):
            # Compute the v, w entry of infoMatrix.  This is I(X_v; C | X_w).
            # Note that I(X_v, X_w; C) = I(X_w; C) + I(X_v ; C | X_w).
            # So I(X_v; C | X_w) = I(X_v, X_w; C) - I(X_w; C).
            X = np.stack( [ features[:, v, :], features[:, w, :]  ], 1);
            print("MADE IT HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(time())
            infoMatrix[v,w] = MI_X(X=X, L=classLabels, N=numSamples, V=2, D=featureSize, C=numClasses, F=5);
            print(time)
            #infoMatrix[v,w] = getIXvCGivenXw(dataset, v, w)
    
    for v in range(numNodes):
        for w in range(numNodes):
            retval[v, w] = infoMatrix[v, w] + infoMatrix[w, v];
    return(retval);



#
#
#
def getCostMatrix(dataset):
    # Use MI_X(X, L, N, V, D, C, F) to compute I(X_v; C | X_w).
    # c(v, w) = D_{KL}(E_{v,w} | C=0 || E_{v,w} | C=1) + R_{v,w},
    # where R_{v,w} = I(X_v; C | X_w) + I(X_w; C | X_v).
    costMatrix = getDkl(dataset) + getR(dataset);
    return(costMatrix);


#
# This is the main function to call for learning the optimal transport
# compression parameters.  Given a dataset, it returns an OTParameters object that
# can be fed into the optimal transport compressor.
#
def learnOTParameters(dataset):
    return(OTParameters(getInitialDistribution(dataset), getCostMatrix(dataset)));



#
# This compresses a dataset of graphs using the given optimal transport parameters.
# 
# OTParams -- an OTParameters object.  This contains an initial distribution and
# a cost matrix.
# numNodes -- The number of nodes in the graph.
# compressionRatio -- The desired compression ratio.  A floating point number.
# 
# The output of this method is a set of vertices.  This is a Python set object
# whose elements are integers that correspond to vertices in the graphs of the
# dataset.
#


#Sensitivity injection
# def OurOTCompress(sensitiveNodes,OTParams, numNodes, compressionRatio):
####
def OurOTCompress(OTParams, numNodes, compressionRatio):
    initDistribution, costMatrix = OTParams.getParameters(); 

    # The idea is to construct a new dataset consisting of a single graph: the complete graph on
    # n vertices (no other information is necessary).  We will take the initial distribution and the cost matrix and feed that
    # into Vikas's method, which will spit out a subset of vertices.
    
    # Create a complete graph on numNodes nodes.
    G = nxToGrakel(nx.complete_graph(numNodes));

    #Sensitivity injection
    # vertices = CompressGraphWithParams(sensitiveNodes,G, compressionRatio, initDistribution, costMatrix);
    #####
    vertices = CompressGraphWithParams(G, compressionRatio, initDistribution, costMatrix);
    vertices = [ v for v in range(len(vertices)) if int(vertices[v]) == 1 ]
    print("The vertices returned by OurOTCompress: " + str(vertices));
    return(vertices);


    


#
# Given a Dataset object, which we assume is gotten by compressing another
# dataset via some method (doesn't matter what), we train and test a graph
# classifier.  We then return the results.
#
def evaluateCompressedDataset(TestDataset):
    graphs = TestDataset.getGraphs(toGrakel=True, attributed = True, flatten=False);
    classLabels = TestDataset.getClassLabels();
    print("Train Class labels: " + str(classLabels))
    print("Size of each graph in the projected dataset: " + str(TestDataset.getGraphSize()))
    print("First graph: " + str(graphs[0]))
    print("Features of first graph: " + str(TestDataset.get(0)[1]))

    # graphsTest = TestDataset.getGraphs(toGrakel=True, attributed = True, flatten=False);
    # classLabelsTest = TestDataset.getClassLabels();
    # print("Test Class labels: " + str(classLabelsTest))
    # print("Size of each graph in the projected dataset: " + str(TestDataset.getGraphSize()))
    # print("First graph: " + str(graphsTest[0]))
    # print("Features of first graph: " + str(TestDataset.get(0)[1]))

    retval = ComputeAccuracy(graphs, classLabels, continuousFeatures=gaussianFeatures)#,graphsTest, classLabelsTest)#, continuousFeatures=gaussianFeatures)
    return(retval)


#
# Use this instead of learnOTParameters() for debugging purposes.
#
def getDummyOTParameters(dataset):
    n = dataset.getGraphSize()
    initDist = np.array([1/n for j in range(n)])
    costMatrix = np.array([[1/n for j in range(n)] for i in range(n)])
    retval = OTParameters(initDist, costMatrix)
    return(retval)


#
# This is our main synthetic experiment.
# We should return the following statistics:
# 0.) Time to train the compressor.
# 1.) Time to compress the dataset.
# 2.) Time to train and test the classifier.
# 3.) Accuracy of the classifier.
# This is for one single instance of the experiment.
#

#Sensitivity injection
# def runSyntheticExperiment(sensitiveNodes,trainingDataset, testDataset,ClassifyOnly, compressionRatio):
###
def runSyntheticExperiment(trainingDataset, testDataset,ClassifyOnly, compressionRatio):
    testingFlag = False     # If this is true, then just use a hard-coded choice of output vertices.
    print("TESTING FLAG IS " + str(testingFlag) + "!")

    startTrainTime = time()
    if not testingFlag:
        # Learn OT parameters using our method.
        print("Learning the OT parameters using our method");
        ourOTParameters = learnOTParameters(trainingDataset);
        #ourOTParameters = getDummyOTParameters(trainingDataset)
    numNodes = trainingDataset.getGraphSize()


    #  Run optimal transport compression using our parameters.
    print("Running our optimal transport compression.");
    if not testingFlag:
        pass
        # Sensitivity injection
        # outputVertices = OurOTCompress(sensitiveNodes, ourOTParameters, numNodes=numNodes, compressionRatio=compressionRatio)
        ###
        outputVertices = OurOTCompress(ourOTParameters, numNodes=numNodes, compressionRatio=compressionRatio)
    else:
        #outputVertices = [0, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 16, 17, 19, 20, 22, 25, 26, 33]
        outputVertices = [26, 33]
    endTrainTime = time()

    startCompressTime = time()
    print("Projecting on testdataset")
    ourCompressedDataset = testDataset.project(outputVertices);
    endCompressTime = time()
    print("Got our compressed dataset.");
    # Train a classifier using our compressed dataset.
    if ClassifyOnly:
        # RESULT_FILENAME = RESULT_FILENAME[:-4]+"baseline.pkl"
        ourResult = evaluateCompressedDataset(testDataset)#(ourCompressedDataset);

    else:
        pass
        ourResult = evaluateCompressedDataset(ourCompressedDataset)#(ourCompressedDataset);
    print("Our results: " + str(ourResult));
    # avgAccuracy, stddevAccuracy, classifyTimeTrain,classifyTimeTest = ourResult
    Accuracies, classifyTimeTrains,classifyTimeTests = ourResult
    retval=endTrainTime - startTrainTime, endCompressTime-startCompressTime,Accuracies, classifyTimeTrains,classifyTimeTests,outputVertices
    # retval = ExperimentResult(endTrainTime - startTrainTime, endCompressTime-startCompressTime, classifyTimeTrain,classifyTimeTest, avgAccuracy, stddevAccuracy)#,outputVertices)
    return(retval)

########################################################
# VIKAS'S METHOD
def runSyntheticExperimentVikas(trainDataset,testDataset, compressionRatio):
    #  Run optimal transport compression using Vikas's method.
    # Compute OT parameters using Vikas's method.

    startCompressTime = time()
    vikasOTParameters = getVikasOTParameters(trainDataset, isDiscrete=gaussianFeatures);
    vikasCompressedDatasetTrain,nodeSubsets = VikasOTCompress(vikasOTParameters, trainDataset, compressionRatio);


    vikasCompressedDatasetTest = testDataset.projectSubsets(nodeSubsets)#changed projectSubsets to project
    # time.sleep(10)
    endCompressTime = time()

    trainTime = -1      # There is no train time.
    # vikasCompressedDatasetTest = VikasOTCompress(vikasOTParameters, testDataset, compressionRatio);
    # Train a classifier using Vikas's method.
    vikasResult = evaluateCompressedDataset(vikasCompressedDatasetTest);
    print("Vikas's results: " + str(vikasResult));
    Accuracies, classifyTimeTrains,classifyTimeTests = vikasResult
    retval=trainTime, endCompressTime-startCompressTime,Accuracies, classifyTimeTrains,classifyTimeTests,nodeSubsets
    # retval = ExperimentResult(trainTime, endCompressTime-startCompressTime, classifyTimeTrain,classifyTimeTest, avgAccuracy, stddevAccuracy)#,nodeSubsets)
    return(retval)

    #TODO: Test the classifiers.  In particular, plot the performance of each method
    # as a function of the compressed graph size.
    # Note that we can use ComputeAccuracy from Vikas's code to do this.


#################################################################
#
# Experiment demonstrating non-monotonicity of the mutual
# information.
#





################################################################
#
# MAIN
#
if __name__ == "__main__":
    #Our method/OTC/Baseline
    runOurExperiment = True
    runVikasExperiment = False
    ClassifierOnly=False
    fixed_node_size=True
    #Mutual Information estimator activating
    MutualInfo=False
    MutualInfoVikas=True

    compressionRatios = [.2,.3,.4,.5,.6,.7,.8]
    #load image datasets and generate their graphs
    if datasetType == "MNIST" or datasetType == "CIFAR" or datasetType == "MiniImageNet":
        X_trains, y_trains, X_tests, y_tests, adj = load_data(Sample_Size=300, k=8,numRuns=numRuns)
        trainingDataset = Dataset()
        i = 0
        for (xs, ys, adj_item) in zip(X_trains, y_trains, adj):
            i += 1
            print("\nIteration number " + str(i))
            G = nx.Graph()
            [G.add_node(i) for i in range(0, xs.shape[0])]
            [G.add_edge(vIdx, wIdx, weight=1) for (vIdx, wIdx) in # if vIdx < wIdx else 0 for (vIdx, wIdx) in
             zip(adj_item.nonzero()[0], adj_item.nonzero()[1])]
            trainingDataset.add(G, xs, ys)
        testDataset = Dataset()
        i = 0
        for (xs, ys, adj_item) in zip(X_tests, y_tests, adj[(int((300 * 4) / 5)):]):
            i += 1
            print("\nIteration number " + str(i))
            G = nx.Graph()
            [G.add_node(i) for i in range(0, xs.shape[0])]
            [G.add_edge(vIdx, wIdx, weight=1) for (vIdx, wIdx) in # if vIdx < wIdx else 0 for (vIdx, wIdx) in
             zip(adj_item.nonzero()[0], adj_item.nonzero()[1])]
            testDataset.add(G, xs, ys)

        print("Read the datasets!")
        if fixed_node_size:
            #Due to inadequate resources the sensitive subset of nodes are considered as the input of all methods
            selectedNodes = scm.Rho(trainingDataset,NODE_SIZE)  # , datasetType)  # Gamma1,Gamma2,datasetType)
            subnodes = np.array(np.where(selectedNodes == 1)).flatten()
            print("Projecting train and test dataset")
            trainingDataset = trainingDataset.project(subnodes)
            testDataset = testDataset.project(subnodes)
            #the generated data is saved for further use
            data2txt(trainingDataset)
            print("This is the edge average Num:"+  np.mean(np.array([len(item.edges) for item in np.array(trainingDataset.points)[:,0]])))
        else:
            allselectedNodes = scm.Rho(trainingDataset)  # , datasetType)  # Gamma1,Gamma2,datasetType)
            subnodes = [np.array(np.where(selectedNodes == 1)).flatten() for selectedNodes in allselectedNodes]
            print("Projecting train and test dataset")
            # ??????????
            trainingDataset = trainingDataset.project(subnodes)
            testDataset = testDataset.project(subnodes)
            # ??????????
    # trainingDataset = trainingDataset.subDataset(set(range(100)))            # Used to be 50
    # testDataset = testDataset.subDataset(set(range(50)))                    # Used to be 40
    if datasetType == "regression":
        trainingDataset = readDataset(TRAINING_FILENAME, run=numRuns)
        testDataset = readDataset(TEST_FILENAME, run=numRuns)
    if datasetType=="NYC":
        trainingDataset = readDataset(TRAINING_FILENAME)
        testDataset = readDataset(TEST_FILENAME)
        print("hi")
        pass
    if datasetType == "syntheticNew":
        trainingDataset = readDataset(TRAINING_FILENAME)
    if datasetType == "synthetic":
        # trainingDataset = readDataset(TRAINING_FILENAME, run=run + 1)
        # testDataset = readDataset(TEST_FILENAME, run=run + 1)
        trainingDataset = readDataset(
            TRAINING_FILENAME[:19] + str(trainingDataSize) + "-SampleSize-" + str(NODE_SIZE) + "-NODE_SIZE-" + str(
                featDim) + "-featDim-" + TRAINING_FILENAME[19:] + str(0 + 1) + "-" + str(numRuns))#run=0
        testDataset = readDataset(
            TEST_FILENAME[:19] + str(trainingDataSize) + "-SampleSize-" + str(NODE_SIZE) + "-NODE_SIZE-" + str(
                featDim) + "-featDim-" + TEST_FILENAME[19:] + str(0 + 1) + "-" + str(numRuns))
        data2txt(trainingDataset)

    MutInfosTrain=[]
    MutInfosTest=[]
    for ratio in compressionRatios:
        print("Performing the experiment for compression ratio " + str(ratio))
        results=[]
        for rhoRatio in [0.2,0.4,0.6,0.8]:
            for rhoPrimeRatio in [.2]:#rho prime can be another sensitivity measure that uses edges but here it is not used
                MIs=[]
                for run in range(numRuns):
                    if runOurExperiment:
                        if MutualInfo == True:
                            file="result_final_dataset_gaussian_dependent_100sample_50nodes_noiseadded.pkl"
                            if MutualInfoVikas:
                                with open(file,'rb') as f:
                                    data = pickle.load(f)
                                    pass
                                    MutInfoTrain=[]
                                    for sampleSet in data.outputVertices[run]:
                                        NewtrainingDataset=trainingDataset.project(sampleSet)
                                        MIIter = NewtrainingDataset
                                        MutInfoTrain.append(MI(MIIter.getFeatures(), MIIter.getAdjacencies(),
                                                          MIIter.getClassLabels(),
                                                          MIIter.size(), MIIter.getGraphSize(),
                                                          MIIter.getFeatureSize()[
                                                              1] if MIIter.getFeatures().ndim == 3 else 1,
                                                          MIIter.getNumClasses(), 3))
                                    MutInfoTrain=np.mean(np.array(MutInfoTrain))
                                    with open('MutualInfo.txt', 'a') as f:
                                        f.write("\n\nVikasSynthetic" + str(MutInfoTrain))
                                    MIs.append(MutInfoTrain)
                            else:
                                with open(file,'rb') as f:
                                    data = pickle.load(f)
                                NewtrainingDataset = trainingDataset.project(data.outputVertices[run])
                                MIIter = NewtrainingDataset
                                MutInfoTrain=MI(MIIter.getFeatures(), MIIter.getAdjacencies(),
                                                       MIIter.getClassLabels(),
                                                       MIIter.size(), MIIter.getGraphSize(),
                                                       MIIter.getFeatureSize()[
                                                           1] if MIIter.getFeatures().ndim == 3 else 1,
                                                       MIIter.getNumClasses(), 3)
                                with open('MutualInfo.txt', 'a') as f:
                                    f.write("\n\nOurSynthetic" + str(MutInfoTrain))
                                MIs.append(MutInfoTrain)
                        else:
                            #if we are not estimating MI then apply our method
                            startRhoTime=time()
                            #find sensitive nodes
                            sensitiveNodes=scm.Rho(trainingDataset,int(NODE_SIZE*ratio*rhoRatio))
                            #Rho Prime:
                            # if rhoRatio!=0:
                            #     sensitiveNodesRhoPrime=scm.Rho_Prime(trainingDataset,sensitiveNodes,int(NODE_SIZE*ratio*rhoRatio*rhoPrimeRatio))
                            #     j=0
                            #     for i in range(len(sensitiveNodes)):
                            #         if(sensitiveNodes[i]!=0):
                            #             if(sensitiveNodesRhoPrime[j]==0):
                            #                 sensitiveNodes[i]=0
                            #             j+=1

                            endRhoTime=time()
                            rhoTime=endRhoTime-startRhoTime#this is the sensitivity measurement time
                            #this is for when we are using rho:
                            # result = runSyntheticExperiment(sensitiveNodes, trainingDataset, testDataset,ClassifierOnly, compressionRatio=ratio);

                            result = runSyntheticExperiment(trainingDataset, testDataset,ClassifierOnly, compressionRatio=ratio);
                            # if type(result) != ExperimentResult:
                            #     raise("Experimental result expected.  Didn't get it.")
                            # result.writeToFile(RESULT_FILENAME + str(ratio) + ".", run=run+1, numRuns=numRuns)
                            print("Result: " + str(result))
                            results.append(result)
                    if MutualInfo!=True:
                        if runVikasExperiment:
                            result = runSyntheticExperimentVikas(trainingDataset, testDataset, compressionRatio=ratio)
                            # result.writeToFile(RESULT_FILENAME + "vikas" + str(ratio) + ".", run=run+1, numRuns = numRuns)
                            results.append(result)
                if MutualInfo:
                    MIResult=np.mean(np.array(MIs))
                    print("\nFinal MI: ",MIResult)
                    # np.array(MIs).writeToFile(file + "MI")
                else:
                    compresserTrainingTimeAvg = np.mean(np.array(results)[:, 0])
                    compressionTestTimeAvg = np.mean(np.array(results)[:, 1])
                    AccuracyAvg = np.mean(np.hstack(np.array(results)[:, 2]))
                    stddevAccuracy = np.std(np.hstack(np.array(results)[:, 2]))
                    classifyTimeTrainAvg = np.mean(np.mean(np.array(results)[:, 3]))
                    classifyTimeTestAvg = np.mean(np.mean(np.array(results)[:, 4]))
                    Vrtces=np.array(results)[:, 5]
                    if runOurExperiment:
                        retval = ExperimentResult(compresserTrainingTimeAvg, compressionTestTimeAvg, classifyTimeTrainAvg,
                                                  classifyTimeTestAvg, AccuracyAvg, stddevAccuracy,rhoTime, np.array(Vrtces))

                        if type(retval) != ExperimentResult:
                            raise ("Experimental result expected.  Didn't get it.")
                        if ClassifierOnly:
                            retval.writeToFile(RESULT_FILENAME[:-4]+ "-baseline.pkl")
                        else:
                            retval.writeToFile(RESULT_FILENAME + str(ratio)+"CompressionRatio-wRho"+str(rhoRatio*100)+"%"+"WO-RhoPrime")#+"-wRhoPrime"+str(rhoPrimeRatio*100)+"%")
                        print("Results: " + str(retval))
                    if runVikasExperiment:
                        rhoTime=0
                        retval = ExperimentResult(compresserTrainingTimeAvg, compressionTestTimeAvg, classifyTimeTrainAvg,
                                                  classifyTimeTestAvg, AccuracyAvg, stddevAccuracy,rhoTime, np.array(Vrtces))

                        retval.writeToFile(RESULT_FILENAME[:-4] + "vikas" + str(ratio)+"CompressionRatio.pkl")

