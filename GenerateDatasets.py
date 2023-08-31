import pickle5 as pickle
import random
import numpy as np
import networkx as nx
import csv
import sklearn
from grakel.utils import graph_from_networkx;
import matplotlib.pyplot as plt
from random import randint

numRuns = 5     # Number of times we perform the whole experiment on a synthetic dataset.
gaussianFeatures = True
# datasetType = "synthetic"
# datasetType = "regression"
#datasetType = "twitter"
# datasetType = "brain"
# datasetType = "syntheticNew"
datasetType = "MNIST"
# datasetType = "CIFAR"
# datasetType = "NYC"
# datasetType = "MiniImageNet"
#



if datasetType == "synthetic":
    if not gaussianFeatures:
        TRAINING_FILENAME = "synthetic-datasets/train-dataset.pkl"
        TEST_FILENAME = "synthetic-datasets/test-dataset.pkl"
        RESULT_FILENAME = "results/result-discrete.pkl"
    else:
        TRAINING_FILENAME = "synthetic-datasets/train-final-dataset-gaussian-dependent-100sample-50nodes-noiseadded-to-feature.pkl"
        TEST_FILENAME = "synthetic-datasets/test-final-dataset-gaussian-dependent-100sample-50nodes-noiseadded-to-feature.pkl"
        RESULT_FILENAME = "results/result-final-dataset-gaussian-dependent-100sample-50nodes-noiseadded-to-feature.pkl"
    NODE_SIZE=50
    featDim = 1
    trainingDataSize = 100;

# Twitter data.
if datasetType == "twitter":
    gaussianFeatures=False
    RAW_FILENAME = "real-datasets/TWITTER-Real-Graph-Partial.nel"
    TRAINING_FILENAME = "real-datasets/twitter-training.pkl"
    TEST_FILENAME = "real-datasets/twitter-testing.pkl"
    RESULT_FILENAME = "results/result-twitter.pkl"

# Brain data
if datasetType == "brain":
    gaussianFeatures = False
    RAW_FILENAME = "real-datasets/OHSU.nel"
    TRAINING_FILENAME = "real-datasets/brain-training.pkl"
    TEST_FILENAME = "real-datasets/brain-testing.pkl"
    RESULT_FILENAME = "results/result-brain.pkl"

if datasetType == "NYC":
    gaussianFeatures = False
    TRAINING_FILENAME = "real-datasets/NYC-training-100-5-1.pkl"#NYC-training-25.pkl"#NYC-training-105-1-new-nodeNum10.pkl"#NYC-training-100-5-1.pkl"#
    TEST_FILENAME = "real-datasets/NYC-testing-100-5-1.pkl"#NYC-testing-105-1-new-nodeNum10.pkl"##
    RESULT_FILENAME = "results/result-NYC-100-5-1vikas.pkl"
    NODE_SIZE=100



# syntheticNew data
if datasetType == "syntheticNew":
    gaussianFeatures = False
    # RAW_FILENAME = "real-datasets/syntheticNew.nel"
    TRAINING_FILENAME = "real-datasets/syntheticNew-training.pkl"
    TEST_FILENAME = "real-datasets/syntheticNew-testing.pkl"
    RESULT_FILENAME = "results/result-syntheticNew.pkl"


# MNIST data
if datasetType == "MNIST":
    gaussianFeatures = False
    # RAW_FILENAME = "real-datasets/syntheticNew.nel"
    TRAINING_FILENAME = "real-datasets/MNIST-training.pkl"
    TEST_FILENAME = "real-datasets/MNIST-testing.pkl"
    RESULT_FILENAME = "results/test-4-1-result-MNIST.pkl"
    NODE_SIZE=100
    MNIST_SIZE = 28  # MNIST,image axis size

# CIFAR data
if datasetType == "CIFAR":
    gaussianFeatures = False
    # RAW_FILENAME = "real-datasets/syntheticNew.nel"
    TRAINING_FILENAME = "real-datasets/CIFAR-training.pkl"
    TEST_FILENAME = "real-datasets/CIFAR-testing.pkl"
    RESULT_FILENAME = "results/1000Samples-2Classes-500Nodes-result-CIFAR-4-1train-test.pkl"
    NODE_SIZE=500
    MNIST_SIZE = 32 #CIFAR 10,image axis size

# MiniImageNet data
if datasetType == "MiniImageNet":
    gaussianFeatures = False
    # RAW_FILENAME = "real-datasets/syntheticNew.nel"
    TRAINING_FILENAME = "real-datasets/MiniImageNet-training.pkl"
    TEST_FILENAME = "real-datasets/MiniImageNet-testing.pkl"
    RESULT_FILENAME = "results/2Classes-300Nodes-result-MiniImageNet-4-1train-test.pkl"
    NODE_SIZE=250
    MNIST_SIZE = 84 #MiniImageNet,image axis size

# aspirin data
if datasetType == "regression":
    gaussianFeatures = False
    # RAW_FILENAME = "real-datasets/syntheticNew.nel"
    TRAINING_FILENAME = "real-datasets/aspirin-training.pkl"
    TEST_FILENAME = "real-datasets/aspirin-testing.pkl"
    RESULT_FILENAME = "results/result-aspirin-baseline.pkl"




#Added to the original one
def OurSubgraph(Graph,nodeSubset):
    nodeSubset.sort()
    G = nx.Graph()
    [G.add_node(i) for i in range(len(nodeSubset))]


    # for item in np.array(Graph.edges()):#[np.array(Graph.edges())[:, 0] == nodeSubset]:
    #     G.add_edge(np.where(item[0]==nodeSubset),np.where(item[1]==nodeSubset),weight=1)

    [G.add_edge(np.where(item[0]==nodeSubset)[0][0],np.where(item[1]==nodeSubset)[0][0],weight=1) if ((item[0] in nodeSubset)& (item[1] in nodeSubset)) else 0 for item in np.array(Graph.edges())]
    return G
#####


#####
# Create a synthetic dataset for testing the optimal transport framework.
#
class Dataset:
    #
    # A dataset consists of a list of (featured graph, label) pairs.
    #

    #
    #
    #
    def __init__(self, pointSet=[]):
        self.points = [];
        self.labels = set();
        for pt in pointSet:
            self.add(pt[0], pt[1], pt[2])

    #
    #
    #
    def __iter__(self):
        self.iter_index = 0;
        return (self);

    #
    #
    #
    def __next__(self):
        if (self.iter_index >= len(self.points)):
            raise StopIteration;
        retval = self.points[self.iter_index];
        # print("Self.iter_index: " + str(self.iter_index))
        # print("Retval: " + str(retval))
        self.iter_index += 1;
        return (retval);

    #
    #
    #
    def __str__(self):
        # TODO: FINISH ME!
        print("Hi, there!");

    #
    # Convert the given dataset to a Grakel dataset and return
    # the result.
    #
    def toGrakel(self):

        # TODO: FINISH ME!
        print("Hi, there!");

    #
    #
    #
    def size(self):
        return (len(self.points));

    #
    #
    #
    def getDatasetSize(self):
        return (self.size());

    #
    # graph is a networkx graph.
    #
    # It is expected that features is a numpy array with
    # shape # of graph nodes x size of a feature vector x 1.
    #
    def add(self, graph, features, label):
        self.points.append((graph, features, label));
        self.labels.add(label);
        self.graphSize = len(graph.nodes);
        self.featureSize = features.shape;

    #
    #
    #
    def get(self, index):
        return (self.points[index]);

        #

    # Get the graphs from this dataset.  They will be a list of graphs,
    # each one encoded according to the toGrakel  and attributed flag.
    # If classLabel is not None, only return those graphs with class label equal
    # to classLabel.
    #
    def getGraphs(self, toGrakel=True, attributed=True, classLabel=None, flatten=True):
        retval = []
        for (G, features, label) in self:
            if (classLabel == None or label == classLabel):
                featureArray = features if attributed else None;
                print("Feature array: " + str(featureArray))
                graph = nxToGrakel(G, featureArray, flatten)
                retval.append(graph)
        return retval

    #
    # Get a numpy array containing all adjacency matrices.
    # Shape is # of samples x number of vtces x number of vtces.
    #
    def getAdjacencies(self, classLabel=None):
        dimention=self.graphSize
        adjacencies = [];
        for (G, _, label) in self:
            # Added to the original one
            adj = np.zeros(shape=(dimention, dimention))
            for (i, j) in np.array(G.edges):
                adj[i-1][j-1]=1
                adj[j-1][i-1]=1
            #
            if (label == classLabel or classLabel == None):
                adjacencies.append(adj);

        retval = np.stack(adjacencies, 0);
        return (retval);

    #
    #
    #
    def getClassLabels(self):
        labels = [];
        for (_, _, label) in self:
            labels.append(label);
        labels = np.array(labels);
        return (labels);

    #
    #
    #
    def getNumClasses(self):
        return (len(self.labels));

    #
    #
    #
    def getFeatureSize(self):
        return (self.featureSize);

    #
    # Get features of this dataset as a numpy array.
    # # of samples x # of vertices x size of a feature.
    #
    def getFeatures(self):
        features = [];
        for (_, feature, _) in self:
            features.append(feature);
        retval = np.stack(features);
        return (retval);

    #
    #
    #
    def getGraphSize(self):
        return (self.graphSize);

    #
    # Given a test set size, create numFolds (train, test) pairs
    # randomly.
    #
    def getTrainTest(self, testSetSize, numFolds=5):
        trains = []
        tests = []
        # for j in range(numFolds):
        train, test = sklearn.model_selection.train_test_split(self.points, random_state=42, test_size=testSetSize)
        train = Dataset(train)
        test = Dataset(test)
        trains.append(train)
        tests.append(test)
        print("Done getting the train and test: " + str(len(trains)))
        return(train, test)

    #
    # Given a python set object encoding a subset of vertices,
    # return a dataset whose graphs are the subgraphs induced by
    # the given node subset.  Note that we DO NOT relabel the graphs.
    #



    def project(self, nodeSubset):
        nodeSubset = list(nodeSubset)
        #print("Node subset: " + str(nodeSubset))
        retval = Dataset();

        for (G, features, classLabel) in self:
            print("Original features shape: " + str(features.shape))
            featProj = features[nodeSubset];# Added to the original one: if multi dim featured :  features[nodeSubset,:]
            print("Projected features shape: " + str(featProj.shape))
            H = OurSubgraph(G,nodeSubset);
            print("Number of nodes in H: " + str(len(H.nodes)))
            print("Features: " + str(featProj))
            retval.add(H, featProj, classLabel);
            # G.remove_nodes_from([n for n in G if n not in set(nodeSubset)])
        return (retval);


    #
    # Given a list of python set objects encoding subsets of vertices,
    # return a dataset whose graphs are the subgraphs induced
    # by the given node subsets.
    #
    def projectSubsets(self, nodeSubsets):
        retval = Dataset()
        for idx, (G, features, classLabel) in enumerate(self):
            featProj = features[list(nodeSubsets[idx])]
            H = OurSubgraph(G,list(nodeSubsets[idx]))
            retval.add(H, featProj, classLabel)
        return(retval)

    #
    # Return a dataset having only the given subset of data points.
    #
    def subDataset(self, dataSubset):
        retval = Dataset()
        for idx in dataSubset:
            item = self.get(idx)
            retval.add(*item)
        return(retval)



###############################################################
#
# Given a filename for a .nel file, open it and parse it into a
# dataset.
#
def nelFileToDataset(nelFilename):
    dataset = Dataset()
    nodeSet = set()
    graphInfo = []
    numPoints = 0
    with open(nelFilename, "r", newline="") as fd:
        reader = csv.reader(fd, delimiter=' ', quotechar='|')


        GnodeIDLst = []       # List of node IDs for this graph.
        GedgeList = []

        for row in reader:
            #Row is a list of strings.
            if len(row) == 0:
                continue
            if row[0] == 'n':
                nodeID = row[2]
                GnodeIDLst.append(nodeID)
                nodeSet.add(nodeID)

            if row[0] == 'e':
                GedgeList.append((int(row[1])-1, int(row[2])-1, float(row[3])))
                #edge = (nodeLst[int(row[1])-1], nodeLst[int(row[2])-1])
                #weight = float(row[3])
                #G.add_edge(edge[0], edge[1], weight=weight)
            if row[0] == 'g':
                print("Read graph " + str(row[2]))
            if row[0] == 'x':
                print("Class: " + str(row[1]))
                classLabel =int((1 + int(row[1]))/2)
                graphInfo.append( (GnodeIDLst, GedgeList, classLabel))
                numPoints += 1
                GnodeIDLst = []
                GedgeList = []

    nodeDict = {}
    for idx, v in enumerate(nodeSet):
        nodeDict[v] = idx

    for gInfo in graphInfo:
        GnodeIDLst, GedgeList, classLabel = gInfo
        G = nx.Graph()
        for index, v in enumerate(nodeSet):
            G.add_node(index)
        nodeFeatures = np.zeros((len(nodeSet), 1))
        for edge in GedgeList:
            vIdx, wIdx, weight = edge
            vID = nodeDict[GnodeIDLst[vIdx]]
            wID = nodeDict[GnodeIDLst[wIdx]]
            G.add_edge(vID, wID, weight=weight)
        dataset.add(G, nodeFeatures, classLabel)

#    for idx, (G, nodeFeatures, classLabel) in enumerate(dataset):
#        print("Original number of nodes: " + str(G.number_of_nodes()))
#        for v in nodeSet:
#            if not (v in G.nodes):
#                G.add_node(v)
#        nodeFeatures = np.zeros((len(nodeSet), 1))
#        newDataset.add(G, nodeFeatures, classLabel)
#        print("Number of nodes: " + str(G.number_of_nodes()) + ", " + str(nodeFeatures.shape))
    print("Dataset: " + str(dataset.getDatasetSize()))
    print("Number of points:" + str(numPoints))
    print("Num nodes: " + str(len(nodeSet)))
    return(dataset)

###############################################################
#Added to the original one

def txtFileToDataset(Adjacency,Indicator,Label,Attribute):
    dataset = Dataset()
    graphInfo = []
    numPoints = 0
    GnodeIDLst = []       # List of node IDs for this graph.
    GedgeList =[]
    GraphNodesNum=21
    edges=np.zeros((np.shape(Adjacency)[0],2))#+1
    edges=Adjacency
    # nodeSet=range(1,300000)
    print("about to generate arrays")

    for i in range(np.max(Indicator)):
        GedgeList=edges[(edges[:,0]>(i*GraphNodesNum)) & (edges[:,0]<=((i+1)*GraphNodesNum))]-(i*GraphNodesNum)#-i*100
        newGedgeList=np.zeros((np.shape(GedgeList)[0],3),dtype=int)+1
        newGedgeList[:,:-1]=GedgeList
        GnodeIDLst=np.array(range(1,GraphNodesNum+1))
        classLabel=Label[i]
        graphInfo.append((GnodeIDLst, newGedgeList, classLabel))
    iter=1
    print("about to start generating graphs")

    for gInfo in graphInfo:
        GnodeIDLst, GedgeList, classLabel = gInfo
        G = nx.Graph()
        [G.add_node(i) for i in range(1,GraphNodesNum+1)]
        nodeFeatures =Attribute[iter-1:iter+GraphNodesNum-1]#######needs to get compatible with multi dim featured graphs
        for edge in GedgeList:
            vIdx, wIdx, weight = edge
            vIdx=vIdx
            wIdx=wIdx
            G.add_edge(vIdx, wIdx, weight=weight)
        dataset.add(G, np.array(nodeFeatures), classLabel)
        iter+=GraphNodesNum
    print("Dataset: " + str(dataset.getDatasetSize()))
    print("Number of points:" + str(numPoints))
    return(dataset)


#####

############



#
# This is an INTERNAL function.
#
def getGrakelForm(list_of_graphs):
    n_graphs = len(list_of_graphs)
    print(list_of_graphs[0])
    Grakel_form = []
    for graph in list_of_graphs:
        vertices, vertex_labels, edges = graph
        vert_dict = {}
        edges_dict = {}
        for vertexID, vertex in enumerate(vertices):
            vert_dict[vertex] = vertex_labels[vertexID]
        new_edges = [(j, i) for (i, j) in edges]
        edges = edges + new_edges
        Grakel_form.append([set(edges), vert_dict, edges_dict])
    return Grakel_form



#
# Given a networkx graph G, add features to the nodes of G.
# If features == None, set each feature to the empty string.
#
# This is an INTERNAL function.
#
def featurizeGraph(G, features=None, stringFeatures=False, flatten=True):
    #if type(features) == type(None):
    #    return(G);
    for idx, v in enumerate(G.nodes):
        print("idx, v: " + str((idx, v)))

        if type(features) == type(None):
            print("Missing features")
            G.nodes[v]["feature"] = ""
        else:
            print("Features were not missing for this vertex")
            if stringFeatures:
                G.nodes[v]["feature"] = str(features[idx])
            else:
                if not flatten:
                    G.nodes[v]["feature"] = features[idx]
                else:
                    G.nodes[v]["feature"] = features[idx,0]
        #print("NOTE: I've turned features into strings.")
        print("Feature: " + str(G.nodes[v]["feature"]))
    return(G);


#
# Convert the given networkx attributed graph to a Grakel graph.
#
# The output has the following form:
# # (edges, vertices, edge_labels).
# # edges is a set of ordered pairs of vertices.
# # vertices is a dictionary mapping vertex names to vertex labels.
# # edge_labels is a dictionary mapping ordered pairs of vertices to edge labels.
# # This is the Grakel form.
#
def nxToGrakel(G, features=None, flatten=True):
    Gprime = featurizeGraph(G, features, stringFeatures=not gaussianFeatures, flatten=flatten)

    edges = set(Gprime.edges)
    vertices = {}
    for node in G.nodes:
        vertices[node] = Gprime.nodes[node]['feature']
    edge_labels = {}
    for e in edges:
        edge_labels[e] = 1

    retval = (edges, vertices, edge_labels)
    return (retval);


#
# Read a dataset from the given file.  Return a Dataset object.
#

#Added to the original one
#Uncomment the comment below
#################

def readDataset(filename, run=-1):
    if run > -1:
        filename += str(run) + "-" + str(numRuns)
    retval = None;


    with open(filename, "rb") as handle:
        retval = pickle.load(handle)
    return (retval)


#
#
#
def writeDataset(dataset, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL);


#
# Create a graph with independent edges having probabilities given by
# edge_p_matrix, which is an n x n numpy array.  It should be symmetric,
# or at least lower triangular.
#
def generateIndependentEdgeGraph(edge_p_matrix):
    num_nodes = edge_p_matrix.shape[0]
    retval = nx.empty_graph(num_nodes)
    counter=0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (random.random() < edge_p_matrix[i,j]):
                #noise injection
                # num=randint(2,30)
                # if (i+j)%num==randint(0,6):
                #     counter+=1
                #     continue
                # #######
                retval.add_edge(i, j)
                retval.add_edge(j, i)
            else:
                pass
                # # noise injection
                # num = randint(2, 30)
                # if (i+j) % num == randint(0,6):
                #     counter+=1
                #     retval.add_edge(i, j)
                #     retval.add_edge(j, i)
                # #######

    return (retval);


#
# Given a numpy array of vertex features
# with shape # of vertices x size of feature,
# generate a networkx graph from the dot product model.
#
def generateDependentEdgeGraph(vertexFeatures):
    def sigmoid(x):
        return(1/(1+np.exp(-x)))

    print(vertexFeatures.shape)
    NODE_SIZE, featureSize = vertexFeatures.shape
    edgePMatrix = np.zeros((NODE_SIZE, NODE_SIZE))
    for i in range(NODE_SIZE):
        for j in range(i+1, NODE_SIZE):
            prob = sigmoid(np.dot(vertexFeatures[i,:], vertexFeatures[j,:]))
            edgePMatrix[i, j] = prob
            edgePMatrix[j, i] = prob
    print(edgePMatrix)
    retval = generateIndependentEdgeGraph(edgePMatrix)
    print("graph size: " + str(len(retval.nodes)))
    print("graph: " + str(retval.edges))
    return(retval)

##################################################################
# Create a simple synthetic dataset.
#
# numPoints -- Number of labeled data points.  Each data point is a
# graph, a vector of node features, and a label for the feature-endowed graph.
#
# classP -- A numpy array of probabilities over the classes. The size of this array is the number of
#   classes in the classification problem.
# NODE_SIZE -- Number of nodes in each graph.
# edgePvecs -- A numpy array ( (# of classes) x NODE_SIZE x NODE_SIZE) giving the probability of
#   a given edge, conditioned on the class label.
# xPvecs -- A numpy array ( (# of classes) x NODE_SIZE x (probability support size)), giving the conditional
#   distribution of the feature of the given node, conditioned on the class.
#
#
#
def createSyntheticDataset(numPoints, classP, edgePvecs, xPvecs,featDim, gaussian=False):
    dataset = Dataset();

    for j in range(0, numPoints):
        # Generate a class label
        label = 3;
        if (random.random() < classP):
            label = 0;
        if ((random.random() > classP)&(random.random() < 2*classP)):
            label = 1;
        if ((random.random() > 2*classP)&(random.random() < 3*classP)):
            label = 2;

        # Generate a graph.
        G = generateIndependentEdgeGraph(edgePvecs[label]);
        NODE_SIZE = len(G.nodes);

        # Generate the features.  features should be a numpy array
        # with shape # of nodes x 1.
        features = np.zeros((NODE_SIZE, featDim));
        for j in range(0, NODE_SIZE):
            if not gaussian:
                # Generate a random feature for node j.  We will use xpvec
                Xj = 0;
                if (random.random() < xPvecs[label][j]):
                    Xj = 1;
            else:
                # Generate a Gaussian feature for Xj.
                # mean is xPvecs[label, j, 0], standard deviation is xPvecs[label, j, 1].
                Xj = [np.random.normal(xPvecs[label, j, 0], xPvecs[label, j, 1], (1, 1)) for i in range(featDim)]

            features[j] = Xj;

        dataset.add(G, features, label);
    return (dataset)


####################################################################################
#
#
def createDependentEdgeDataset(numPoints, NODE_SIZE, classP, gaussianParams,featDim):
    dataset = Dataset()
    for j in range(0, numPoints):
        # Generate a class label
        label = 0;
        if (random.random() < classP):
            label = 1;
        features = np.zeros((NODE_SIZE, featDim));
        counter=0
        for j in range(0, NODE_SIZE):
            # Generate a Gaussian feature for Xj.
            # mean is xPvecs[label, j, 0], standard deviation is xPvecs[label, j, 1].
            Xj = [np.random.normal(gaussianParams[label, j, 0], gaussianParams[label, j, 1], (1, 1)) for i in range(featDim)]
            #noise injection
            num=randint(1,7)
            if j%num==0:
                counter+=1
                features[j-1] = (features[j-2]+features[j-1]+Xj)/3;
            ########
            features[j] = Xj;
        G = generateDependentEdgeGraph(features)
        dataset.add(G, features, label);
    return(dataset)


#
# Given an NEL filename, read the file and create training and test
# dataset object lists.
#
def createDatasetsFromNel(filename, testRatio = .3, numDatasets=5):
    dataset = nelFileToDataset(filename)
    testSize = int(dataset.getDatasetSize() * testRatio)
    print("test size: " + str(testSize))
    trains, tests = dataset.getTrainTest(testSize, numDatasets)
    print("MADE IT HERE")
    return(trains, tests)

######################################################################
#
# MAIN: Generates some datasets and writes them to files.
#
if __name__=="__main__":

    #TODO: Preprocessing for generation of the datasets.
    if datasetType == "synthetic":
        # Generate the dataset.
        testDataSize = int(trainingDataSize / 5);
        classP = 0.45;
        # NODE_SIZE = 200;

        # Generate the edge probabilities.
        edgePvecs =   np.empty((4, NODE_SIZE, NODE_SIZE));
        for i in range(NODE_SIZE):
            for j in range(NODE_SIZE):
                edgePvecs[0,i, j] = .4;
                edgePvecs[1, i, j] = .5;

        # Generate the feature parameters.
        if not gaussianFeatures:
            xPvecs = np.empty((2, NODE_SIZE, 1));
            for j in range(NODE_SIZE):
                xPvecs[0, j] = 1.0 - j/NODE_SIZE;
                xPvecs[1, j] = 0.0;
        else:
            xPvecs = np.empty((4, NODE_SIZE, 2))
            for j in range(NODE_SIZE):
                # Set the means.
                xPvecs[0, j, 0] = j/NODE_SIZE * 5        # Most informative nodes are later.
                xPvecs[1, j, 0] = j/NODE_SIZE * 4
                # Unit standard deviation.
                xPvecs[0, j, 1] = 1
                xPvecs[1, j, 1] = 1
    elif datasetType=="regression":
        Adjfilepath = "aspirin/aspirin_A.txt"
        Indfilepath = "aspirin/aspirin_graph_indicator.txt"
        lblfilepath = "aspirin/aspirin_graph_attributes.txt"
        Atrfilepath = "aspirin/aspirin_node_attributes.txt"
        Adjacency = np.loadtxt(Adjfilepath, delimiter=",", dtype=int)
        Indicator = np.loadtxt(Indfilepath, dtype=int)
        label = np.loadtxt(lblfilepath, dtype=float)
        Label=np.copy(label)

        Label=np.array([int(item) for item in Label])
        label = np.copy(Label)
        Attribute = np.loadtxt(Atrfilepath,delimiter=",", dtype=float)
        print("Loading done")
        Label[(label>=-406729)&(label<=-406708)]=4
        Label[(label>=-406735)&(label<=-406730)]=3
        Label[(label>=-406739)&(label<=-406736)]=2
        Label[(label>=-406743)&(label<=-406740)]=1
        Label[(label>=-406755)&(label<=-406744)]=0
        dataset = txtFileToDataset(Adjacency, Indicator, Label, Attribute)
        testsize = int(dataset.getDatasetSize() * 0.3)
        trainingDataset, testDataset = dataset.getTrainTest(testsize)

    elif datasetType == "twitter" or datasetType == "brain":
        print("Reading dataset: " + datasetType)
        trainingDatasets, testDatasets = createDatasetsFromNel(RAW_FILENAME)
        print("Made it here!")
        #TODO: Read the raw NEL file.  Convert parts of it to training and test datasets.


    #######
    for run in range(numRuns):
        if datasetType == "synthetic":
            if not gaussianFeatures:
                trainingDataset = createSyntheticDataset(trainingDataSize, classP, edgePvecs, xPvecs,featDim, gaussian=gaussianFeatures)
                testDataset = createSyntheticDataset(testDataSize, classP, edgePvecs, xPvecs,featDim, gaussian=gaussianFeatures)
            else:
                trainingDataset = createDependentEdgeDataset(trainingDataSize, NODE_SIZE, classP, xPvecs,featDim)
                testDataset = createDependentEdgeDataset(testDataSize, NODE_SIZE, classP, xPvecs,featDim)
        # If datasetType is "twitter" or "brain", just write the same dataset 5 times.
        elif datasetType == "brain" or datasetType=="twitter":
            trainingDataset = trainingDatasets[run]
            testDataset = testDatasets[run]

        print("Writing dataset #" + str(run) + "out of " + str(numRuns) + "...")
        writeDataset(trainingDataset, TRAINING_FILENAME[:19]+str(trainingDataSize)+"-SampleSize-"+str(NODE_SIZE)+"-NODE_SIZE-"+str(featDim)+"-featDim-"+TRAINING_FILENAME[19:] + str(run+1)+"-"+str(numRuns))
        writeDataset(testDataset, TEST_FILENAME[:19]+str(trainingDataSize)+"-SampleSize-"+str(NODE_SIZE)+"-NODE_SIZE-"+str(featDim)+"-featDim-"+TEST_FILENAME[19:] + str(run+1)+"-"+str(numRuns))
        print("Done writing training and test datasets to files.");
