from __future__ import print_function

import grakel

print(__doc__)

from time import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from grakel import datasets
from grakel import GraphKernel
from grakel import Graph
from grakel.kernels import PropagationAttr
from grakel.kernels import GraphHopper
import numpy as np


#
# Convert the list G of graphs to the appropriate grakel format
# for use with the GraphHopper kernel.
#
def toEdgeDictionary(G):
    retval = []
    for graph in G:
        print("Graph to be converted to grakel: " + str(graph))
        edgeDictionary = {}

        for v in graph[1].keys():
            edgeDictionary[v] = []

        for (v, w) in graph[0]:
            #if not v in edgeDictionary:
            #    edgeDictionary[v] = []
            edgeDictionary[v].append(w)
        print("Edge dictionary: " + str(edgeDictionary))
        graph = Graph(edgeDictionary, node_labels=graph[1], edge_labels=graph[2])
        print("graph: " + str(graph))
        retval.append(graph)
    return retval


#
# G is some indexable object containing grakel graphs.
# y is some indexable object containing class labels.
# 
#
def ComputeAccuracy(G, y, test_size=0.5, n_runs=5, cv_folds = 5, continuousFeatures = False):
    accuracy = np.zeros(n_runs)
    timesTrain = np.zeros(n_runs)
    timesTest = np.zeros(n_runs)

    if continuousFeatures:
        G = toEdgeDictionary(G)

    # if continuousFeatures:
    #     G_test = toEdgeDictionary(G_test)

    for run in range(n_runs):
        
        # Split graph data and labels into train and test
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=test_size, random_state=42*run)
        print(G_train[0])
        
        # Initialise a weisfeiler kernel, with a dirac base_kernel
        if not continuousFeatures:
            gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5},
                             {"name": "subtree_wl"}], normalize=True)   #, Nystroem=20
        else:
            print("Continuous features!!!!!!!!!!!!")
            #gk = PropagationAttr(normalize=True)
            #gk = GraphHopper(normalize=True, kernel_type="gaussian")
            gk = GraphKernel(kernel=[{"name": "graph_hopper", "kernel_type": "gaussian"}], normalize=True)
            #gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5},
            #                 {"name": "subtree_wl"}], normalize=True)   #, Nystroem=20
            #gk = GraphKernel(kernel=[{"name": "multiscale_laplacian", "n_samples": 10}], normalize=True)
            #TODO: FINISH ME!  Use the multiscale laplacian kernel

        print("Start the cross validation")
        #Cross-validate to estimate the best hyperparameters
        svc = svm.SVC(kernel='precomputed', random_state = run)
        param_grid = {'svc__C': [0.1, 1, 10]}
        estimator = Pipeline([('kernel', gk), ('svc',svc)])
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv= cv_folds)
        print("In the middle of the cross validation")
        #print(clf.get_params().keys())
        clf.fit(G_train, y_train)
        print("Done fitting")
        C_best = clf.best_estimator_.named_steps['svc'].C
        #print('C_best ', C_best)
        print("Done with cross validation")

        #Start the clock
        start = time()

        # Compute the kernel matrices
        K_train = gk.fit_transform(G_train)
        #used to be:
        # end = time()
        # timesTrain[run] = end - start         #total train time

        # Train an SVM
        clf = svm.SVC(kernel='precomputed', C=C_best)
        clf.fit(K_train, y_train)
        end = time()
        timesTrain[run] = end - start         #total train time

        # Predict on test data and compute accuracy
        start=time()
        K_test = gk.transform(G_test)
        y_pred = clf.predict(K_test)
        # acc = accuracy_score(y_test, y_pred)
        accuracy[run] = accuracy_score(y_test, y_pred)
        end=time()
        timesTest[run]=end-start
        #######
    # average_acc = np.mean(accuracy)           #mean accuracy
    # std_acc =  np.std(accuracy)               #standard deviation
    # average_timeTrain = np.mean(timesTrain)
    # average_timeTest = np.mean(timesTest)

    return accuracy,timesTrain,timesTest#, std_acc, average_timeTrain,average_timeTest


