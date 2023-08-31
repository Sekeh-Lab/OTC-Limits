"""
=============================================================
Classification on datasets compressed via optimal transport. 
=============================================================

"""


from main import *
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


from P1_Main import * 


def find_rho_0(F):    
    degrees = np.sum(F, axis=0) 
    sum_of_degrees = np.sum(F)
    num_vertices = F.shape[1]
    rho_0 =  np.array([degree/sum_of_degrees for degree in degrees])
    return rho_0



#
#  rho_0 -- The initial distribution on the nodes.
#  F -- The incidence matrix of the graph.
#  c -- The cost matrix.
#  lambda_1 -- Some number.  I haven't figured out what it is yet.  (AM).
#  K -- Some number.
#
#Sensitivity injection
# def CompressGraph(sensitiveNodes,rho_0, F, c, lambda_1, K, alpha_eps=0.1, alpha_t=0.1, alpha_zeta=0.1, iter_max=100, tol=1e-4): #alphas = 0.01
#####
def CompressGraph(rho_0, F, c, lambda_1, K, alpha_eps=0.1, alpha_t=0.1, alpha_zeta=0.1, iter_max=100, tol=1e-4): #alphas = 0.01
    
    #Initialize epsilon, t, and rho_1
    num_vertices = F.shape[1]
    epsilon =  K*np.ones(num_vertices)/num_vertices  
    t  =  np.zeros(num_vertices) 
    zeta = 0
    
    
    #Obtain the epsilon solution from the relaxed Boolean program
    epsilon_hat = SolveDual(rho_0, F, c, K, lambda_1, epsilon, t, zeta, 
                           alpha_eps=alpha_eps, alpha_t=alpha_t, alpha_zeta=alpha_zeta, iter_max=iter_max, tol=tol)
    
    
    #### perform rounding
    ro_1 = epsilon_hat.copy()

    # Added to original:#this is the one
    # ro_1[sensitiveNodes == 1] = max(ro_1)
    #

    ro_1_min_indices =  np.argpartition(ro_1, len(ro_1)-K)[:len(ro_1)-K]
    ro_1_rounded = np.ones(len(ro_1)) 
    ro_1_rounded[ro_1_min_indices] = 0
    ro_1_rounded[ro_1==0] = 0

    ######## Added to original:
    ######## ro_1_rounded[sensitiveNodes == 1] = 1

    return ro_1_rounded 


def FormIncidentMatrix(edges, vertices):
    n_edges = len(edges)
    n_vertices = len(vertices)
    F = np.zeros((n_edges, n_vertices), dtype=int)
    for edge_number, edge in enumerate(edges):
        (i, j) = edge
        F[edge_number][i] = 1
        F[edge_number][j] = 1
    return F


def CompressedSubgraph(rho_1, edges, vertices, vertex_labels):
    d_v = len(vertices)
    d_e = len(edges)
    remaining_vertices = [vertices[i] for i in range(d_v) if rho_1[i]]
    remaining_vertex_labels = [vertex_labels[i] for i in range(d_v) if rho_1[i]]
    remaining_edges = [edges[e] for e in range(d_e) if rho_1[edges[e][0]] and rho_1[edges[e][1]]]
    return remaining_vertices, remaining_vertex_labels, remaining_edges


def ComputeCost(edges, vertex_labels, same_label_cost=1, distinct_label_cost=10, isDiscrete=True):
    costs = np.zeros(len(edges))
    for edgeID, edge in enumerate(edges):
        (i, j) = edge
        if not isDiscrete:
            if vertex_labels[i] == vertex_labels[j]:
                costs[edgeID] = same_label_cost
            else:
                costs[edgeID] = distinct_label_cost
        else:
            costs[edgeID] = abs(vertex_labels[i] - vertex_labels[j])
    return costs


#
# graph has the following format:
# (edges, vertices, edge_labels).
# edges is a set of ordered pairs of vertices.
# vertices is a dictionary mapping vertex names to vertex labels.
# edge_labels is a dictionary mapping ordered pairs of vertices to edge labels.
# This is the Grakel form.
#
def GetGraphInfo(graph):
    edges = list(graph[0])
    edges = [edge for edge in edges if edge[0] < edge[1]] #each edge is repeated twice, so we remove one
    edges = sorted(edges, key = lambda x: (x[0], x[1])) 
    vertices = sorted(list(graph[1].keys()))
    vertex_labels = [graph[1][vertex] for vertex in vertices]
    edge_labels = [graph[2][edge] if edge in graph[2].keys() else -1 for edge in edges]
    min_vertex = vertices[0]
    vertices = [vertex-min_vertex for vertex in vertices]
    edges = [(edge[0]-min_vertex, edge[1]-min_vertex) for edge in edges]
    return vertices, vertex_labels, edges, edge_labels


#
# vertices --
#
def ExpressInGrakelForm(list_of_graphs):  
    n_graphs = len(list_of_graphs)
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
#
#
def reshapeCostMatrix(costMatrix):
    retval = []
    numRows, numCols = costMatrix.shape
    for i in range(numRows):
        for j in range(i+1, numRows):
            retval.append((costMatrix[i,j]))
    retval = np.array(retval)
    return(retval)


#
# Given a grakel graph G, compress the given graph with the given parameters.
# The output of this function is list-type object whose v'th element corresponds
# to vertex v.  It is either 0 or 1.
#
#Sensitivity injection
# def CompressGraphWithParams(sensitiveNodes,G, compressionRatio, initDistribution, costMatrix, lambda_1=0.1, alpha_eps=1, alpha_t=1, alpha_zeta=1, iter_max=100, tol=1e-4):
####
def CompressGraphWithParams(G, compressionRatio, initDistribution, costMatrix, lambda_1=0.1, alpha_eps=1, alpha_t=1, alpha_zeta=1, iter_max=100, tol=1e-4):
    vertices, vertex_labels, edges, edge_labels = GetGraphInfo(G);
    print("Number of edges: " + str(len(edges)))
    print("Number of vertices: " + str(len(vertices)))
    F = FormIncidentMatrix(edges, vertices);
    print("F.size:" + str(F.shape))
    K = int(np.ceil(compressionRatio*len(vertices)));
    if (len(costMatrix.shape) > 1):
        costMatrix = reshapeCostMatrix(costMatrix)
    print("costMatrix shape: " + str(costMatrix.shape))
    print("Cost matrix: " + str(costMatrix))
    print("Initial distribution: " + str(initDistribution))
    #Sensitivity injection
    # outputDistRounded = CompressGraph(sensitiveNodes,initDistribution, F, costMatrix, lambda_1, K, iter_max = iter_max, alpha_eps=alpha_eps, alpha_t=alpha_t, alpha_zeta=alpha_zeta, tol=tol);
    ####
    outputDistRounded = CompressGraph(initDistribution, F, costMatrix, lambda_1, K, iter_max = iter_max, alpha_eps=alpha_eps, alpha_t=alpha_t, alpha_zeta=alpha_zeta, tol=tol);
    return(outputDistRounded);



#
# This is the function that we want to use.
#
# dataset -- Currently, it's a string, which is used to download a dataset.  This is not what we want.
#
# ratio -- The compression ratio. 
#
def CompressAllGraphs(dataset, ratio = 0.5, lambda_1 = 0.1, c_same=1, c_distinct=2, nodes = None, alpha_eps=1, alpha_t=1, alpha_zeta=1, iter_max=100, tol=1e-4):
    #AM: TODO: We need to modify here.  dataset is a string, which is used to download a dataset.  This is not what we want.
    graphs_info = datasets.fetch_dataset(dataset, verbose=False)  
    graphs, graph_labels = graphs_info.data, graphs_info.target 
    graph_info = []
    for graphID, graph in enumerate(graphs):
        vertices, vertex_labels, edges, edge_labels = GetGraphInfo(graph)
        F = FormIncidentMatrix(edges, vertices)
        c = ComputeCost(edges, vertex_labels, same_label_cost=c_same, distinct_label_cost=c_distinct)
        rho_0 = find_rho_0(F)
        
        if nodes is None:
            K = int(np.ceil(ratio*len(vertices)))
        else:
            K = nodes[graphID]
            
        #Obtain the support of the compressed distribution
        rho_1_rounded = CompressGraph(rho_0, F, c, lambda_1, K, iter_max = iter_max, alpha_eps=alpha_eps, alpha_t=alpha_t, alpha_zeta=alpha_zeta, tol=tol) 
        
        #Get the compressed subgraph
        r_vertices, r_vlabels, r_edges = CompressedSubgraph(rho_1_rounded, edges, vertices, vertex_labels)
        graph_info.append([r_vertices, r_vlabels, r_edges])
        
    graphs_in_grakel_form = ExpressInGrakelForm(graph_info)

    return graphs_in_grakel_form, graph_labels


###############
#
# Input: dataset is a Dataset object (from GenerateDatasets.py).
# Returns a python list of OTParameters objects.
#
def getVikasOTParameters(dataset, c_same=1, c_distinct=2, isDiscrete=True):
    retval = [];
    for (G, features, _) in dataset:
        # Create the OT parameters for graph G.
        # Convert G and its features to Grakel form.
        graph = nxToGrakel(G, features);
        #print("graph:" + str(graph))
        print(type(graph))
        print(len(graph))
        #print(graph[0])
        #print(graph[1])
        #print(graph[2])
        edges = graph[1]
        print("Number of edges:" + str(len(edges)))
        vertices, vertex_labels, edges, edge_labels = GetGraphInfo(graph)
        F = FormIncidentMatrix(edges, vertices)
        c = ComputeCost(edges, vertex_labels, same_label_cost=c_same, distinct_label_cost=c_distinct, isDiscrete=isDiscrete)
        rho_0 = find_rho_0(F)
        otp = OTParameters(rho_0, c)
        retval.append(otp)
    return(retval)

#
# Unlike our method, this OT compression method spits out a new dataset.
# The new dataset will have different graphs and features.
#
def VikasOTCompress(otParameters, dataset, compressionRatio):
    nodeSubsets = []
    for idx, dataPt in enumerate(dataset):
        print(idx)
        otParam = otParameters[idx]
        (G, features, classLabel) = dataPt
        Ggrakel = nxToGrakel(G, features)
        initDist, costMatrix = otParam.getParameters()
        #Use CompressGraphWithParams.
        roundedDist = CompressGraphWithParams(Ggrakel, compressionRatio, initDist, costMatrix)
        # Append the set of nodes corresponding to the support of roundedDist.
        nodeSubset = set()
        for nodeIndex in range(len(roundedDist)):
            if roundedDist[nodeIndex] == 1:
                nodeSubset.add(nodeIndex)
        nodeSubsets.append(nodeSubset)
    newDataset = dataset.projectSubsets(nodeSubsets)
    return(newDataset,nodeSubsets)


##################################################################
#
# Vikas's main code.
#
if __name__ == "__main__":
    #
    ##### Our datasets
    data = ['DHFR', 'MSRC_21C', 'MSRC_9', 'BZR_MD', 'Mutagenicity']

    ###### Hyperparameters
    lambda_1 = 1
    ratio = 0.5     #### desired amount of compression for each graph. we won't be using it for our purposes since we compress to as many nodes as REC
    alpha = 0.1
    c_same_base = 0.01           ##### same label cost
    c_distinct_base = 0.02       ##### distinct label cost
    max_iterations = 25

    output_folder = 'OTCCompressedResults5runs'+str(ratio)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    ##### We need this since we compress graphs to same number of nodes as the REC method.
    spectral_folder = os.path.join('MultipleRuns', 'SpectralCompressedResults5runs'+ str(ratio))



    #AM: dataset is a string.
    for dataset in data:
        #try:
            print("\n \n Dataset :", dataset)
            print("----------------------------------")

            accuracies_all = np.zeros((5, 7))
            times_all = np.zeros((5, 7))

            for run in range(5):

                nodes_filename = os.path.join(spectral_folder, 'n_nodes_' + dataset.lower() + '_' +  str(run) + '.csv')
                df = pd.read_csv(nodes_filename, header=None)
                nodes = df.values.flatten()

                #AM: df is a Pandas csv-type object.

                for c_same, c_distinct in [(c_same_base, c_distinct_base)]:

                    time_compress_init = time()

                    graphs, graph_labels = CompressAllGraphs(dataset, lambda_1 = lambda_1, ratio=ratio, c_same=c_same, c_distinct=c_distinct, nodes = nodes,                                              iter_max=25, alpha_eps=alpha, alpha_t=alpha, alpha_zeta=alpha,  tol=1e-2)
                    time_compress_final = time()

                    time_taken = time_compress_final - time_compress_init

                    print('Total time elapsed in compression ', time_taken)


                    output_file = os.path.join(output_folder, dataset.lower() + str(run))

                    with open(output_file, 'w') as f:
                        for train_index, train_size in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
                            avg_acc, std_acc, average_time = ComputeAccuracy(graphs, graph_labels, test_size=round(1-train_size,1),
                                                                                         n_runs=5, cv_folds = 5)

                            f.write('Train size ' +  str(train_size) + ', accuracy: ' + str(avg_acc) + ' +/- ' + str(std_acc) + ', compress time: ' + str(time_taken) + '\n')
                            print('Train size: ', train_size)
                            print('Accuracy: ', avg_acc, ' +/- ', std_acc)

                            accuracies_all[run, train_index] = avg_acc
                            times_all[run, train_index] = time_taken

            mean_accuracy = np.mean(accuracies_all, axis=0)
            mean_time = np.mean(times_all, axis=0)

            mean_accuracy = np.mean(accuracies_all, axis=0)
            std_accuracy = np.std(accuracies_all, axis=0)
            mean_time = np.mean(times_all, axis=0)

            overall_output_file = os.path.join(output_folder, dataset.lower() + '_all')

            with open(overall_output_file, 'w') as f:
                for train_index, train_size in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
                    avg_acc, std_acc, average_time = mean_accuracy[train_index], std_accuracy[train_index], mean_time[train_index]

                    f.write('Train size ' +  str(train_size) + ', accuracy: ' + str(avg_acc) + ' +/- ' + str(std_acc) + ', compress time: ' + str(average_time) + '\n')


            print('Mean Accuracies', mean_accuracy)
            print('Mean times', mean_time)

            print("***********************************")
        #except:
        #    continue

