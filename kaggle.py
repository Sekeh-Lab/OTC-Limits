import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray'  # grayscale looks better

"""
This code is largely take from M. Defferrard's Github
https://github.com/mdeff/cnn_graph/blob/master/nips2016/mnist.ipynb.
"""
import argparse
import numpy as np
import scipy.sparse as sp
from keras.datasets import mnist as m
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import random
import cifar as c
from GenerateDatasets import *;
from miniimagenettools.mini_imagenet_dataloader import MiniImageNetDataLoader
from miniimagenettools import mini_imagenet_generator as Mini


def load_data(k=8, Sample_Size=0.0,numRuns=5, random_state=None):# noise_level=0.0,
    """
    Loads the MNIST dataset and a K-NN graph to perform graph signal
    classification, as described by [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).
    The K-NN graph is statically determined from a regular grid of pixels using
    the 2d coordinates.

    The node features of each graph are the MNIST digits vectorized and rescaled
    to [0, 1].
    Two nodes are connected if they are neighbours according to the K-NN graph.
    Labels are the MNIST class associated to each sample.

    :param k: int, number of neighbours for each node;
    :param noise_level: fraction of edges to flip (from 0 to 1 and vice versa);

    :return:
        - X_train, y_train: training node features and labels;
        - X_val, y_val: validation node features and labels;
        - X_test, y_test: test node features and labels;
        - A: adjacency matrix of the grid;
    """
    A = _mnist_grid_graph(k)
    if random_state is not None:
        np.random.seed(random_state)
    As=[]
    nodenum=MNIST_SIZE**2
    for i in range(Sample_Size):#+int(Sample_Size/9)):
        noise_level=random.randint(15,100)
        noise_level=noise_level/nodenum**2
        Adj = _flip_random_edges(A, noise_level).astype(np.float32)
        As.append(Adj)
    #MNIST:
    if datasetType == "MNIST":

        (X_train, y_train), (X_test, y_test) = m.load_data()

        X_train, X_test = X_train / 255.0, X_test / 255.0
        X_train = X_train.reshape(-1, MNIST_SIZE ** 2)
        X_test = X_test.reshape(-1, MNIST_SIZE ** 2)

        # train_index = np.random.choice(len(X_train[(y_train == 0) | (y_train == 1)]), size=int((Sample_Size * 4) / 5),
        #                                replace=False)
        # X_train = X_train[(y_train == 0) | (y_train == 1)][train_index]
        # y_train = y_train[(y_train == 0) | (y_train == 1)][train_index]
        #
        # test_index = np.random.choice(len(X_test[(y_test == 0) | (y_test == 1)]), size=int(Sample_Size / 5),
        #                               replace=False)
        # X_test = X_test[(y_test == 0) | (y_test == 1)][test_index]
        # y_test = y_test[(y_test == 0) | (y_test == 1)][test_index]
        # np.save("mnist/X_train.npy", X_train)
        # np.save("mnist/X_test.npy", X_test)
        # np.save("mnist/y_train.npy", y_train)
        # np.save("mnist/y_test.npy", y_test)
        #
        # X_train = np.load("mnist/X_train.npy")
        # X_test = np.load("mnist/X_test.npy")
        # y_train = np.load("mnist/y_train.npy")
        # y_test = np.load("mnist/y_test.npy")

        X_train = X_train[(y_train == 0) | (y_train == 1)][0:int((Sample_Size * 4) / 5)]
        y_train = y_train[(y_train == 0) | (y_train == 1)][0:int((Sample_Size * 4) / 5)]
        X_test = X_test[(y_test == 0) | (y_test == 1)][0:int(Sample_Size / 5)]
        y_test = y_test[(y_test == 0) | (y_test == 1)][0:int(Sample_Size / 5)]

    #

    #CIFAR10
    if datasetType == "CIFAR":
        (X_train, y_train,X_test, y_test)=c.load_data()
        X_train=np.array(X_train)
        X_test=np.array(X_test)
        y_train=np.array(y_train)
        y_test=np.array(y_test)

        X_train, X_test = X_train / 255.0, X_test / 255.0
        X_train = X_train.reshape(-1, MNIST_SIZE ** 2)
        X_test = X_test.reshape(-1, MNIST_SIZE ** 2)
        train_index=np.random.choice(len(X_train[(y_train == 0) | (y_train == 1)]), size=int((Sample_Size*4)/5),
                             replace=False)
        X_train=X_train[(y_train == 0) | (y_train == 1)][train_index]
        y_train=y_train[(y_train == 0) | (y_train == 1)][train_index]
        
        test_index=np.random.choice(len(X_test[(y_test == 0) | (y_test == 1)]), size=int(Sample_Size/5),
                             replace=False)
        X_test=X_test[(y_test == 0) | (y_test == 1)][test_index]
        y_test=y_test[(y_test == 0) | (y_test == 1)][test_index]
        np.save("cifar-10/X_train.npy",X_train)
        np.save("cifar-10/X_test.npy",X_test)
        np.save("cifar-10/y_train.npy",y_train)
        np.save("cifar-10/y_test.npy",y_test)

        X_train=np.load("cifar-10/X_train.npy")
        X_test=np.load("cifar-10/X_test.npy")
        y_train=np.load("cifar-10/y_train.npy")
        y_test=np.load("cifar-10/y_test.npy")

        # X_train = X_train[(y_train == 0) | (y_train == 1)][0:int((Sample_Size * 5) / 6)]
        # y_train = y_train[(y_train == 0) | (y_train == 1)][0:int((Sample_Size * 5) / 6)]
        # X_test = X_test[(y_test == 0) | (y_test == 1)][0:int(Sample_Size / 6)]
        # y_test = y_test[(y_test == 0) | (y_test == 1)][0:int(Sample_Size / 6)]

    #


    #MiniImageNet
    if datasetType == "MiniImageNet":
        # parser = argparse.ArgumentParser(description='')
        # parser.add_argument('--tar_dir', type=str,default="MiniImageNet/ILSVRC2012_img_train.tar")
        # parser.add_argument('--imagenet_dir', type=str)
        # parser.add_argument('--image_resize', type=int, default=84)
        #
        # args = parser.parse_args()
        #
        # # MiniImagenet(root="./MiniImageNet",meta_train=True,download=True)
        # dataset_generator = Mini.MiniImageNetGenerator(args)
        # # dataset_generator.untar_mini()
        # # dataset_generator.process_original_files()
        # dataloader = MiniImageNetDataLoader(shot_num=5, way_num=5, episode_test_sample_num=15)
        #
        # # dataloader.generate_data_list(phase='train')
        # # dataloader.generate_data_list(phase='val')
        # # dataloader.generate_data_list(phase='test')
        #
        # dataloader.load_list(phase='all')
        # total_train_step=20000
        # total_test_img, total_test_label, total_train_img, total_train_label = \
        #     dataloader.get_batch(phase='train', idx=0)
        # total_train_img=np.mean(total_train_img, axis=-1)
        # total_test_img=np.mean(total_test_img, axis=-1)
        # total_train_label=np.array([np.where(item == 1)[0][0] for item in total_train_label])
        # total_test_label=np.array([np.where(item == 1)[0][0] for item in total_test_label])
        # idx=1
        # while idx<total_train_step:
        #     episode_test_img, episode_test_label, episode_train_img, episode_train_label = \
        #         dataloader.get_batch(phase='train', idx=idx)
        #     total_train_img=np.concatenate((total_train_img,np.mean(episode_train_img, axis=-1)),axis=0)
        #     total_test_img=np.concatenate((total_test_img,np.mean(episode_test_img, axis=-1)),axis=0)
        #     episode_train_label = np.array([np.where(item == 1)[0][0] for item in episode_train_label])
        #     episode_test_label = np.array([np.where(item == 1)[0][0] for item in episode_test_label])
        #     total_train_label=np.concatenate((total_train_label,episode_train_label),axis=0)
        #     total_test_label=np.concatenate((total_test_label,episode_test_label),axis=0)
        #     idx+=random.randint(1,1000)
        #     pass
        # np.save("MiniImageNet/total_train_img.npy",total_train_img)
        # np.save("MiniImageNet/total_test_img.npy",total_test_img)
        # np.save("MiniImageNet/total_train_label.npy",total_train_label)
        # np.save("MiniImageNet/total_test_label.npy",total_test_label)
        # total_train_img=np.load("MiniImageNet/total_train_img.npy")
        # total_test_img=np.load("MiniImageNet/total_test_img.npy")
        # total_train_label=np.load("MiniImageNet/total_train_label.npy")
        # total_test_label=np.load("MiniImageNet/total_test_label.npy")
        # pass
        # train_index=np.random.choice(len(total_train_img[(total_train_label == 0) | (total_train_label == 1)]), size=int((Sample_Size*5)/6),
        #                      replace=False)
        # X_train=total_train_img[(total_train_label == 0) | (total_train_label == 1)][train_index]
        # y_train=total_train_label[(total_train_label == 0) | (total_train_label == 1)][train_index]
        # test_index=np.random.choice(len(total_test_img[(total_test_label == 0) | (total_test_label == 1)]), size=int(Sample_Size/6),
        #                      replace=False)
        # X_test=total_test_img[(total_test_label == 0) | (total_test_label == 1)][test_index]
        # y_test=total_test_label[(total_test_label == 0) | (total_test_label == 1)][test_index]
        # X_train, X_test = X_train / 255.0, X_test / 255.0
        # X_train=np.array([item.flatten() for item in X_train])
        # X_test=np.array([item.flatten() for item in X_test])
        # np.save("MiniImageNet/X_train.npy",X_train)
        # np.save("MiniImageNet/X_test.npy",X_test)
        # np.save("MiniImageNet/y_train.npy",y_train)
        # np.save("MiniImageNet/y_test.npy",y_test)

        X_train=np.load("MiniImageNet/X_train.npy")
        X_test=np.load("MiniImageNet/X_test.npy")
        y_train=np.load("MiniImageNet/y_train.npy")
        y_test=np.load("MiniImageNet/y_test.npy")


        print("hi")
    #





    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(As)


def _grid_coordinates(side):
    """
    Returns 2D coordinates for a square grid of equally spaced nodes.
    :param side: int, the side of the grid (i.e., the grid has side * side nodes).
    :return: np.array of shape (side * side, 2).
    """
    M = side ** 2
    x = np.linspace(0, 1, side, dtype=np.float32)
    y = np.linspace(0, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def _get_adj_from_data(X, k, **kwargs):
    """
    Computes adjacency matrix of a K-NN graph from the given data.
    :param X: rank 1 np.array, the 2D coordinates of pixels on the grid.
    :param kwargs: kwargs for sklearn.neighbors.kneighbors_graph (see docs
    [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html)).
    :return: scipy sparse matrix.
    """
    A = kneighbors_graph(X, k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))

    return A


def _mnist_grid_graph(k):
    """
    Get the adjacency matrix for the KNN graph.
    :param k: int, number of neighbours for each node;
    :return:
    """
    X = _grid_coordinates(MNIST_SIZE)
    A = _get_adj_from_data(
        X, k, mode='connectivity', metric='euclidean', include_self=False
    )

    return A


def _flip_random_edges(A, percent):
    """
    Flips values of A randomly.
    :param A: binary scipy sparse matrix.
    :param percent: percent of the edges to flip.
    :return: binary scipy sparse matrix.
    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError('A must be a square matrix.')
    dtype = A.dtype
    A = sp.lil_matrix(A).astype(np.bool)
    n_elem = A.shape[0] ** 2
    n_elem_to_flip = round(percent * n_elem)
    unique_idx = np.random.choice(n_elem, replace=False, size=n_elem_to_flip)
    row_idx = unique_idx // A.shape[0]
    col_idx = unique_idx % A.shape[0]
    idxs = np.stack((row_idx, col_idx)).T
    for i in idxs:
        i = tuple(i)
        A[i] = np.logical_not(A[i])
    A = A.tocsr().astype(dtype)
    A.eliminate_zeros()
    return A

def draw_graph_mpl(g, pos=None, ax=None, layout_func=nx.drawing.layout.kamada_kawai_layout):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    else:
        fig = None
    if pos is None:
        pos = layout_func(g)
    node_color = []
    node_labels = {}
    shift_pos = {}
    for k in g:
        node_color.append(g.nodes[k].get('color', 'green'))
        node_labels[k] = g.nodes[k].get('label', k)
        shift_pos[k] = [pos[k][0], pos[k][1]]

    edge_color = []
    edge_width = []
    for e in g.edges():
        edge_color.append(g.edges[e].get('color', 'black'))
        edge_width.append(g.edges[e].get('width', 0.5))
    nx.draw_networkx_edges(g, pos, edge_color=edge_color, width=edge_width, ax=ax, alpha=0.5)#font_weight='bold',
    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_shape='p', node_size=1024, alpha=0.75)
    nx.draw_networkx_labels(g, shift_pos, labels=node_labels, ax=ax)#, arrows=True)
    ax.autoscale()
    return fig, ax, pos




