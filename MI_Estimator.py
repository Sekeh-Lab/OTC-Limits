# GCN MI Estimator

import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import itertools

# Computing the mutual information I(G_v,X_v;C)
def MI(X, E, L, N, V, D, C, F):
    # Find KNN distances for a number of samples for normalizing bandwidth
    def find_knn(A, d):
        np.random.seed(3334)

        r = 500
        # random samples from A
        A = A.reshape((-1, 1))
        N = A.shape[0]

        k = max(math.floor(0.43 * N ** (2 / 3 + 0.17 * (d / (d + 1))) * math.exp(-1.0 / np.max([10000, d ** 4]))), 1)
        T = np.random.choice(A.reshape(-1, ), size=r).reshape(-1, 1);
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
        distances, indices = nbrs.kneighbors(T)
        d = np.mean(distances[:, -1])
        return d

    # Returns epsilon and random shifts b
    def gen_eps(XW):
        #Added to the original one
        if len(np.array([XW]).flatten())==1:
            XW = np.array([XW])
        d_X = XW.shape[0]
        #
        # Find KNN distances for a number of samples for normalizing bandwidth
        eps = find_knn(XW, d_X) + 0.0001

        return eps

    # Define H1 (LSH) for a vector X (X is just one sample)
    def H1(XW, b, eps):

        # dimension of X
        #Added to the original one
        if len(np.array([XW]).flatten()) == 1:
            XW = np.array([XW])
        d_X = XW.shape[0]

        XW = XW.reshape(1, d_X)

        # If not scalar
        if d_X > 1:
            X_te = 1.0 * (np.squeeze(XW) + b) / eps
        elif eps > 0:
            X_te = 1.0 * (XW + b) / eps
        else:
            X_te = XW

        # Discretize X
        X_t = np.floor(X_te)
        if d_X > 1:
            R = tuple(X_t.tolist())
        else:
            R = np.squeeze(X_t).item()
        return R

    def H_1(X):
        eps = gen_eps(X)
        b = 0.1 * np.random.rand(D) * eps
        return H1(X, b, eps)

    def H_2(X):
        return int(np.sum(X) % F)

    N_i_c = np.zeros((V, F, C))
    N_ii_c = np.zeros((V, V, F, C))
    N_ii_ec = np.zeros((V, V, F, C))

    for x in range(N):
        if x % 300 == 0:
            print('Sample {}...'.format(x))

        for y in range(V):
            H_X1 = H_2(H_1(X[x, y]))
            N_i_c[y, H_X1, L[x]] += 1

            for z in range(V):
                try:
                    H_X2 = H_2(H_1(X[x, z]))
                except Exception as e:
                    print(e)
                if H_X1 == H_X2:
                    N_ii_c[y, z, H_X2, L[x]] += 1
                    if E[x, y, z]:
                        N_ii_ec[y, z, H_X2, L[x]] += 1

    N_i = np.sum(N_i_c, -1)  # Number of collisions per vertex
    N_ii = np.sum(N_ii_c, -1)  # Number of collisions per vertex pair
    N_ii_e = np.sum(N_ii_ec, -1)  # Number of collisions per edge

    r_i_c = N_i_c / N
    r_ii_c = N_ii_c / N
    r_ii_ec = N_ii_ec / N

    r_i = N_i / N
    r_ii = N_ii / N
    r_ii_e = N_ii_e / N

    def r_frac(x, y, c=False):
        with np.errstate(invalid='ignore'):
            return np.nan_to_num(r_ii_ec[x, y] / r_ii_c[x, y] if c else r_ii_e[x, y] / r_ii[x, y])

    swap1=np.stack([r_frac(i, j) * (1 - r_frac(i, j)) for i, j in list(itertools.combinations(range(V), 2))])
    swap2=np.stack([r_frac(i, j, True) * (1 - r_frac(i, j, True)) for i, j in list(itertools.combinations(range(V), 2))])

    flag=False  # to inform that the MI is not the actual value, we are largening it to avoid zeros and be able to compare MIs
    pc_i=np.prod(r_i_c, 0) * np.prod(swap2[(swap2[:,0,0]!=0)&(swap2[:,0,1]!=0)&(swap1[:,0]!=0)],0)

    p_i=np.prod(r_i, 0) * np.prod(swap1[(swap1[:,0]!=0)&(swap2[:,0,0]!=0)&(swap2[:,0,1]!=0)],0)

    p_c = np.bincount(L, minlength=C) / N
    g = lambda x: np.log(x)
    U_1 = 100 * np.ones((F, C))
    U_2 = 0.001 * np.ones((F, C))

    with np.errstate(invalid='ignore', divide='ignore'):
        p_frac = np.nan_to_num( np.maximum(np.minimum(g(pc_i / p_i[:, np.newaxis]), U_1), U_2))
    I = np.sum(p_c * np.sum(p_frac, 0),dtype='f')
    return I#,flag


# %% [code]
# A simulation to validate the code
# V = 25  # Number of vertices # works for only <28 vertices, others get zero([5,6-> <7],[7-13 -> <5],[14-27 -> <4]
# N = 5000  # Number of samples
# D = 1  # Size of a feature
# C = 2  # Number of classes
# F = 5  # Number of collision bins
# for N in [1000,800,500,300]:#gets 0 for >=30
#
# A vertex's samples has a mean between -10 and 10, and a variance between .1 and 10
# X = np.stack([np.random.normal(np.random.uniform(-10, 10), np.random.uniform(.1, 10), (N, D))
#               for _ in range(V)], 1)
# E_v = np.random.choice(2, (N, V, V), p=(.1, .9))  # Edges (0 = no edge, 1 = edge)
# E = (E_v + np.swapaxes(E_v, 1, 2)) // 2  # Adjacency matrices (symmetric)
# L = np.random.choice(C, N)  # Class labels
# print(MI(X,E, L, N, V, D, C, F))

# for i in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
#     with open('results/result-NYC.pkl'+str(i)+'CompressionRatio-wRho0%', 'rb') as f:
#         data = pickle.load(f)
#
#     [MI(X,E, L, N, V, D, C, F) for nodes in data.outputVertices]
# for F in [2]:
#     mi=MI(X,E, L, N, V, D, C, F)
#     print(mi)# E_v, E, L, N, V, D, C, F)

# pass
# # %% [code]
# # Computing the mutual information I(X_u;C|X_v)
# # we assume that X_u and X_v have a same dimension
# # V = 2
# # N = Number of samples
# # D= Size of a feature
# # C= Number of classes
# # F = Number of collision bins
#
# # X = A vertex's samples
# # E_v= Edges (0 = no edge, 1 = edge)
# # E= Adjacency matrices (symmetric)
# # L = Class labels
# # X=(X_u,X_v) sample
# def MI_X(X, L, N, V, D, C, F):
#     # Find KNN distances for a number of samples for normalizing bandwidth
#     def find_knn(A, d):
#         np.random.seed(3334)
#         # np.random.seed()
#         # np.random.seed(seed=int(time.time()))
#         r = 500
#         # random samples from A
#         A = A.reshape((-1, 1))
#         N = A.shape[0]
#
#         k = max(math.floor(0.43 * N ** (2 / 3 + 0.17 * (d / (d + 1))) * math.exp(-1.0 / np.max([10000, d ** 4]))), 1)
#         # print('k,d', k, d)
#         T = np.random.choice(A.reshape(-1, ), size=r).reshape(-1, 1);
#         print
#         nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
#         distances, indices = nbrs.kneighbors(T)
#         d = np.mean(distances[:, -1])
#         return d
#
#     # Returns epsilon and random shifts b
#     def gen_eps(XW):
#         d_X = XW.shape[0]
#         # Find KNN distances for a number of samples for normalizing bandwidth
#         eps = find_knn(XW, d_X) + 0.0001
#         return eps
#
#     # Define H1 (LSH) for a vector X (X is just one sample)
#     def H1(XW, b, eps):
#
#         # dimension of X
#         d_X = XW.shape[0]
#         # d_W = W.shape[1]
#         XW = XW.reshape(1, d_X)
#
#         # If not scalar
#         if d_X > 1:
#             X_te = 1.0 * (np.squeeze(XW) + b) / eps
#         elif eps > 0:
#             X_te = 1.0 * (XW + b) / eps
#         else:
#             X_te = XW
#
#         # Discretize X
#         X_t = np.floor(X_te)
#         if d_X > 1:
#             R = tuple(X_t.tolist())
#         else:
#             R = np.squeeze(X_t).item()
#         return R
#
#     def H_1(X):
#         eps = gen_eps(X)
#         b = 0.1 * np.random.rand(D) * eps
#         return H1(X, b, eps)
#
#     def H_2(X):
#         return int(np.sum(X) % F)
#
#     N_i_c = np.zeros((V, F, C))
#     N_ii_c = np.zeros((V, V, F, C))
#
#     for x in range(N):
#         if x % 100 == 0:
#             print('Sample {}...'.format(x))
#
#         for y in range(V):
#             H_X1 = H_2(H_1(X[x, y]))
#             N_i_c[y, H_X1, L[x]] += 1
#
#             for z in range(V):
#                 H_X2 = H_2(H_1(X[x, z]))
#                 if H_X1 == H_X2:
#                     N_ii_c[y, z, H_X2, L[x]] += 1
#
#     N_i = np.sum(N_i_c, -1)  # Number of collisions per vertex
#     N_ii = np.sum(N_ii_c, -1)  # Number of collisions per vertex pair
#
#     r_i_c = N_i_c / N
#     r_ii_c = N_ii_c / N
#
#     r_i = N_i / N  # Rate of collisions per vertex
#     r_ii = N_ii / N  # Rate of collisions per vertex pair
#
#     def r_frac(x, y, c=False):
#         with np.errstate(invalid='ignore'):
#             return np.nan_to_num(r_ii_c[x, y] / r_i_c[x] if c else r_ii[x, y] / r_i[x])
#
#     pc_i = np.prod(np.stack([r_frac(i, j, True)
#                              for i, j in list(itertools.combinations(range(V), 2))]), 0)
#
#     p_i = np.prod(np.stack([r_frac(i, j)
#                             for i, j in list(itertools.combinations(range(V), 2))]), 0)
#
#     q_i = np.prod(np.stack([r_i_c[i]
#                             for i, j in list(itertools.combinations(range(V), 2))]), 0)
#
#     p_c = np.bincount(L, minlength=C) / N
#     g = lambda x: np.log(x)
#     U_1 = 10 * np.ones((F, C))
#     U_2 = 0.001 * np.ones((F, C))
#
#     with np.errstate(invalid='ignore', divide='ignore'):
#         p_frac = np.nan_to_num(pc_i * q_i * np.maximum(np.minimum(g(pc_i / p_i[:, np.newaxis]), U_1), U_2))
#     I = np.sum(p_c * np.sum(p_frac, 0))
#     return I
#
# # %% [code]
# # A simulation to validate the code for I(X_u;C|X_v)
# V = 2  # Number of vertices
# N = 1000  # Number of samples
# D = 5  # Size of a feature
# C = 2  # Number of classes
# F = 5  # Number of collision bins
#
# # A vertex's samples has a mean between -10 and 10, and a variance between .1 and 10
# X = np.stack([np.random.normal(np.random.uniform(-10, 10), np.random.uniform(.1, 10), (N, D))
#               for _ in range(V)], 1)
# E_v = np.random.choice(2, (N, V, V), p=(.1, .9))  # Edges (0 = no edge, 1 = edge)
# E = (E_v + np.swapaxes(E_v, 1, 2)) // 2  # Adjacency matrices (symmetric)
# L = np.random.choice(C, N)  # Class labels
#
# MI_X(X, L, N, V, D, C, F)
#
# # %% [code]
# # Computing the mutual information I(X_v;C)
# # we assume that X_u and X_v have a same dimension
# # V = 1
# # N = Number of samples
# # D= Size of a feature
# # C= Number of classes
# # F = Number of collision bins
#
# # X = A vertex's samples
# # E_v= Edges (0 = no edge, 1 = edge)
# # E= Adjacency matrices (symmetric)
# # L = Class labels
# # X=(X_u,X_v) sample
# def MI_Xv(X, L, N, V, D, C, F):
#     # Find KNN distances for a number of samples for normalizing bandwidth
#     def find_knn(A, d):
#         np.random.seed(3334)
#         # np.random.seed()
#         # np.random.seed(seed=int(time.time()))
#         r = 500
#         # random samples from A
#         A = A.reshape((-1, 1))
#         N = A.shape[0]
#
#         k = max(math.floor(0.43 * N ** (2 / 3 + 0.17 * (d / (d + 1))) * math.exp(-1.0 / np.max([10000, d ** 4]))), 1)
#         # print('k,d', k, d)
#         T = np.random.choice(A.reshape(-1, ), size=r).reshape(-1, 1);
#         print
#         nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
#         distances, indices = nbrs.kneighbors(T)
#         d = np.mean(distances[:, -1])
#         return d
#
#     # Returns epsilon and random shifts b
#     def gen_eps(XW):
#         d_X = XW.shape[0]
#         # Find KNN distances for a number of samples for normalizing bandwidth
#         eps = find_knn(XW, d_X) + 0.0001
#         return eps
#
#     # Define H1 (LSH) for a vector X (X is just one sample)
#     def H1(XW, b, eps):
#
#         # dimension of X
#         d_X = XW.shape[0]
#         # d_W = W.shape[1]
#         XW = XW.reshape(1, d_X)
#
#         # If not scalar
#         if d_X > 1:
#             X_te = 1.0 * (np.squeeze(XW) + b) / eps
#         elif eps > 0:
#             X_te = 1.0 * (XW + b) / eps
#         else:
#             X_te = XW
#
#         # Discretize X
#         X_t = np.floor(X_te)
#         if d_X > 1:
#             R = tuple(X_t.tolist())
#         else:
#             R = np.squeeze(X_t).item()
#         return R
#
#     def H_1(X):
#         eps = gen_eps(X)
#         b = 0.1 * np.random.rand(D) * eps
#         return H1(X, b, eps)
#
#     def H_2(X):
#         return int(np.sum(X) % F)
#
#     N_i_c = np.zeros((V, F, C))
#
#     for x in range(N):
#         if x % 100 == 0:
#             print('Sample {}...'.format(x))
#
#         for y in range(V):
#             H_X1 = H_2(H_1(X[x, y]))
#             N_i_c[y, H_X1, L[x]] += 1
#
#     N_i = np.sum(N_i_c, -1)  # Number of collisions per vertex
#
#     r_i_c = N_i_c / N
#
#     r_i = N_i / N  # Rate of collisions per vertex
#
#     pc_i = np.reshape(r_i_c, (D, C))
#
#     p_i = np.reshape(r_i, D)
#
#     p_c = np.bincount(L, minlength=C) / N
#     g = lambda x: np.log(x)
#     U_1 = 10 * np.ones((F, C))
#     U_2 = 0.001 * np.ones((F, C))
#
#     with np.errstate(invalid='ignore', divide='ignore'):
#         p_frac = np.nan_to_num(pc_i * np.maximum(np.minimum(g(pc_i / p_i[:, np.newaxis]), U_1), U_2))
#     I = np.sum(p_c * np.sum(p_frac, 0))
#     return I
#
# # %% [code]
# # A simulation to validate the code for I(X_u;C|X_v)
# V = 1  # Number of vertices
# N = 1000  # Number of samples
# D = 5  # Size of a feature
# C = 2  # Number of classes
# F = 5  # Number of collision bins
#
# # A vertex's samples has a mean between -10 and 10, and a variance between .1 and 10
# X = np.stack([np.random.normal(np.random.uniform(-10, 10), np.random.uniform(.1, 10), (N, D))
#               for _ in range(V)], 1)
# E_v = np.random.choice(2, (N, V, V), p=(.1, .9))  # Edges (0 = no edge, 1 = edge)
# E = (E_v + np.swapaxes(E_v, 1, 2)) // 2  # Adjacency matrices (symmetric)
# L = np.random.choice(C, N)  # Class labels
#
# MI_Xv(X, L, N, V, D, C, F)
#



