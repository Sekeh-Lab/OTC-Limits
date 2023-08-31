import numpy as np
from numpy import linalg as LA


def ProjectT(t_hat, A, A_transpose, L, c, tol = 1e-8, iter_max = 100):
    t_j = 1
    w = np.zeros(A_transpose.shape[1])
    y_j_minus_1 = w
    for j in range(iter_max):
        u = t_hat + np.dot(A_transpose, w)
        Au = np.dot(A, u)
        v =  np.clip(Au - L*w, -c, c)
        y_j = w - 1.0*(Au - v)/L 
        t_j_plus_1 = 0.5 * (1 + np.sqrt(1 + 4*t_j*t_j))
        w = y_j + 1.0*(t_j - 1)*(y_j - y_j_minus_1)/t_j_plus_1
        t_j = t_j_plus_1
        if np.max(np.abs(y_j - y_j_minus_1)) <= tol:
            break
        y_j_minus_1 = y_j
    
    x = t_hat + np.dot(A_transpose, y_j)
    return x  



#### Test
#F = np.array([[1, 1, 0], [1, 0, 1]], dtype=int)  #(1, 2), (1,3) in edges
#F_transpose = np.transpose(F)
#t = np.array([1, -2, 4], dtype=float)
#c = np.array([4, 5], dtype=float)
#c1 = np.array([4, 3], dtype=float)
#L = 2*np.power(LA.norm(F, ord=2), 2)
#
#
#projected_t_c = ProjectT(t, F, F_transpose, L, c)
#print('c = ', c)
#print('Projected t with c: ', projected_t_c)
#print("Gap ", np.dot(t-projected_t_c, t-projected_t_c))
#
#projected_t_c1 = ProjectT(t, F, F_transpose, L, c1)
#print('\n c1 = ', c1)
#print('Projected t with c1: ', projected_t_c1)
#print("Gap ", np.dot(t-projected_t_c1, t-projected_t_c1))