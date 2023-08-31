import numpy as np

''' Project epsilon on the space defined by intersection of box constraints and the hyperplane constraints. '''

def ProjectEps(eps_hat, K, tol = 1e-8, iter_max=20):
    #print(eps_hat)
    d = len(eps_hat)
    
    ##Clip each coordinate to the interval [0, 1] to satisfy box constraints and return the resulting vector if hyperplane constraints are satisfied 
    eps_star = np.clip(eps_hat, 0, 1)
    if sum(eps_star) <= K:
        return eps_star                                         #hyperplane constraints are satisfied, so we are done
    
    ##Do binary search for a maximum of iter_max steps
    iter = 0
    low = 1.0*(sum(eps_hat) - K)/d
    high = np.max(eps_hat) - 1.0*K/d
    while low <= high and iter < iter_max:
        Lambda = 0.5*(low + high)                               
        eps_star = eps_hat - Lambda
        eps_star = np.clip(eps_star, 0, 1)               #clip the values outside [0, 1] to have all values of eps_star between 0 and 1
        if abs(sum(eps_star) - K) <= tol:
            return eps_star
        elif sum(eps_star) > K:
            low = 0.5*(low + high)                              #look for a higher lambda
        else:
            high = 0.5*(low + high)                             #look for a lower lambda
        
        iter = iter + 1
            
    return eps_star

#### Test
#eps = np.array([0.3, 0.5, 0.2], dtype=float)
#K = 0.5
#x = ProjectEps(eps, K)
#print(x)