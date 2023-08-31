
import numpy as np
from numpy import linalg as LA
from ProjectEps import ProjectEps
from ProjectT import ProjectT


"""
Implements customized Mirror Prox Algorithms (Algorithm 2) 
"""


def SolveDual(rho_0, F, c, K, lambda_1, epsilon, t, zeta, alpha_eps=1, alpha_t=1, alpha_zeta=1, iter_max=100, tol=1e-4):

    #### Initialization step: 
    epsilon_prev = epsilon
    t_prev = t
    zeta_prev = zeta
    F_transpose = np.transpose(F)
    L = 2*np.power(LA.norm(F, ord=2), 2)  
    running_sum_epsilon = 0.0
    running_sum_alpha_eps = 0.0
    
    #### Define (sub-)gradients with respect to epsilon, t, and zeta
    eps_gradient = lambda r, r_plus:  -(0.5/lambda_1) * np.multiply(2*r-r_plus, r_plus)
    t_gradient = lambda f, rho_0:  f - rho_0         #f = (1.0/lambda_1) * np.multiply(np.multiply(epsilon, r), r_plus) 
    zeta_gradient = lambda f: np.sum(f) - 1
    
    #t_gradient = lambda epsilon, r, r_plus, rho_0:  (1.0/lambda_!) * np.multiply(np.multiply(epsilon, r), r_plus)  - rho_0
    
    #### Main Loop
    for iter in range(iter_max):   
        
        r_prev = -(t_prev + zeta_prev)
        r_plus_prev = np.clip(r_prev, a_min=0, a_max=None)
        f_prev = (1.0/lambda_1) * np.multiply(np.multiply(epsilon_prev, r_prev), r_plus_prev) 
        
        # Gradient step for epsilon
        epsilon_hat = epsilon_prev - alpha_eps * eps_gradient(r_prev, r_plus_prev) 
        epsilon_hat = ProjectEps(epsilon_hat, K, tol)
        
        # Gradient step for t
        t_hat = t_prev + alpha_t * t_gradient(f_prev, rho_0)    
        t_hat = ProjectT(t_hat, F, F_transpose, L, c, tol)
        
        # Gradient step for zeta
        zeta_hat = zeta_prev + alpha_zeta * zeta_gradient(f_prev)   #No projection needs to be done
        
        
        r_hat = -(t_hat + zeta_hat)
        r_plus_hat = np.clip(r_hat, a_min=0, a_max=None)
        f_hat = (1.0/lambda_1) * np.multiply(np.multiply(epsilon_hat, r_hat), r_plus_hat)
        
        
        # Extra-gradient step for epsilon
        epsilon_tilde = epsilon_prev - alpha_eps * eps_gradient(r_hat, r_plus_hat) 
        epsilon_tilde = ProjectEps(epsilon_tilde, K, tol)
        
        # Extra-gradient step for t
        t_tilde = t_prev + alpha_t * t_gradient(f_hat, rho_0)    
        t_tilde = ProjectT(t_tilde, F, F_transpose, L, c, tol)
        
        # Extra-gradient step for zeta
        zeta_tilde = zeta_prev + alpha_zeta * zeta_gradient(f_hat)   #No projection needs to be done
        
        
        # Update the previous epsilon, t, and zeta
        epsilon_prev = epsilon_tilde
        t_prev = t_tilde
        zeta_prev = zeta_tilde
        
        # Update the running sums for epsilon
        running_sum_epsilon += alpha_eps * epsilon_hat
        running_sum_alpha_eps += alpha_eps
        
    #### Compute the approximate epsilon for the relaxed problem    
    epsilon_hat = running_sum_epsilon/running_sum_alpha_eps
    
    return epsilon_hat   
        


