import numpy as np
import matplotlib.pyplot as plt 

### PROBLEM 1: FIND THE SAMPLE POINTS xk

# Legendre polynomials & its derivatives 
def L(x, n):
    if n == 0:  
        return 1 
    elif n == 1: 
        return x 
    else: 
        return 1/n*((2*n-1)*x*L(x, n-1) - (n-1)*L(x, n-2))
    

def L_der(x, d):
    if d % 2 == 0: 
        j = int((d-2)/2)
        sum_elem = np.array([(4*i+3)*L(x, 2*i+1) for i in range(j+1)])
    else: 
        j = int((d-1)/2)
        sum_elem = np.array([(4*i+1)*L(x, 2*i) for i in range(j+1)])
    
    if len(sum_elem) == 0: 
        return 0
    else: 
        return np.sum(sum_elem)


def newton_raphson(f, fd, x, *f_args, iter=0, threshold=1e-5):
    if abs(f(x, *f_args) - 0) < threshold: 
        iter +=1
        return x, iter 
    else: 
        x -= f(x, *f_args)/fd(x, *f_args)
        iter += 1 
        return newton_raphson(f, fd, x, *f_args, iter=iter, threshold=threshold)


# use Newton-Raphson to find roots 
roots_of_L = []
for i in range(1, 8): 
    num_roots = 0 
    for x_guess in [1, 0.5, -0.25, 0.25, -0.5, -1]:
        if (num_roots < i): 
            roots_of_L.append(newton_raphson(L, L_der, x_guess, i, iter=0, threshold=1e-14)[0])
            num_roots += 1 

# only keep distinct roots 
roots_of_L = set(roots_of_L)
print(roots_of_L)

## PROBLEM 2: FIND THE WEIGHTS wk

def phi(x, k, x_points):
    phi_val = 1 
    x_k = x_points[k-1]
    for m, point in enumerate(x_points): 
        if m != (k-1): 
            phi_val *= (x-point)/(x_k - point)
    return phi_val

# make a plot of PN(x) and all φk(x) for N = 7






