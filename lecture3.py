import numpy as np 

def riemann(a, b, n, f, *f_args, **f_kwargs):
    xk = np.linspace(a, b, n, endpoint=False)
    fk = f(xk, *f_args, **f_kwargs)
    h = (b-a)/n
    return np.sum(fk)*h

def trap(a, b, n, f, *f_args, **f_kwargs):
    xk = np.linspace(a, b, n)
    fk = f(xk, *f_args, **f_kwargs)
    h = xk[1] - xk[0]
    return 0.5 * h * (fk[0] + fk[n-1] + 2*fk[1:-1].sum())

def simpsons(a, b, n, f, *f_args, **f_kwargs):
    xk = np.linspace(a, b, n+1)
    fk = f(xk, *f_args, **f_kwargs)
    h = xk[1] - xk[0]
    return h * (fk[0] + fk[1] + 4*fk[1:-1:2].sum() + 2*fk[2:-2:2].sum())/3.

def f(x): 
    return np.sin(x)

print(riemann(0, np.pi, 10, f))
print(trap(0, np.pi, 10, f))
print(simpsons(0, np.pi, 10, f))