import numpy as np 
import matplotlib.pyplot as plt 

## PROBLEM 1: QUANTUM MECHANICS ON THE COMPUTER 
## Let φ(x)= dΨ/dx. Then, dφ(x)/dx = -2m * (EΨ(x)-V(x)Ψ(x))

def sch_rhs(t, x, m, E, V, *V_args, **V_kwargs):
    psix = x[0]
    
    dphi = np.zeros(2, dtype=float)
    dphi[0] = x[1]
    dphi[1] = -2*m*((E*psix) - (V(t, *V_args, **V_kwargs)*psix))

    return dphi


def runge_kutta(f, t, h, xn, *f_args, **f_kwargs):
    xn = np.array(xn, dtype=float)
    k1 = f(t, xn, *f_args, **f_kwargs)
    k2 = f(t + h/2, xn + h/2*k1, *f_args, **f_kwargs)
    k3 = f(t + h/2, xn + h/2*k2, *f_args, **f_kwargs)
    k4 = f(t + h, xn + h*k3, *f_args, **f_kwargs)

    xn += h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return xn 

def rk_schr(a, b, n, x, filename, corr, m, E, V, *V_args, **V_kwargs): 
    x = np.array(x)
    t = a 
    h = (b-a)/n
    print(h)
    psi_vals = []
    x_vals = []
    corr_vals = []
    for i in range(n): 
        t += h
        xt = runge_kutta(sch_rhs, t, h, x, m, E, V, *V_args, **V_kwargs)
        x = xt 

        x_vals.append(t)
        psi_vals.append(xt[0])
        corr_vals.append(corr(t, E))

    plt.figure(figsize=(8, 8))

    print(max(psi_vals))

    plt.plot(x_vals, psi_vals, label="numerical $\psi(x)$")
    # plt.plot(x_vals, corr_vals, label="analytic $\psi(x)$", linestyle="--")

    plt.xlabel('x', fontsize=12)
    plt.ylabel('$\psi(x)$', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./Homeworks/{filename}', dpi=300)

    return 


# Run RK4 for this problem 
n = 10000 
m = 1

def test_V1(x): return 0 
def corr1(x, E): return np.sin(np.sqrt(2*E)*x)
def test_V2(x, m, w): return m*(w**2)*(x**2)/2
def corr2_gs(x, E0): 
    w = 2*E0 
    return 27275.403313676434*np.exp(-(m*w*x**2)/2)

x1_init = [0, 1]

E = 10
# normalization constant 0.22360679699168837
# rk_schr(0, 10, n, x1_init, "free_system", corr1, m, E, test_V1)
E = (np.pi**2)/200
# normalization constant 3.183098861837903
# rk_schr(0, 10, n, x1_init, "free_system_E2", corr1, m, E, test_V1)

w = 1 
E0 = w/2

x2_init = [0, 1]
# rk_schr(-5, 5, n, x2_init, "sho", corr2_gs, m, E0, test_V2, m, w)
# Plotting multiples of E0 give excited states of the SHO. 

# rk_schr(-5, 5, n, x2_init, "sho_3E", corr2_gs, m, 3*E0, test_V2, m, w)
# rk_schr(-5, 5, n, x2_init, "sho_5E", corr2_gs, m, 5*E0, test_V2, m, w)


## PROBLEM 2: THE SHOOTING METHOD 


