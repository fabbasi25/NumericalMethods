import numpy as np 
import matplotlib.pyplot as plt 
import math 

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


    plt.plot(x_vals, psi_vals, label="numerical $\psi(x)$")
    # plt.plot(x_vals, corr_vals, label="analytic $\psi(x)$", linestyle="--")

    plt.xlabel('x', fontsize=12)
    plt.ylabel('$\psi(x)$', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./Homeworks/{filename}', dpi=300)

    return x


# Run RK4 for this problem 
n = 1000
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
rk_schr(0, 10, n, x1_init, "free_system", corr1, m, E, test_V1)
E = (np.pi**2)/200
# normalization constant 3.183098861837903
rk_schr(0, 10, n, x1_init, "free_system_E2", corr1, m, E, test_V1)

w = 1 
E0 = w/2

x2_init = [0, 1]
rk_schr(-5, 5, n, x2_init, "sho", corr2_gs, m, E0, test_V2, m, w)
# Plotting multiples of E0 give excited states of the SHO. 

rk_schr(-5, 5, n, x2_init, "sho_3E", corr2_gs, m, 3*E0, test_V2, m, w)
rk_schr(-5, 5, n, x2_init, "sho_32E", corr2_gs, m, 3.5*E0, test_V2, m, w)

rk_schr(-5, 5, n, x2_init, "sho_5E", corr2_gs, m, 5*E0, test_V2, m, w)


## PROBLEM 2: THE SHOOTING METHOD 
def rk_isw(E, a, b, n, x, m, V, *V_args, **V_kwargs): 
    x = np.array(x)
    t = a 
    h = (b-a)/n

    for i in range(n): 
        t += h
        xt = runge_kutta(sch_rhs, t, h, x, m, E, V, *V_args, **V_kwargs)
        x = xt 

    return x[0]

w = 1 
E_vals = np.linspace(0, 3*w, 1000)
psi_vals = []

for E in E_vals: 
    psi_vals.append(rk_isw(E, -5, 5, n, [0, 1], m, test_V2, m, w))
plt.figure(figsize=(8, 8))


plt.plot(E_vals, psi_vals, label="numerical $\psi(x)$")

plt.xlabel('E', fontsize=12)
plt.ylabel('$\psi(x)$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'./Homeworks/inf_sq', dpi=300)

# bisection 

def root_finder(Emin, Emax, f, *f_args, threshold=1e-5, **fkwargs): 
    x0 = Emin   
    roots = [] 
    while x0 <= Emax:
        a = x0 
        b = a + 0.01 

        if f(a, *f_args, **fkwargs)*f(b, *f_args, **fkwargs) < 0: 
            while abs(a-b) > threshold:
                mid = (a+b)/2
                fmid = f(mid, *f_args, **fkwargs)
                if fmid == 0: 
                    break
                if f(a, *f_args, **fkwargs)*fmid < 0: 
                    b = mid 
                elif f(b, *f_args, **fkwargs)*fmid < 0:
                    a = mid 

            roots.append(mid)
        
        x0 += 0.01 

    return roots 

# shooting method for free system 
E_min = 0
E_max = 1.5

i_values = [i for i in range(5)]
correct = [(np.pi**2)/(200)*(i+1)**2 for i in range(5)]
E_roots = root_finder(E_min, E_max, rk_isw, 0, 10, n, [0, 1], m, test_V1, threshold=1e-10)


plt.figure(figsize=(8, 8))

plt.plot(i_values, E_roots, label="numerical estimates", marker="o")
plt.plot(i_values, correct, label="predictions", marker="*")

plt.xlabel('n', fontsize=12)
plt.ylabel('$E_n$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'./Homeworks/shooting_zero_free', dpi=300)

    
# shooting method for harmonic oscillator 

E_min = 0
E_max = 5
i_values = [i for i in range(5)]
correct = [w/2*(2*i+1) for i in range(5)]
E_roots = root_finder(E_min, E_max, rk_isw, -5, 5, n, [0, 1], m, test_V2, m, w, threshold=1e-10)

plt.figure(figsize=(8, 8))

plt.plot(i_values, E_roots, label="numerical estimates", marker="o")
plt.plot(i_values, correct, label="predictions", marker="*", linestyle="--")

plt.xlabel('n', fontsize=12)
plt.ylabel('$E_n$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'./Homeworks/shooting_zero_sho', dpi=300)



