import numpy as np
import matplotlib.pyplot as plt 

### PROBLEM 1: FIND THE SAMPLE POINTS xk

# Legendre polynomials & its derivatives 
def L(x, d):
    if d == 0:  
        return 1 
    elif d == 1: 
        return x 
    else: 
        return 1/d*((2*d-1)*x*L(x, d-1) - (d-1)*L(x, d-2))
    

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
def roots_of_L(i): 
    roots_ = []
    num_roots = 0 
    if L(0, i) == 0: 
        roots_.append(0)
        num_roots += 1

    if i > 1:  
        for x_guess in [1, 0.75, 0.5, -0.25, 0.25, -0.5, -0.75, -1]:
            new_root = newton_raphson(L, L_der, x_guess, i, iter=0, threshold=1e-14)[0]
            new = True
            for root in roots_: 
                if abs(root - new_root) < 1e-6: 
                    new = False 
            if new: 
                roots_.append(new_root)
            num_roots += 1
    
    # only keep distinct roots 
    return np.array(roots_)


### PROBLEM 2: FIND THE WEIGHTS wk

def phi(x, k, x_points):
    # x_points is the array of sample points 
    phi_val = np.ones_like(x) 
    x_k = x_points[k-1]
    for m, point in enumerate(x_points): 
        if m != (k-1): 
            phi_val *= (x-point)/(x_k - point)
    return phi_val

# make a plot of PN(x) and all φk(x) for N = 7

plt.figure(figsize=(10, 6))
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]

N = 7
x_values = np.linspace(-1.1, 1.1, 100)
P7 = L(x_values, 7)
plt.plot(x_values, P7, color='red', label="$P_7(x)$", lw=3, zorder=0)
x_points_for_P7 = roots_of_L(7)


for k_ in range(1, 8):
    phi_k = phi(x_values, k_, x_points_for_P7)
    plt.plot(x_values, phi_k, color=colors[k_-1], label=f"$\phi_{{{k_}}}$") 


plt.scatter(
    x_points_for_P7,
    np.zeros_like(x_points_for_P7),
    marker="x",
    s=75,
    color="black",
    zorder=7,
    label="Roots of $P_7$"
)

plt.axhline(
    1.0,
    linestyle=":",
    linewidth=1.4,
    color="black",
    alpha=0.6,
    zorder=0
)

plt.xlabel('$x$', fontsize=12)
plt.ylabel('Polynomial functions', fontsize=12)
plt.title(f'$P_7$ Legendre polynomial and its indicator polynomials',
            fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Homeworks/P7.png', dpi=300)
# plt.show()

# compute weights 

def simpsons(a, b, n, f, *f_args, **f_kwargs):
    xk = np.linspace(a, b, n)
    fk = f(xk, *f_args, **f_kwargs)
    h = xk[1] - xk[0]
    return h * (fk[0] + fk[1] + 4*fk[1:-1:2].sum() + 2*fk[2:-2:2].sum())/3.

def wk(k, x_points):
    return simpsons(-1, 1, 100_000, phi, k, x_points)

print("Weight 1 for P7 are ", wk(1, x_points_for_P7))

### PROBLEM 3: INTEGRATE!

def test_e(x): 
    return np.exp(x)

def test_rational(x):
    return np.cbrt(x) + 1/(1+100*(x-5)**2)

def test_abs(x): 
    return x**5*np.abs(x)

def gaussian_int(a, b, func, Nsub, N, *fargs, **fkwargs):
    aj = []
    for j in range(Nsub):
        aj.append(a + (b-a)*j/Nsub)
    
    int_result = 0 
    sample_points = roots_of_L(N)
    k_values = np.arange(1, N+1)
    weights_k = []

    for k in k_values:
        weights_k.append(wk(k, sample_points))
    
    weights_k = np.array(weights_k)

        
    for j in range(Nsub-1): 
        # integrate from aj to aj+1
        this_term = (aj[j+1]-aj[j]) / 2 * func(sample_points, *fargs, **fkwargs) * weights_k
        int_result += np.sum(this_term)
    
    return int_result

# plot 

def plot_int_results(a, b, test_func, correct_eval, function_str, filename):
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    
    plt.figure(figsize=(10, 6))

    for N in range(1, 8):
        int_errors_N = []
        func_evals = []
        for Nsub_ in range(1, 2000, 100):
            int_result = gaussian_int(a, b, test_func, Nsub_, N)
            int_error = abs((correct_eval - int_result)/correct_eval)
            if int_error < 0: 
                print(int_error)
            int_errors_N.append(int_error)
            func_evals.append(N*Nsub_)

        plt.plot(func_evals, int_errors_N, color=colors[N-1], label=f"$N = {{{N}}}$") 


    plt.xlabel('Function evaluations (log)', fontsize=12)
    plt.ylabel('Error in integration', fontsize=12)
    plt.title(f'Error in Integration for the Function {function_str}',
                fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f'./Homeworks/{filename}.png', dpi=300)


# plotting test functions 
correct_exp = np.e -1
correct_rat = 1/20 * (-15 + 150 * 10**(1/3) + 2 * np.arctan(np.deg2rad(40)) + 2 * np.arctan(np.deg2rad(50)))
correct_abs = 16384/7 - 4096*np.pi + 3072*np.pi**2 - 1280*np.pi**3 + 320*np.pi**4 - 48*np.pi**5 + 4*np.pi**6 - (2*np.pi**7)/7

plot_int_results(0, 1, test_e, correct_exp, "$e^x$", "exp")
plot_int_results(1, 10, test_rational, correct_rat, "$x^{1/3} + 1/(1 + 100 (x - 5)^2)$", "rational")
plot_int_results(-np.pi, 4-np.pi, test_abs, correct_abs, "$x^5|x|$", "abs")