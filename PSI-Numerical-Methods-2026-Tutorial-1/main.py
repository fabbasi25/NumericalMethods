# HELLO WORLD for PSI Numerical Methods 2026

def myexp(x, N=10):
    """
    This function computes exp(x) via the Taylor Series using terms up to
    order N.
    """

    t = 1.0  # The value of the current term
    y = 1.0  # The value of exp(x) up to this point

    # We initialize with the 0th order term, so run the loop
    # for the remaining N terms.
    for i in range(1, N+1):
        t *= x/i  # Update the term: tn = x^n / n!
        y += t    # add tn to y

    # We're done!
    return y

def kinetic_energy(m, v): 
    c = 299792458
    gamma = 1/(1-(v**2))**0.5
    if (abs(v) != 0) & (abs(gamma-1) < 1e-9): 
        return m*v**2/2
        
    energy = (gamma-1)*m*c**2
    return energy

def linear_bisector(a, b, f, n=0, threshold=1e-5):
    if abs(f(a)) < threshold:
        return a, n+1  
    elif abs(f(b)) < threshold: 
        return b, n+1
    elif abs(a-b) < threshold:
        mid = (a+b)/2
        return mid, n+1 
    elif f(a)*f(b) < 0: 
        mid = (a+b)/2
        if abs(f(mid)) < threshold: 
            return mid, n+1 
        elif f(mid) > 0: 
            return linear_bisector(a, mid, f, n+1)
        else: 
            return linear_bisector(mid, b, f, n+1)

def newton_raphson(f, fd, x, n=0, threshold=1e-5):
    if abs(f(x) - 0) < threshold: 
        n +=1
        return x, n  
    else: 
        x -= f(x)/fd(x)
        n += 1 
        return newton_raphson(f, fd, x, n)

def lin_fun(x, m=0.3, c=-1): 
    return m*x + c 

def lin_fun_der(x, m=0.3, c=-1):
    return m 

def long_cubic(x): 
    return 49*x**3 - 161*x**2 - 440*x + 1452

def long_cubic_der(x): 
    return 49*3*x**2 - 161*2*x - 440 

def cubic(x):
    return x**3 - 7*x + 2 

def cubic_der(x): 
    return 3*x**2 - 7 
 
if __name__ == "__main__":

    ## PROBLEM 3 

    print("Hello PSI 2026!")

    import numpy 
    print(numpy.e**1.2)

    # import matplotlib.pyplot as plt 
    # N = 1000 
    # x_data_list = [i*2*numpy.pi/N for i in range(N+1)]
    # x_data = numpy.array(x_data_list)
    # y_data_cos = numpy.cos(x_data)
    # y_data_sin = numpy.sin(x_data)
    # graph = plt.figure(figsize=(9,9), layout='tight')
    # plt.plot(x_data, y_data_cos, color="red", label="cos(x)")
    # plt.plot(x_data, y_data_sin, color="blue", label="sin(x)")
    # plt.legend()

    # plt.savefig("./PSI-Numerical-Methods-2026-Tutorial-1/trig.png", transparent=False)

    ## PROBLEM 4 
    start = 1.0 
    current = 1.0 
    additional_number = 1e-20
    found_diff = False
    while not found_diff: 
        current += additional_number
        if abs(current-start) != 0:
            print("needed to add ", additional_number)
            found_diff = True
        else: 
            additional_number*=10

    start = numpy.float64(1e10)
    found_large = False
    while not found_large: 
        if numpy.isinf(start*10): 
            found_large = True 
            print(start + 7.9e307)
        else:
            start *= 10

    #   baseball (m = 150 g, v = 40 m/s), a cosmic ray (m = 1.67× 10−27kg, v = (1− 10−15)c),
    #   a power walker (m = 70 kg, v = 3 m/s), and 
    #   a red blood cell in a capillary (m = 0.1 ng, v = 1.02mm/s).
    c = 299792458
    print("energy of baseball is ", kinetic_energy(0.15, 40/c))
    print("energy of cosmic ray is ", kinetic_energy(1.67e-27, v = (1- 1e-15)))
    print("energy of power walker is ", kinetic_energy(70, 3/c))
    print("energy of red blood cell is ", kinetic_energy(0.1e-9, 1.02e-3/c))

    ## PROBLEM 5 
    print("The correct answer to solve 0.3x-1=y is 3.333.")
    lb_test_func_linear = linear_bisector(-10, 10, lin_fun)
    nr_test_func_linear = newton_raphson(lin_fun, lin_fun_der, 10)
    print("With linear bisector, we find ", lb_test_func_linear[0], " computed in ", lb_test_func_linear[1], " steps.")
    print("With Newton-Raphson method, we find ", nr_test_func_linear[0], " computed in ", nr_test_func_linear[1], " steps.")

    for t in [1e-2, 1e-4, 1e-8]: 
        lb_test_func_cubic = linear_bisector(1, 20, cubic, n=0, threshold=t)
        nr_test_func_cubic = newton_raphson(cubic, cubic_der, 10, n=0, threshold=t)

        print("With linear bisector, we find ", lb_test_func_cubic[0], " computed in ", lb_test_func_cubic[1], f" steps when the threshold is {t}.")
        print("With Newton-Raphson method, we find ", nr_test_func_cubic[0], " computed in ", nr_test_func_cubic[1], f" steps when the threshold is {t}.")

    long_cubic_nr = newton_raphson(long_cubic, long_cubic_der, 10)
    print("With Newton-Raphson method, we find ", long_cubic_nr[0], " computed in ", long_cubic_nr[1], f" steps.")
    try: 
        long_cubic_lb = linear_bisector(0, 30, long_cubic)
        print("With linear bisector, we find ", long_cubic_lb[0], " computed in ", long_cubic_lb[1], " steps.")
    except: 
        print("Could not find a root with linear bisection method")
    print("e(1) with  5 terms is", myexp(1.0, 5))
    print("e(1) with 10 terms is", myexp(1.0, 10))
    print("e(1) with 20 terms is", myexp(1.0, 20))
    print("e(1) with 40 terms is", myexp(1.0, 40))
