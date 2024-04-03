import numpy as np

def f(poly_coefs, x):
    alfa, beta, gamma = poly_coefs
    return alfa*x**2 + beta*x + gamma

def trapezoid_rule(f, a, b, N):
    x = np.linspace(a, b, N)
    x_0 = x[0]
    x_N = x[-1]

    boundary_terms = 0.5*(f(x_N) + f(x_0))
    inner_terms = 0
    for i in range(N):
        inner_terms += f(x[i])

    integral = ((b - a)/N) * (inner_terms + boundary_terms)
    return integral