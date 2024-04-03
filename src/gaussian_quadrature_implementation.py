import numpy as np

def f(poly_coefs, x):
    alfa, beta, gamma = poly_coefs
    return alfa*x**2 + beta*x + gamma

def two_point_gaussian_quadrature(f, a, b):
    x = np.array([-np.sqrt(3)/3, np.sqrt(3)/3])
    c = np.array([1, 1])

    integral = 0
    for i in range(len(x)):
        integral += c[i] * f((b - a) / 2 * x[i] + (a + b) / 2)
    return integral * (b - a) / 2
