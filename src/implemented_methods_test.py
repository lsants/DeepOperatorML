import numpy as np

def gauss_quadrature_two_points(f, a, b):
    # Nodes and weights for 2-point Gauss-Legendre quadrature
    x = np.array([-0.5773502691896257, 0.5773502691896257])  # Nodes
    w = np.array([1.0, 1.0])  # Weights
    integral = 0.0
    for i in range(len(x)):
        integral += w[i] * f((b - a) / 2 * x[i] + (a + b) / 2)
    return integral * (b - a) / 2

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

# Define the function to integrate

def f(x):
    return x**2 + x + 1

# Define integration limits
a = 0.1
b = 1
N = 10000

# Perform Gaussian quadrature with two points
result = gauss_quadrature_two_points(f, a, b)
print("Approximate integral using Gaussian quadrature with two points:", result)

result = trapezoid_rule(f, a, b, N)
print("Approximate integral using trapezoid rule:", result)