import numpy as np

def gauss_quadrature_two_points(f_values, a, b):
    f_values = np.asarray(f_values)
    # Nodes and weights for 2-point Gauss-Legendre quadrature
    x = np.array([-0.5773502691896257, 0.5773502691896257])
    w = np.array([1.0, 1.0])

    transformed_x = (b - a) / 2 * x + (a + b) / 2

    # Interpolate f_values to match the nodes (this step assumes f_values corresponds to evenly spaced x-values)
    interpolated_f_values = np.interp(transformed_x, np.linspace(a, b, len(f_values)), f_values)

    integral = np.sum(w * interpolated_f_values)

    return integral * (b - a) / 2

def trapezoid_rule(f_values, a, b, N):
    f_values = np.asarray(f_values)
    boundary_terms = 0.5 * (f_values[0] + f_values[-1])
    inner_terms = np.sum(f_values[1:-1])
    integral = ((b - a) / (N - 1)) * (inner_terms + boundary_terms)

    return integral

if __name__ == '__main__':

    f = lambda x: np.cos(x)

    # Define integration limits
    a = 0.1
    b = 1
    N = 10000

    f_values = [f(x) for x in range(100)]
    result = gauss_quadrature_two_points(f_values, a, b)
    print("Approximate integral using Gaussian quadrature with two points:", result)

    result = trapezoid_rule(f_values, a, b, N)
    print("Approximate integral using trapezoid rule:", result)