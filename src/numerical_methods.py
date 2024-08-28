import numpy as np

def gauss_quadrature_two_points(f_values, a, b):
    # Nodes and weights for 2-point Gauss-Legendre quadrature
    x = np.array([-np.sqrt(3)/3, np.sqrt(3)/3])
    w = np.array([1.0, 1.0])

    _, cols = f_values.shape
    x_values = np.linspace(a, b, cols)
    transformed_x = (b - a) / 2 * x + (a + b) / 2

    # Interpolate f_values to match the nodes
    interpolated_f_values = np.array([np.interp(transformed_x, x_values, f_values_row)
                              for f_values_row in f_values
                              ])
    integral = np.sum(w * interpolated_f_values, axis=1) * (b - a) / 2

    return integral

def trapezoid_rule(f_values, a, b, N):
    boundary_terms = 0.5 * (f_values[:,0] + f_values[:,-1])
    inner_terms = np.sum(f_values[:,1:-1], axis=1)
    integral = ((b - a) / (N - 1)) * (inner_terms + boundary_terms)

    return integral

if __name__ == '__main__':
    a = 0.1
    b = 1
    N = 100

    x = np.linspace(a,b, N).reshape(1,-1)
    f = np.cos(x)
    result = trapezoid_rule(f, a, b, N)
    print("Approximate integral using trapezoid rule:", result.item())
    
    result = gauss_quadrature_two_points(f, a, b)
    print("Approximate integral using Gaussian quadrature with two points:", result.item())
