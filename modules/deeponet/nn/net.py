
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from scipy.special import eval_legendre


# neural network
class FCNet(torch.nn.Module):
    def __init__(self, num_ins=3,
                 num_outs=3,
                 num_layers=10,
                 hidden_size=50,
                 activation=torch.nn.Tanh):
        super(FCNet, self).__init__()

        layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = activation

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.eps = 1e-7

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)

        torch.clip(x, -1.+self.eps, 1-self.eps)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

class ChebyKAN(nn.Module):
    def __init__(self):
        super(ChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(2, 32, 8)
        self.ln1 = nn.LayerNorm(32)
        self.chebykan2 = ChebyKANLayer(32, 32, 8)
        self.ln2 = nn.LayerNorm(32)
        self.chebykan3 = ChebyKANLayer(32, 3, 8)

    def forward(self, x):
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x

class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, order):
        super(LegendreKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.order = order

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, order + 1, output_dim))
        init.xavier_uniform_(self.weights)

    def legendre_polynomials(self, x, order):
        """
        Computes Legendre polynomials up to the given order.
        """
        P = torch.stack([torch.from_numpy(eval_legendre(n, x.detach().cpu().numpy())).to(x.device).float() for n in range(order + 1)], dim=-1)
        return P

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        normalized_distances = 2 * (distances - distances.min()) / (distances.max() - distances.min()) - 1  # Normalizing to [-1, 1]

        basis_values = self.legendre_polynomials(normalized_distances, self.order)
        weighted_basis = torch.einsum('bcn,cnf->bcf', basis_values, self.weights)

        output = torch.sum(weighted_basis, dim=1)
        return output


class LegendreKAN(nn.Module):
    def __init__(self):
        super(LegendreKAN, self).__init__()
        self.legendre_kan_layer1 = LegendreKANLayer(2, 16, 100, 5)
        self.act1 = nn.Tanh()
        self.legendre_kan_layer2 = LegendreKANLayer(16, 16,100, 5)
        self.act2 = nn.Tanh()
        self.legendre_kan_layer3 = LegendreKANLayer(16, 3, 100, 5)
        #self.output_weights = nn.Parameter(torch.empty(hidden_dim, output_dim))
        #init.xavier_uniform_(self.output_weights)

    def forward(self, x):
        x = self.legendre_kan_layer1(x)
        x = self.act1(x)
        x = self.legendre_kan_layer2(x)
        x = self.act2(x)
        x = self.legendre_kan_layer3(x)
#        x = F.relu(x)
#        x = torch.matmul(x, self.output_weights)
        return x

# This is inspired by Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim   = output_dim
        self.a        = a
        self.b        = b
        self.degree   = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0: ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a-self.b) + (self.a+self.b+2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k  = (2*i+self.a+self.b)*(2*i+self.a+self.b-1) / (2*i*(i+self.a+self.b))
            theta_k1 = (2*i+self.a+self.b-1)*(self.a*self.a-self.b*self.b) / (2*i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            theta_k2 = (i+self.a-1)*(i+self.b-1)*(2*i+self.a+self.b) / (i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :, i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

class JacobiKAN(nn.Module):
    def __init__(self):
        super(JacobiKAN, self).__init__()
        self.jacobikan1 = JacobiKANLayer(2, 32, 3, a=1.0, b=1.0)
#        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.ln1 = nn.Tanh()
        self.jacobikan2 = JacobiKANLayer(32, 32, 3, a=1.0, b=1.0)
        self.ln2 = nn.Tanh()
        self.jacobikan3 = JacobiKANLayer(32, 3, 3, a=1.0, b=1.0)
#        self.ln2 = nn.LayerNorm(32)
#        self.ln2 = nn.LayerNorm(32)
        self.ln3 = nn.Tanh()
#        self.jacobikan4 = JacobiKANLayer(32, 3, 3, a=1.0, b=1.0)

    def forward(self, x):
        x = x.view(-1, 2)  # Flatten the images
        x = self.jacobikan1(x)
#        x = self.ln1(x)
        x = self.jacobikan2(x)
#        x = self.ln2(x)
        x = self.jacobikan3(x)
#        x = self.ln3(x)
#        x = self.jacobikan4(x)
        return x
