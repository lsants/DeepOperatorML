
import torch
import torch.nn as nn
import torch.nn.init as init
from scipy.special import eval_legendre

class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_centers, order):
        super(LegendreKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_centers = n_centers
        self.order = order

        self.centers = nn.Parameter(torch.empty(n_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(n_centers, order + 1, output_dim))
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