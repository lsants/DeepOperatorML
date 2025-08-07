import torch

class ChebyshevKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyshevKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.eps = 1e-7
        self.cheby_coeffs = torch.nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

        torch.nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )
        torch.clip(x, -1.+self.eps, 1-self.eps)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )
        y = y.view(-1, self.outdim)

        return y
    
class ChebyshevKAN(torch.nn.Module):
    def __init__(self, input_dim, layers, output_dim, degree):
        super(ChebyshevKAN, self).__init__()
        layers = [input_dim] + layers + [output_dim]
        self.linears = torch.nn.ModuleList()
        self.degree = degree
        n_layers = len(layers)
        for layer_index in range(n_layers - 1):
            self.linears.append(
                ChebyshevKANLayer(
                    layers[layer_index],
                    layers[layer_index + 1],
                    self.degree
                ),
            )
            self.linears.append(torch.nn.LayerNorm(layers[layer_index + 1]))

    def forward(self, x):
        for layer_index in range(len(self.linears) - 1):
            x = self.linears[layer_index](x)
        return x