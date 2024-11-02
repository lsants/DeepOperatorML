import torch

class TestEvaluator:
    def __init__(self, model, error_norm):
        self.model = model
        self.error_norm = error_norm

    def __call__(self, g_u, pred):
        self.model.eval()
        with torch.no_grad():
            test_error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return test_error.detach().numpy()