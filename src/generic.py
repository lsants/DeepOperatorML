import numpy as np

def compute_integral(coefs):
    alpha, beta, gamma, B = coefs
    return (alpha / 3) * B**3 + (beta / 2) * B**2 + \
        gamma * B
