"""
Helper functions for calculating MMD

Refactored implementations w.r.t. utils_HD.py

Copyright (c) 2022 Lisa Koch
"""

import torch
   
   
def _mmd2_and_variance(K_XX, K_XY, K_YY, biased=False):
    m = K_XX.shape[0]  # Assumes X, Y are same shape

    hh = K_XX + K_YY - K_XY - K_XY.transpose(0,1)   
    hh_diag = torch.diagonal(hh)
    
    if biased:
        mmd2 = hh.sum() / (m*m)
    else:   
        mmd2 = (hh.sum() - hh_diag.sum()) / (m * (m-1))

    V1 = torch.dot(hh.sum(1)/m,hh.sum(1)/m) / m
    V2 = hh.sum() / m**2
    var_est = 4*(V1 - V2**2)

    return mmd2, var_est, None

def mmd_and_var(f_x, f_y, x, y, f_sigma, sigma, epsilon=10**-10, biased=False):
    
    K_xx = kernel_liu(f_x, f_x, x, x, f_sigma, sigma, epsilon)
    K_yy = kernel_liu(f_y, f_y, y, y, f_sigma, sigma, epsilon)
    K_xy = kernel_liu(f_x, f_y, x, y, f_sigma, sigma, epsilon)
    
    return _mmd2_and_variance(K_xx, K_xy, K_yy, biased=biased)

def kernel_liu(f_x, f_y, x, y, f_sigma, sigma, epsilon=10**-10):
    kernel = (1 - epsilon) * gaussian_kernel(f_x, f_y, f_sigma) * gaussian_kernel(x, y, sigma) + epsilon * gaussian_kernel(x, y, sigma)
    return kernel
    
def gaussian_kernel(x, y, sigma):
    """Gaussian kernel. Note that sigma = (2*sig**2)
    """
    ret_val = torch.exp( - pdist(x, y)**2 / sigma)
    return ret_val

def pdist(x, y):
    """calculate pairwise distances on tensors without batch dimension

    Args:
        x ([type]): tensor of shape [n, d]
        y ([type]): tensor of shape [m, d]

    Returns:
        [type]: tensor of shape [n, m]. normed distance!
    """
    
    # create batch dimension to use torch batched pairwise distance calculation
    b_x = x.view(1, *x.shape)
    b_y = y.view(1, *y.shape)
    b_dist = torch.cdist(b_x, b_y, p=2)
    
    # remove batch dimension
    dist = b_dist.view(*b_dist.shape[1:])
    
    return dist
    
    
if __name__ == '__main__':
    pass