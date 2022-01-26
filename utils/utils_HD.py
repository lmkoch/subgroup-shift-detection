"""
Helper functions for calculating MMD and performing MMD test

This module contains original code from: https://github.com/fengliu90/DK-for-TST


MIT License

Copyright (c) 2021 Feng Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import torch

def get_item(x):
    """get the numpy value from a torch tensor."""
    x = x.cpu().detach().numpy()
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)

    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)

    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, device, dtype, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    mmd_value = get_item(TEMP[0])
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

