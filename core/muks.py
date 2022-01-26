import logging
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp
    
def mass_ks_test(x, y):
    """mass-univariate two-sample kolmogorov-smirnov test

    Args:
        x ([type]): 1D numpy array
        y ([type]): 1D numpy array
    """
    
    pvals = []
    num_dims = x.shape[1]
    for dim in range(num_dims):
        _, p_val = ks_2samp(x[:, dim], y[:, dim])
        pvals.append(p_val)
        
    corrected_pval = np.min(np.array(pvals)) * num_dims
    
    return np.minimum(corrected_pval, 1.0)


def muks(dataloader_p, dataloader_q, num_repetitions, model,
                         alpha=0.05):
    """Evation of MUKS test

    Returns:
        power and type 1 err 
    """
            
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
               
    model = model.to(device)
    
    sm = nn.Softmax(dim=1)
    
    iterator_p = enumerate(dataloader_p)
    iterator_q = enumerate(dataloader_q)
        
    count_rejects = 0
    count_rejects_h0 = 0
    
    model.eval()
    with torch.no_grad():   
        for idx in range(num_repetitions):
            try:
                _, (x_p, _) = next(iterator_p)
                _, (x_p2, _) = next(iterator_p)
                batch_idx, (x_q, _) = next(iterator_q)
            except:
                logging.info(f'{num_repetitions} larger than dataset size. \
                      Wrap around after {batch_idx + 1} batches.')
                iterator_p = enumerate(dataloader_p)
                iterator_q = enumerate(dataloader_q)
                _, (x_p, _) = next(iterator_p)
                _, (x_p2, _) = next(iterator_p)
                batch_idx, (x_q, _) = next(iterator_q)

            x_p, x_q, x_p2 = x_p.to(device), x_q.to(device), x_p2.to(device)

            outputs = model(x_p)
            softmax_p = sm(outputs)
            
            outputs_p2 = model(x_p2)
            softmax_p2 = sm(outputs_p2)
            
            outputs_q = model(x_q)
            softmax_q = sm(outputs_q)

            pval = mass_ks_test(softmax_p.cpu().numpy(), softmax_q.cpu().numpy())
            pval0 = mass_ks_test(softmax_p.cpu().numpy(), softmax_p2.cpu().numpy())
            
            count_rejects += pval < alpha
            count_rejects_h0 += pval0 < alpha

    reject_rate = count_rejects / (idx+1)  
    reject_rate_h0 = count_rejects_h0 / (idx+1)             
    
    return reject_rate, reject_rate_h0


if __name__ == "__main__":
    pass



