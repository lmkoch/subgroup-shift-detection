import os

import pandas as pd
import numpy as np

from core.dataset import dataset_fn
from core.model import model_fn, get_classification_model
from core.mmdd import trainer_object_fn

from core.muks import muks

        
def stderr_proportion(p, n):
    return np.sqrt(p * (1-p) / n)

def eval(exp_dir, exp_name, params, seed, split, sample_sizes=[10, 30, 50, 100, 500], 
         num_reps=100, num_permutations=1000):
    """Analysis of test power vs sample size for both MMD-D and MUKS

    Args:
        exp_dir ([type]): exp base directory
        exp_name ([type]): experiment name (hashed config)
        params (Dict): [description]
        seed (int): random seed
        split (str): fold to evaluate, e.g. 'validation' or 'test
        sample_sizes (list, optional): Defaults to [10, 30, 50, 100, 500].
        num_reps (int, optional): for calculation rejection rates. Defaults to 100.
        num_permutations (int, optional): for MMD-D permutation test. Defaults to 1000.
    """
     
    log_dir = os.path.join(exp_dir, exp_name)
    out_csv = os.path.join(log_dir, f'{split}_consistency_analysis.csv')

    df = pd.DataFrame(columns=['sample_size','power', 'power_stderr', 
                            'type_1err', 'type_1err_stderr', 'method'])

    for batch_size in sample_sizes:
        
        params['dataset']['dl']['batch_size'] = batch_size
        dataloader = dataset_fn(seed=seed, params_dict=params['dataset'])

        # MMD-D
        model  = model_fn(seed=seed, params=params['model'])
        trainer = trainer_object_fn(model=model, dataloaders=dataloader, seed=seed, 
                                    log_dir=log_dir, **params['trainer'])

        res = trainer.performance_measures(dataloader[split]['p'], dataloader[split]['q'], num_batches=num_reps,
                                            num_permutations=num_permutations)
        
        res_mmd = {'exp_hash': exp_name,
            'sample_size': batch_size,
            'power': res['reject_rate'],
            'power_stderr': stderr_proportion(res['reject_rate'], batch_size),
            'type_1err': res['type_1_err'] ,
            'type_1err_stderr': stderr_proportion(res['type_1_err'] , batch_size),
            'method': 'mmd'}          
        
        # MUKS
    
        model = get_classification_model(params['model'])
        reject_rate, type_1_err = muks(dataloader[split]['p'], dataloader[split]['q'], num_reps, model)

        res_rabanser = {'exp_hash': exp_name,
            'sample_size': batch_size,
            'power': reject_rate,
            'power_stderr': stderr_proportion(reject_rate, batch_size),
            'type_1err': type_1_err,
            'type_1err_stderr': stderr_proportion(type_1_err, batch_size),
            'method': 'rabanser'}     
                        
        print('---------------------------------')  
        print(f'sample size: {batch_size}')                                                  
        print(f'mmd: {res_mmd}')                                                  
        print(f'rabanser: {res_rabanser}')                                                  

        df = df.append(pd.DataFrame(res_mmd, index=['']), ignore_index=True)
        df = df.append(pd.DataFrame(res_rabanser, index=['']), ignore_index=True)

    df.to_csv(out_csv)
