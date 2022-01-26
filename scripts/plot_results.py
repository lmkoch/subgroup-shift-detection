import os
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
import argparse
 
import numpy as np
import pandas as pd

import torchvision

from utils.helpers import flatten_dict, set_rcParams
from core.dataset import dataset_fn

def save_example_images(exp_dir, exp_name, seed=1000):
    
    config_file = os.path.join(exp_dir, exp_name, 'config.yaml')
    with open(config_file) as fhandle:
        params = yaml.safe_load(fhandle)
                        
    params['dataset']['dl']['batch_size'] = 64
    dataloader = dataset_fn(seed=seed, params_dict=params['dataset'])

    log_dir = os.path.join(exp_dir, exp_name)
       
    for dist in ['p', 'q']:    
        out_fig = os.path.join(log_dir, f'panel_{dist}.png')  
    
        _, (img, _) = next(enumerate(dataloader['train'][dist]))
        img_grid = torchvision.utils.make_grid(img, normalize=True)
        img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
        plt.imsave(out_fig, img_grid)

def load_all_results(hashes, exp_dir, split='test'):

    all_configs = {}
    all_results = []

    for exp_name in hashes:
        config_file = os.path.join(exp_dir, exp_name, 'config.yaml')
        with open(config_file) as fhandle:
            params = yaml.safe_load(fhandle)
        
        all_configs[exp_name] = flatten_dict(params, sep='_')
    
        results_file = os.path.join(exp_dir, exp_name, f'{split}_consistency_analysis.csv')
        
        if os.path.exists(results_file):
            all_results.append(pd.read_csv(results_file))
        else:
            raise ValueError('results are not available. run consistency analysis first')
                                                    
    configs = pd.DataFrame.from_dict(all_configs, orient='index')
    results = pd.concat(all_results)
    configs.rename_axis('exp_hash', inplace=True)
    results = results.set_index(['exp_hash', 'sample_size'], drop=False)
    df = configs.join(results)
    
    return df

def mmd_model_selection(exp_dir, eval_dir):
    """Compare validation performance for different hyperparameter and model choices for MMD-D
    Requires consistency analysis to have been run before. 

    Raises:
        ValueError: [description]
    """

    set_rcParams()

    hashes = os.listdir(exp_dir)
    df = load_all_results(hashes, exp_dir, split='validation')
    
    # line plot of val rejection rate vs lamba param
    loss_types = ['original']
    loss_labels = ['MMD ratio', 'MMD']

    lambdas = [10**-4, 10**-2, 10**0, 10**2]

    out_fig_lambda_hyperparam = os.path.join(eval_dir, 'lambda_hyperparams.pdf')
    fig, ax = plt.subplots(figsize=(3, 2))

    df = df[df['method'] == 'mmd']
    tmp = df[df['sample_size'] == 50]
    mmd = tmp.loc[tmp['trainer_loss_type']=='mmd2', 'power'].values[0]
    
    tmp = tmp[tmp['trainer_loss_type'] == 'original']
    ax.plot(tmp['trainer_loss_lambda'], tmp['power'],
                marker='o')

    ax.axhline(mmd, color='k', ls='--', label='hi') 


    for hue in loss_types:
        subset = tmp[tmp['trainer_loss_type']==hue]
        ax.fill_between(subset['trainer_loss_lambda'], 
                        subset['power']-subset['power_stderr'], 
                        subset['power']+subset['power_stderr'],
                        alpha=0.5)
    ax.set_ylim(0, 1.05)     
    ax.set(xscale="log")    
    ax.set_xticks(lambdas)
            
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Test power')
    ax.legend(loss_labels, loc='lower right', frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(out_fig_lambda_hyperparam)
    fig.show()
    print('plotted model selection.')

def plot_subgroup_results_mnist_camelyon(eval_dir, exp_dir_mnist, exp_dir_camelyon):

    os.makedirs(eval_dir, exist_ok=True)

    set_rcParams()

    exp_dir = {'mnist':exp_dir_mnist,
               'camelyon': exp_dir_camelyon}
     
    subgroup_idx = {'mnist': 5,
                    'camelyon': 2}
    
    df = {'mnist': None, 'camelyon': None}
    for dataset in ['mnist', 'camelyon']:
        
        hashes = os.listdir(exp_dir[dataset])

        df[dataset] = load_all_results(hashes, exp_dir[dataset], split='test')   
        df[dataset]['mixing_proportions'] = np.array(df[dataset]['dataset_dl_q_sampling_weights'].tolist())[:,subgroup_idx[dataset]]
     
    out_fig_weights = os.path.join(eval_dir, 'panel_complete.png')

    weights = [1, 5, 10, 100]
        
    methods = ['mmd', 'rabanser']
    legend_labels = ['MMD-D (MNIST)', 'MMD-D (Camelyon)', 'MUKS (MNIST)', 'MUKS (Camelyon)']         

    fig, ax = plt.subplots(3, len(weights), figsize=(6, 5), sharey='row')
    
    for idx, weight in enumerate(weights):
        row_ax = ax[0, :]
         
        df['mnist']['dataset'] = 'mnist'
        df['camelyon']['dataset'] = 'camelyon'
                
        whole_df = pd.concat([df['mnist'], df['camelyon']])
        tmp = whole_df[whole_df['mixing_proportions'] == weight].reset_index(drop=True)
        mnist = df['mnist'][df['mnist']['mixing_proportions'] == weight].reset_index(drop=True)
        cam = df['camelyon'][df['camelyon']['mixing_proportions'] == weight].reset_index(drop=True)

        
        row_ax[idx].set_title(f'w={weight}')
        sns.lineplot(data=tmp, x='sample_size', y='power',
                            hue='method', hue_order=methods,
                            style='dataset', markers=True, dashes=False,
                            ax=row_ax[idx])
                
        colors = sns.color_palette()
        
        for cidx, hue in enumerate(methods):
            subset = mnist[mnist['method']==hue]
            row_ax[idx].fill_between(subset['sample_size'], 
                            subset['power']-subset['power_stderr'], 
                            subset['power']+subset['power_stderr'],
                            color=colors[cidx],
                            alpha=0.5)
            
        for cidx, hue in enumerate(methods):
            subset = cam[cam['method']==hue]
            row_ax[idx].fill_between(subset['sample_size'], 
                            subset['power']-subset['power_stderr'], 
                            subset['power']+subset['power_stderr'],
                            color=colors[cidx],
                            alpha=0.5)

            
        row_ax[idx].set_ylim(0, 1.05)     
        row_ax[idx].set(xscale="log")    
        row_ax[idx].grid(axis='x') 

        row_ax[idx].set_ylabel('Test power')   
        row_ax[idx].set_xlabel(r'Sample size m')  
        
        if idx > 0:
            row_ax[idx].get_legend().remove()
        else:
            row_ax[idx].legend(legend_labels, loc='best', frameon=False)           
        
        # images
        # MNIST
        exp_name = mnist.iloc[0]['exp_hash']
        
        img_file = os.path.join(exp_dir['mnist'], exp_name, 'panel_q.png')
        image = plt.imread(img_file)
        ax[1, idx].imshow(image)
        ax[1, idx].axis('off') 


        # Camelyon
        exp_name = cam.iloc[0]['exp_hash']
        
        img_file = os.path.join(exp_dir['camelyon'], exp_name, 'panel_q.png')
        image = plt.imread(img_file)
        ax[2, idx].imshow(image)
        ax[2, idx].axis('off') 
        ax[2, idx].set_ylabel('Camelyon17 examples')  
                   
                   
    plt.tight_layout()
    fig.savefig(out_fig_weights)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_dir_mnist", action="store", type=str, help="mnist experiment folder",
        default='./experiments/hypothesis-tests/mnist'
    )
    parser.add_argument(
        "--exp_dir_camelyon", action="store", type=str, help="camelyon experiment folder",
        default='./experiments/hypothesis-tests/camelyon'
    )
    parser.add_argument(
        "--exp_dir_hyperparam", action="store", type=str, help="hyperparam search exp folder",
        default='./experiments/hypothesis-tests/mnist_hyperparam'
    )
    parser.add_argument(
        "--eval_dir", action="store", type=str, 
        help="Results directory (will be created if it does not exist)",
        default='./experiments/eval'
    ) 
    parser.add_argument(
        "--run_hyperparam", action="store", type=bool, help="",
        default=True
    )   
    args = parser.parse_args()
           
    # plot example data
    hashes = os.listdir(args.exp_dir_mnist)
    for exp_name in hashes:
        save_example_images(args.exp_dir_mnist, exp_name)

    hashes = os.listdir(args.exp_dir_camelyon)
    for exp_name in hashes:
        save_example_images(args.exp_dir_camelyon, exp_name)

    plot_subgroup_results_mnist_camelyon(args.eval_dir, args.exp_dir_mnist, args.exp_dir_camelyon)
                                                   
    # model selection
    if args.run_hyperparam:
        mmd_model_selection(args.exp_dir_hyperparam, args.eval_dir)
    
    print('done')