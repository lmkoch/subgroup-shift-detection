#!/usr/bin/python3

import os
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import griddata

import torch
import torch.nn as nn

from core.model import MNISTNet
from core.dataset import dataset_fn
from utils.config import load_config, create_exp_from_config
from utils.helpers import set_rcParams

from core import odin

def run_grid_search(dl_in, dl_out, model, temperatures, epsilons, num_img, results_gridsearch_csv):
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    columns = ['temperature', 'epsilon', 'method', 'rocauc', 'fpr95']
    df = pd.DataFrame(columns=columns)
    
    for temper in temperatures:
        for epsi in epsilons:
        
            df_in = odin.predict_scores(model, device, dl_in, epsi, temper, num_img)
            df_out = odin.predict_scores(model, device, dl_out, epsi, temper, num_img)
        
            for method in ['base', 'odin']:
                roc_auc, fpr95 = odin.evaluate_scores(df_in[df_in['method'] == method]['score'], 
                                                df_out[df_out['method'] == method]['score'])

                row = {'temperature': temper, 'epsilon': epsi, 'method': method,
                    'rocauc': roc_auc, 'fpr95': fpr95}            
                df = df.append(row, ignore_index=True)
            
            print(f'-----------------------------------------------------')
            print(f'Hyperparams t={temper}, eps={epsi}')
            print(f'AUC: {roc_auc}')
            print(f'FPR95: {fpr95}')

    # validation results:        
    df.to_csv(results_gridsearch_csv)
        
def plot_gridsearch_results(df, temperatures, epsilons, log_dir):
    
    set_rcParams()
    
    X, Y = np.meshgrid(temperatures, epsilons)
    subset = df.loc[df['method'] == 'odin']

    for measure in ['rocauc', 'fpr95']:
        fig, ax = plt.subplots(figsize=(3, 3))

        grid_z0 = griddata(subset[['temperature', 'epsilon']], subset[measure], (X, Y), method='nearest')
        
        cmap = 'crest'
        if measure == 'rocauc':
            vmin, vmax = 0.5, 1.0
            vmin, vmax = None, None
            cmap = f'{cmap}_r'

        elif measure == 'fpr95':
            vmin, vmax = 0.0, 1.0
            vmin, vmax = None, None
        
        ax = sns.heatmap(grid_z0, annot=True, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'temperature $\tau$')
        ax.set_xticklabels(temperatures)
        ax.set_ylabel(r'perturbation $\epsilon$')
        ax.set_yticklabels(epsilons)
        
        file_name = os.path.join(log_dir, f'ood_{measure}.pdf')
        fig.savefig(file_name)

def eval_best_param(dl_in, dl_out, model, gridsearch_df, results_csv):
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    
    subset = gridsearch_df.loc[gridsearch_df['method'] == 'odin']
    best_row = subset[subset.fpr95 == subset.fpr95.min()]

    temper = best_row['temperature'].values[0]
    epsi = best_row['epsilon'].values[0]

    num_img = len(dl_in.dataset)

    df_in = odin.predict_scores(model, device, dl_in, epsi, temper, num_img)
    df_out = odin.predict_scores(model, device, dl_out, epsi, temper, num_img)

    columns = ['temperature', 'epsilon', 'method', 'rocauc', 'fpr95']
    df = pd.DataFrame(columns=columns)

    for method in ['base', 'odin']:
        roc_auc, fpr95 = odin.evaluate_scores(df_in[df_in['method'] == method]['score'], 
                                        df_out[df_out['method'] == method]['score'])

        row = {'temperature': temper, 'epsilon': epsi, 'method': method,
            'rocauc': roc_auc, 'fpr95': fpr95}            
        df = df.append(row, ignore_index=True)

    df.to_csv(results_csv)        

def main(exp_dir, config_file, seed, run_gridsearch=True, run_plot=True, run_eval=True):

    exp_name = create_exp_from_config(config_file, args.exp_dir)

    print(f'run ODIN for configuration: {exp_name}')

    
    # paths
    log_dir = os.path.join(exp_dir, exp_name)
    results_gridsearch_csv = os.path.join(log_dir, 'ood_gridsearch.csv')
    results_test_csv = os.path.join(log_dir, 'ood_test.csv')

    # hyperparam range:   
    temperatures = [1, 10, 100, 1000]
    epsilons = [0, 0.001, 0.002, 0.003, 0.004]

    ###############################################################################################################################
    # Data preparation
    ###############################################################################################################################
    
    params = load_config(config_file)

    dataloader = dataset_fn(seed=seed, params_dict=params['dataset'])

    if params['model']['task_classifier_type'] == 'mnist':
        model = MNISTNet(n_outputs=params['model']['n_outputs'],
                         checkpoint_path=params['model']['task_classifier_path'],
                         download=True)
    else:
        raise NotImplementedError        

    ###############################################################################################################################
    # Hyperparameter search and evaluation on test fold
    ###############################################################################################################################

    model.eval()
          
    if not run_gridsearch and not os.path.exists(results_gridsearch_csv):
        raise ValueError('must run grid search.')
          
    if run_gridsearch:
        num_img = 1000
        dl_in = dataloader['validation']['p']
        dl_out = dataloader['validation']['q']
        run_grid_search(dl_in, dl_out, model, temperatures, epsilons, 
                    num_img, results_gridsearch_csv)
        
    if run_plot:
        df = pd.read_csv(results_gridsearch_csv)
        plot_gridsearch_results(df, temperatures, epsilons, log_dir)
      
    if run_eval:
        dl_in = dataloader['test']['p']
        dl_out = dataloader['test']['q']
        df = pd.read_csv(results_gridsearch_csv)
        eval_best_param(dl_in, dl_out, model, df, results_test_csv)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run single ODIN experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_dir", action="store", type=str, help="experiment folder", default='./experiments/individual-ood'
    )
    parser.add_argument(
        "--config_file", action="store", type=str, help="config file", default='./config/odin_mnist_5x100.yaml'
    )  
    parser.add_argument(
        "--seed", dest="seed", action="store", default=1000, type=int, help="random seed",
    )
    parser.add_argument('--run_gridsearch', default=True, type=bool, help='gridsearch flag')
    parser.add_argument('--run_plot', default=True, type=bool, help='plot flag')
    parser.add_argument('--run_eval', default=True, type=bool, help='eval flag')
   
    args = parser.parse_args()
             
    os.makedirs(args.exp_dir, exist_ok=True)
                       
    main(args.exp_dir, args.config_file, args.seed, run_gridsearch=args.run_gridsearch, 
         run_plot=args.run_plot, run_eval=args.run_eval)

    print('done')



