import os
import argparse

from utils.config import load_config, save_config


def exps_mmd_final_params(config_file, exp_dir, subgroup_index=5):
    #####################################################################
    # subpopulation shifts - final experiments on best hyperparams
    #####################################################################

    config = load_config(config_file)
    
    weights = [1, 5, 10, 100]
    for weight in weights:
        config['dataset']['dl']['q']['sampling_weights'][subgroup_index] = weight
        save_config(config, exp_dir)

def mnist_mmd_hyperparam_sweep(config_file, exp_dir):
    """Hyperparameter sweep for subpopulation shifts on MNIST

    Args:
        config_file: config template for MNIST
    """

    os.makedirs(exp_dir, exist_ok=True)

    config = load_config(config_file)

    weight = 5
    lambdas = [10**-4, 10**-2, 10**0, 10**2]
    loss_types = ['mmd2', 'original', 'additive']

    config['dataset']['dl']['q']['sampling_weights'][5] = weight

    for loss_type in loss_types:
        for lam in lambdas:
                        
            config['trainer']['loss_type'] = loss_type
            config['trainer']['loss_lambda'] = lam
            
            save_config(config, exp_dir)
            
            if loss_type == 'mmd2':
                # lambda param does not exist - no need for hyperparam sweep
                break
    

if __name__ == '__main__':
    
    # Each experiment is located in a folder, which contains 
    # - full specification config file
    # - trained models, intermediate results
    # - final results and figures 
    #
    # Here, we generate all experiment configurations
    # 1. Load configuration templates
    # 2. Adjust specific parameters (e.g. subshift strength)
    # 3. Hash the config
    # 3. Save full config in unique (hash) experiment folder
    #
    # To actually execute the experiments (e.g. train and evaluate tests), separate scripts
    # are subsequently run that only take a config file as an input
    #

    parser = argparse.ArgumentParser(
        description="Run single experiment"
    )
    parser.add_argument(
        "--exp_base_dir", action="store", type=str, help="experiment folder",
        default='./experiments'
    )
      
    args = parser.parse_args()
        
    exp_dir = os.path.join(args.exp_base_dir, 'hypothesis-tests', 'mnist_hyperparam')
    config_file = './config/mnist.yaml'
    mnist_mmd_hyperparam_sweep(config_file, exp_dir)

    exp_dir = os.path.join(args.exp_base_dir, 'hypothesis-tests', 'mnist')
    config_file = './config/mnist.yaml'    
    exps_mmd_final_params(config_file, exp_dir, subgroup_index=5)

    exp_dir = os.path.join(args.exp_base_dir, 'hypothesis-tests', 'camelyon')
    config_file = './config/camelyon.yaml'
    exps_mmd_final_params(config_file, exp_dir, subgroup_index=2)

    print('Prepped experiments.')