#!/usr/bin/python3

import argparse
import os

from core.dataset import dataset_fn
from core.model import model_fn
from core.mmdd import trainer_object_fn
from core.eval import eval

from utils.config import create_exp_from_config, load_config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run single experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_dir", action="store", type=str, help="experiment folder",
        default='./experiments/hypothesis-tests/mnist'
    )
    parser.add_argument(
        "--config_file", action="store", type=str, help="config file", 
        default='./experiments/hypothesis-tests/mnist/5c3010e7e9f5de06c7d55ecbed422251/config.yaml'
    )  
    parser.add_argument(
        "--seed", action="store", default=1000, type=int, help="random seed",
    )
    parser.add_argument(
        "--eval_splits", action="store", default=[], nargs='+', 
        help="List of splits to be evaluated, e.g. --eval_splits validation test",
    )
      
    args = parser.parse_args()

    # Creates experiment folder and places config file inside
    # (overwrites, if already there)
    exp_name = create_exp_from_config(args.config_file, args.exp_dir)
    params = load_config(args.config_file)

    log_dir = os.path.join(args.exp_dir, exp_name)

    ###############################################################################################################################
    # Preparation
    ###############################################################################################################################
   
    dataloader = dataset_fn(seed=args.seed, params_dict=params['dataset'])
    model  = model_fn(seed=args.seed, params=params['model'])
    trainer = trainer_object_fn(model=model, dataloaders=dataloader, seed=args.seed, 
                                log_dir=log_dir, **params['trainer'])

    ###############################################################################################################################
    # Run training
    ###############################################################################################################################

    trainer.train()

    ###############################################################################################################################
    # Eval MMD-D and MUKS on various sample sizes
    ###############################################################################################################################

    for split in args.eval_splits:
        eval(args.exp_dir, exp_name, params, args.seed, split, 
             sample_sizes=[10, 30, 50, 100, 200, 500], num_reps=100, num_permutations=1000)
        
    print('done')



