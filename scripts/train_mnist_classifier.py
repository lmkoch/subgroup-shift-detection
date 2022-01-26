#!/usr/bin/python3

import argparse
import logging

from core.model import MNISTNet
from core.dataset import dataset_fn
from utils.config import load_config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Train MNIST classifier"
    )
    parser.add_argument(
        "--config_file", action="store", type=str, help="config file", 
        default='./config/classification_mnist.yaml'
    )  
    parser.add_argument(
        "--seed", dest="seed", action="store", default=1000, type=int, help="",
    )
   
    args = parser.parse_args()
    
    params = load_config(args.config_file)
        
    ###############################################################################################################################
    # Data preparation
    ###############################################################################################################################

    dataloader = dataset_fn(seed=args.seed, params_dict=params['dataset'])

    ###############################################################################################################################
    # Prepare model and training
    ###############################################################################################################################
    
    if params['model']['task_classifier_type'] == 'mnist':
        model = MNISTNet(n_outputs=params['model']['n_outputs'],
                         checkpoint_path=params['model']['task_classifier_path'],
                         download=False)
    else:
        raise NotImplementedError     
       
    model.train_model(dataloader_train=dataloader['train']['p'])     
    acc = model.eval_model(dataloader=dataloader['validation']['p'])

    logging.info(f'Val acc: {acc}')

    logging.info('done')



