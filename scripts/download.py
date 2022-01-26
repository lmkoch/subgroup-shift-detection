#!/usr/bin/python3
"""Download all prerequisites: datasets and pretrained models.
"""


import argparse

from torchvision import datasets
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

from core.model import MNISTNet, CamelyonDensenet

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Download data and pretrained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_mnist", action="store", type=str, help="MNIST data root",
        default='./data'
    )
    parser.add_argument(
        "--data_camelyon", action="store", type=str, help="Camelyon17 data root",
        default='./data'
    )    
    parser.add_argument(
        "--model_mnist", action="store", type=str, help="MNIST task classifier path",
        default='./experiments/classification-models/mnist.pt'
    ) 
    parser.add_argument(
        "--model_mnist_no5", action="store", type=str, help="MNIST task (no digit 5) classifier path",
        default='./experiments/classification-models/mnist_no5.pt'
    ) 
    parser.add_argument(
        "--model_camelyon", action="store", type=str, help="Camelyon17 task classifier path",
        default='./experiments/classification-models/camelyon17_seed9_best_model.pth'
    ) 
      
    args = parser.parse_args()

    MNISTNet(n_outputs=10, checkpoint_path=args.model_mnist, download=True)
    MNISTNet(n_outputs=9, checkpoint_path=args.model_mnist_no5, download=True)
    CamelyonDensenet(checkpoint_path=args.model_camelyon, download=True)

    datasets.MNIST(args.data_mnist, download=True)
    Camelyon17Dataset(root_dir=args.data_camelyon, download=True)


    print('done')



