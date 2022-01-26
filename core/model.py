from typing import Dict
import os
import logging

import torch
import torch.nn as nn
import torchvision.models as models

import torch.optim as optim
from torch.nn.functional import one_hot

import numpy as np

from wilds.datasets.download_utils import extract_archive
import gdown


def get_classification_model(params):
        
    checkpoint_path = params['task_classifier_path']
        
    if params['task_classifier_type'] == 'mnist': 
        model = MNISTNet(checkpoint_path=checkpoint_path)
    elif params['task_classifier_type'] == 'camelyon': 
        outer_model = CamelyonDensenet(checkpoint_path=checkpoint_path)
        model = outer_model.model
    else:
        raise NotImplementedError
    
    return model

class MNISTNet(nn.Module):
    """Adaptable input channels, 
    rest from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    """
    def __init__(self, in_channels=1, n_outputs=10, checkpoint_path=None, download=True):
        super(MNISTNet, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet18(pretrained=True)

        # Change the input layer to take Grayscale image, instead of RGB images. 
        # Hence in_channels is set as 1 or 3 respectively
        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_outputs)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.n_outputs = n_outputs

        out_dir = os.path.dirname(checkpoint_path)
        os.makedirs(out_dir, exist_ok=True)

        self.checkpoint_path = checkpoint_path         
    
        self.logger = logging.getLogger('mnist_classification')
        self.logger.addHandler(logging.FileHandler(os.path.join(out_dir, 'mnist.log')))
        self.logger.setLevel(logging.INFO)        

        trained_models_url = 'https://drive.google.com/uc?id=1fYmNcgvm91YnMRX4isGAXVors9I17oWD'       
        archive = os.path.join(out_dir, 'archive.tar.gz')
                  
        if os.path.exists(checkpoint_path):
            self.load_checkpoint()
        else:
            if download:
                gdown.download(trained_models_url, archive, quiet=False)

                self.logger.info("Extracting {} to {}".format(archive, out_dir))
                extract_archive(archive, out_dir, True)
                
                self.load_checkpoint()       
            else:
                self.logger.warning('Dataset does not exist. Download it or train the model')
                
    def forward(self, x):
        return self.model(x)

    def save_results(self, epoch, batch_loss):
        
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': batch_loss},
                   self.checkpoint_path)

    def train_model(self, dataloader_train):

        self.model.train()

        for epoch in range(3):

            running_loss = 0.0
            for i, data in enumerate(dataloader_train):
                inputs, labels = data

                if self.n_outputs == 9:
                    # adjustments specific for the 9-way classificaiton
                    # TODO: currently hardcoded to exclude 5. Could make configurable
                    labels_onehot = one_hot(labels, num_classes=10).type(torch.float)

                    class_idx = torch.ones(10, dtype=torch.bool)
                    class_idx[5] = 0
                    
                    labels = labels_onehot[:, class_idx]
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    self.logger.info('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
        
        self.save_results(epoch, running_loss / 100)


    def load_checkpoint(self):
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_loss = checkpoint['loss']
        
        self.model = self.model.to(self.device)
        
        return epoch, batch_loss

    def eval_model(self, dataloader):

        self.model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs

        correct = 0
        total = 0

        with torch.no_grad():
            for _, (x, y) in enumerate(dataloader):

                outputs = self.model(x)
                                
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                
                if self.n_outputs == 9:
                    predicted[predicted>4] += 1
                
                total += predicted.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total
        self.logger.info(f'Val accuracy: {accuracy}')

        return accuracy



class CamelyonDensenet(nn.Module): 
    """This is a wrapper around densenet to allow loading trained
    Camelyon ERM models.
    """
    def __init__(self, checkpoint_path=None, download=True):
        super().__init__()
        self.model = models.densenet121(num_classes=2)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        out_dir = os.path.dirname(checkpoint_path)
        os.makedirs(out_dir, exist_ok=True)
        
        self.logger = logging.getLogger('camelyon_classification')
        self.logger.addHandler(logging.FileHandler(os.path.join(out_dir, 'camelyon.log')))
        self.logger.setLevel(logging.INFO) 
         
        self.checkpoint_path = checkpoint_path               

        trained_models_url = 'https://drive.google.com/uc?id=1fYmNcgvm91YnMRX4isGAXVors9I17oWD'       
        archive = os.path.join(out_dir, 'archive.tar.gz')

        if os.path.exists(checkpoint_path):
            self.load_checkpoint()
        else:
            if download:
                gdown.download(trained_models_url, archive, quiet=False)

                self.logger.info("Extracting {} to {}".format(archive, out_dir))
                extract_archive(archive, out_dir, True)
                
                self.load_checkpoint()       
            else:
                self.logger.warning('Dataset does not exist. Download it or train the model')

    def load_checkpoint(self):
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['algorithm'])
        
        self.model = self.model.to(self.device)
        
        return self.model


# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self, n_channels=1, img_size=32):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(n_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 128))
        
        # Other parameters
        self.epsilonOPT = nn.Parameter(torch.log(torch.rand(1)* 10 ** (-10)))
        self.sigmaOPT = nn.Parameter(torch.ones(1) * np.sqrt(2*32*32))
        self.sigma0OPT = nn.Parameter(torch.ones(1) * np.sqrt(0.005))        

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

    @property
    def ep(self):
        return torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))

    @property
    def sigma_sq(self):
        return self.sigmaOPT ** 2

    @property
    def sigma0_sq(self):
        return self.sigma0OPT ** 2


def model_fn(seed: int, params: Dict) -> torch.nn.Module:
    """
    Builds a model object for the given config
    Args:
        data_loaders: a dictionary of data loaders
        seed: random seed (e.g. for model initialization)
    Returns:
        Instance of torch.nn.Module
    """

    required_arguments = ['channels', 'img_size']
    for ele in required_arguments:
        assert ele in params
    
    # just for safety: remove any potential unexpected items    
    params = {k: v for k, v in params.items() if k in required_arguments}    
        
    torch.manual_seed(seed)  # for reproducibility (almost)

    model = Featurizer(n_channels=params['channels'], 
                   img_size=params['img_size'])

    return model
