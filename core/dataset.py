from typing import Dict
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

from utils.helpers import balanced_weights


def dataset_fn(seed: int, params_dict) -> Dict:
    """
    Returns data loaders for the given config
    Args:
        seed: random seed that will make shuffling and other random operations deterministic
    Returns:
        data_loaders: containing "train", "validation" and "test" data loaders
    """    
    np.random.seed(seed)

    required_keys = ['ds', 'dl']

    for ele in required_keys:
        assert ele in params_dict

    # TODO input validation - check that params_dict contains all the right keys

    params_ds = params_dict['ds']
    params_dl = params_dict['dl']
    
    dataloader = {'train': {}, 'val': {}, 'test': {}}
    for p_q in ['p', 'q']:
        
        dataset = get_dataset(params_ds[p_q]['dataset'], 
                              params_ds[p_q]['data_root'], 
                              img_size=params_ds['img_size'],
                              preproc_mean=params_ds['mean'],
                              preproc_std=params_ds['std'])        
        
        for split in ['train', 'val', 'test']:

            dataloader[split][p_q] = get_dataloader(dataset[split], 
                                                    batch_size=params_dl['batch_size'],
                                                    use_sampling=params_dl[p_q]['use_sampling'],
                                                    sampling_by_variable=params_dl[p_q]['sampling_by_variable'],
                                                    sampling_weights=params_dl[p_q]['sampling_weights'],
                                                    num_workers=params_dl['num_workers'],
                                                    pin_memory=params_dl['pin_memory']
                                                    )
        
    return {
        "train": dataloader['train'],
        "validation": dataloader['val'],
        "test": dataloader['test'],
    }
    
        

def get_dataset(dataset_type: str, data_root: str, 
                img_size, preproc_mean, preproc_std):

    transform = transforms.Compose([transforms.Resize(img_size), 
                        transforms.ToTensor(),
                        transforms.Normalize(preproc_mean, preproc_std)])

    dataset = {}

    if dataset_type == 'mnist':        
                        
        mnist_train = datasets.MNIST(data_root, 
                                    transform=transform,
                                    download=True, 
                                    train=True)

        dataset['test'] = datasets.MNIST(data_root,
                                        transform=transform, 
                                        download=True, 
                                        train=False)

        train_indices, val_indices, _, _ = train_test_split(
            range(len(mnist_train)),
            mnist_train.targets,
            stratify=mnist_train.targets,
            test_size=1./6,
        )

        # generate subset based on indices
        dataset['train'] = Subset(mnist_train, train_indices)
        dataset['val'] = Subset(mnist_train, val_indices)

    elif dataset_type == 'camelyon':

        full_dataset = LisaCamelyon17Dataset(root_dir=data_root, split_scheme='vanilla', 
                                             download=True)
        
        dataset['train'] = full_dataset.get_subset('train', transform=transform)
        dataset['val'] = full_dataset.get_subset('val', transform=transform)
        dataset['test'] = full_dataset.get_subset('test', transform=transform)

    else:
        raise NotImplementedError(f'Dataset not implemented: {dataset_type}')

    return dataset


def get_dataloader(dataset, batch_size: int,
                   use_sampling: bool,
                   sampling_by_variable: str,
                   sampling_weights: list,
                   num_workers: int = 4,
                   pin_memory: bool = True
                   ):
    """Get dataloader based on a dataset and minibatch sampling strategy 

    Args:
        dataset (VisionDataset): 
        data_params (Dict): Minibatch sampling strategy
        batch_size (int): 

    Returns:
        Dataloader: 
    """

    if use_sampling:
        # weights for balanced minibatches
        weights_train = balanced_weights(dataset, 
                                         rebalance_weights=sampling_weights, 
                                         balance_variable=sampling_by_variable)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))  
        dataloader_kwargs = {'sampler': sampler}
    else:     
        dataloader_kwargs = {'shuffle': True}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True, 
        num_workers=num_workers,
        pin_memory=pin_memory, 
        **dataloader_kwargs
    )

    return dataloader

class SubgroupSampler(torch.utils.data.sampler.Sampler):
    """Sample only specific labels

    """
    def __init__(self, data_source, label=5, subgroup='targets'):
        
        subgroup = getattr(data_source, subgroup).clone().detach()
        
        self.mask = subgroup == label
        self.indices = torch.nonzero(self.mask)
        self.data_source = data_source

    def __iter__(self):
        
        return iter([self.indices[i].item() for i in torch.randperm(len(self.indices))])    

    def __len__(self):
        return len(self.indices)
    
class LisaCamelyon17Dataset(Camelyon17Dataset):

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = LisaWILDSSubset(self, split_idx, transform)
        return subset
class LisaWILDSSubset(WILDSDataset):

    def __init__(self, dataset, indices, transform, do_transform_y=False):
        """
        This acts like `torch.utils.data.Subset`, but on `WILDSDatasets`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.

        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y

    def __getitem__(self, idx):
        x, y, _ = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]
    
    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)

    @property
    def hospitals(self):
        return self.metadata_array[:, 0]

