from typing import Dict, Tuple, List, Any
import os
import logging
import datetime
import pickle
import numpy as np

from functools import partial

import torch
from torch import nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

from utils.mmd import mmd_and_var
from utils import helpers
from utils.utils_HD import TST_MMD_u


def loss_original(feat_p, feat_q, x_p, x_q, sigma, sigma0, ep, lam=10 ** (-6)):
    # Compute Compute J
    
    m, n = x_p.shape[0], x_q.shape[0]    
    mmd2, varEst, _ = mmd_and_var(feat_p, feat_q, x_p.view(m, -1), x_q.view(n, -1), sigma0, sigma, ep)

    return_val = torch.div(-1 * mmd2, torch.sqrt(varEst + lam))
    return return_val

def loss_log_original(feat_p, feat_q, x_p, x_q, sigma, sigma0, ep, lam=10 ** (-6)):
    # Compute Compute J
    numerical_eps = 10**-6
    m, n = x_p.shape[0], x_q.shape[0]    
    mmd2, varEst, _ = mmd_and_var(feat_p, feat_q, x_p.view(m, -1), x_q.view(n, -1), sigma0, sigma, ep)

    return_val = - torch.log(mmd2 + numerical_eps) \
                    + 0.5 * torch.log(varEst + lam)
    return return_val

def loss_additive(feat_p, feat_q, x_p, x_q, sigma, sigma0, ep, lam=10 ** (-6)):
    """As suggested by Arthur Gretton

    Returns:
        [type]: [description]
    """
    m, n = x_p.shape[0], x_q.shape[0]    
    mmd2, varEst, _ = mmd_and_var(feat_p, feat_q, x_p.view(m, -1), x_q.view(n, -1), sigma0, sigma, ep)

    return_val = - mmd2 + lam * torch.sqrt(varEst)

    return return_val    

def loss_log_mmd2(feat_p, feat_q, x_p, x_q, sigma, sigma0, ep):
    # Compute Compute J
    
    numerical_eps = 10 ** (-6)
    m, n = x_p.shape[0], x_q.shape[0]    
    mmd2, _, _ = mmd_and_var(feat_p, feat_q, x_p.view(m, -1), x_q.view(n, -1), sigma0, sigma, ep)

    return_val = - torch.log(mmd2 + numerical_eps)
    
    return return_val    


def loss_mmd2(feat_p, feat_q, x_p, x_q, sigma, sigma0, ep, biased=False):
    # Compute Compute J
    
    m, n = x_p.shape[0], x_q.shape[0]    
    mmd2, _, _ = mmd_and_var(feat_p, feat_q, x_p.view(m, -1), x_q.view(n, -1), sigma0, sigma, ep,
                             biased=biased)

    return_val = - mmd2
    
    return return_val    


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict,
        seed: int,
        log_dir: str,
        eval_config: Dict,
        epochs: int = 5,
        loss_type: str = 'original',
        loss_lambda: float = 10**-6,
        learning_rate: float = 10**-4,
    ) -> None:

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model = model.to(self.device)
                
        self.trainloader_p = dataloaders["train"]['p']
        self.trainloader_q = dataloaders["train"]['q']
        self.valloader_p = dataloaders["validation"]['p']
        self.valloader_q = dataloaders["validation"]['q']
        self.seed = seed
        self.epochs = epochs
        
        self.log_dir = log_dir
        
        self.loss_fn = loss_original
        
        # original
        if loss_type == 'original':
            self.loss_fn = partial(loss_original, lam=loss_lambda)
        elif loss_type == 'log_original':
            self.loss_fn = partial(loss_log_original, lam=loss_lambda)
        elif loss_type == 'log_mmd2':
            self.loss_fn = partial(loss_log_mmd2)
        elif loss_type == 'mmd2':
            self.loss_fn = partial(loss_mmd2)
        elif loss_type == 'additive':
            self.loss_fn = partial(loss_additive, lam=loss_lambda)
        else: 
            self.loss_fn = None
                        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.eval_config = eval_config
        
        self.use_tensorboard = eval_config['use_tensorboard']
        
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)

        self.logger = logging.getLogger('mmd')
        self.logger.addHandler(logging.FileHandler(os.path.join(self.log_dir, 'train.log')))
        self.logger.setLevel(logging.INFO)

        if self.check_if_already_trained():
            self.logger.info('already trained. load model.')
            self.load_results_and_checkpoint()
        

    def train_step(self, x_p: torch.Tensor, x_q: torch.Tensor) -> None:
        
        self.optimizer.zero_grad()
        self.model.train()
        
        feat_p = self.model(x_p)
        feat_q = self.model(x_q)
                
        ep = self.model.ep
        sigma = self.model.sigma_sq
        sigma0_u = self.model.sigma0_sq
                
        loss = self.loss_fn(feat_p, feat_q, x_p, x_q, sigma, sigma0_u, ep)
        loss.backward()
        self.optimizer.step()
               
        return loss.item()

    def performance_measures(self, dataloader_p, dataloader_q, num_batches, num_permutations=100):
        """Calculate a bunch of performance measures

        Returns:
            Dictionary containing all calculated measures as well as fpr and tpr arrays
        """
                
        iterator_p = enumerate(dataloader_p)
        iterator_q = enumerate(dataloader_q)
                
        running_rejects = 0.0
        running_rejects_h0 = 0.0
        running_loss = 0.0
        
        for idx in range(num_batches):
            try:
                _, (x_p, _) = next(iterator_p)
                _, (x_p2, _) = next(iterator_p)
                batch_idx, (x_q, _) = next(iterator_q)
            except:
                self.logger.info(f'{num_batches} larger than dataset size. \
                      Wrap around after {batch_idx + 1} batches.')
                iterator_p = enumerate(dataloader_p)
                iterator_q = enumerate(dataloader_q)
                _, (x_p, _) = next(iterator_p)
                _, (x_p2, _) = next(iterator_p)
                batch_idx, (x_q, _) = next(iterator_q)
            
            x_p, x_q, x_p2 = x_p.to(self.device), x_q.to(self.device), x_p2.to(self.device)
            
            feat_p = self.model(x_p)
            feat_q = self.model(x_q)
                    
            ep = self.model.ep
            sigma = self.model.sigma_sq
            sigma0_u = self.model.sigma0_sq
                
            loss = self.loss_fn(feat_p, feat_q, x_p, x_q, sigma, sigma0_u, ep)

            # Below relies on official MMD-D implementation. 
            # Note: MMD-based loss (above) was reimplemented (refactored and tested) in a redundant 
            #       implementation, but MMD-based actual test (below) uses original implementation by 
            #       Liu et al (2020) because of efficient implementation of permutation test. 
            #       TODO: refactor permutation test as well for readability and no redundancy
            
            # power
            S = torch.cat([x_p, x_q], 0)
            batch_size = x_p.shape[0]
            dtype = x_p.dtype
            Sv = S.view(2 * batch_size, -1)
            
            h_u, _, _ = TST_MMD_u(self.model(S), num_permutations, batch_size, Sv, 
                                  sigma, sigma0_u, ep, 
                                  self.eval_config['alpha'], self.device, dtype)

            # type I error
            S = torch.cat([x_p, x_p2], 0)
            Sv = S.view(2 * batch_size, -1)
            h_u_h0, _, _ = TST_MMD_u(self.model(S), num_permutations, batch_size, Sv, 
                                  sigma, sigma0_u, ep, 
                                  self.eval_config['alpha'], self.device, dtype)

            # Gather results
            running_rejects += h_u
            running_rejects_h0 += h_u_h0
            running_loss += loss.item()
      
        reject_rate = running_rejects / (idx+1)  
        reject_rate_h0 = running_rejects_h0 / (idx+1)             
        avg_loss = running_loss / (idx+1)

        results = {'loss': avg_loss, 'reject_rate': reject_rate, 'type_1_err': reject_rate_h0}
        
        if np.isnan(avg_loss):
            # the results are not valid
            # return nans
            # TODO: this should perhaps be handled in a nicer way
            results = {'loss': avg_loss, 'reject_rate': np.nan, 'type_1_err': np.nan}
            
        return results


    def val_step(self, global_step, generate_img_grid=True):

        self.model.eval()
        
        num_val_batches = self.eval_config['num_eval_batches']
        num_permutations = self.eval_config['n_permute']
        val_results = self.performance_measures(self.valloader_p, self.valloader_q, num_val_batches, num_permutations)
        train_results = self.performance_measures(self.trainloader_p, self.trainloader_q, num_val_batches, num_permutations)
        
        self.logger.debug('val step: create images')

        if generate_img_grid:
            train_results['img_p'] = helpers.make_image_grid(self.trainloader_p, num_images=64)  
            train_results['img_q'] = helpers.make_image_grid(self.trainloader_q, num_images=64)  
        
        if self.use_tensorboard:
            self.writer.add_scalar('val_loss', val_results['loss'], global_step)
            self.writer.add_scalar('val_reject_rate', val_results['reject_rate'], global_step)
            self.writer.add_scalar('val_type_1_err', val_results['type_1_err'], global_step)
            
            self.writer.add_scalar('train_loss', train_results['loss'], global_step)
            self.writer.add_scalar('train_reject_rate', train_results['reject_rate'], global_step)
            self.writer.add_scalar('train_type_1_err', train_results['type_1_err'], global_step)

            self.writer.add_image('train_images_p', train_results['img_p'], global_step)
            self.writer.add_image('train_images_q', train_results['img_q'], global_step)
            
            self.writer.flush()      
        
        return train_results, val_results

    def train(self) -> Tuple[float, Tuple[List[float], int], Dict]:
        torch.manual_seed(self.seed)      

        if self.check_if_already_trained():
            _, _, val_results, train_results = self.load_results_and_checkpoint()
            print('model already trained - do not train again.')
            return

        batches_per_epoch = -1
                
        for epoch in range(self.epochs):

            dl_tr_enumerator = enumerate(self.trainloader_p)
            dl_tr_f_enumerator = enumerate(self.trainloader_q)

            while True:

                try:
                    batch_idx, (imgs_p, _) = next(dl_tr_enumerator)
                    _, (imgs_q, _) = next(dl_tr_f_enumerator)
                except StopIteration:
                    batches_per_epoch = batch_idx + 1
                    break
                    
                imgs_p = imgs_p.to(self.device)
                imgs_q = imgs_q.to(self.device)

                batch_loss = self.train_step(imgs_p, imgs_q)

                global_step = epoch * batches_per_epoch + batch_idx

                # do some reporting
                if (global_step) % self.eval_config['eval_interval_globalstep'] == 0:
                    train_results, val_results = self.val_step(global_step)   
                                        
                    self.logger.info(f"[{datetime.datetime.now():%H:%M:%S}] \
                        [epoch {epoch}/{self.epochs}] [batch {batch_idx}] [global_step {global_step}] \
                        [tr loss: {train_results['loss']:.4f}] [val loss: {val_results['loss']:.4f}]")
        
        if self.use_tensorboard:
            self.writer.close()
            
        self.save_results(epoch, batch_loss, val_results, train_results)

    @property
    def checkpoint_path(self):
        return os.path.join(self.log_dir, 'model.pt')

    @property
    def trainres_path(self):
        return os.path.join(self.log_dir, 'train_res.pickle')
    
    @property
    def valres_path(self):
        return os.path.join(self.log_dir, 'val_res.pickle')

    def check_if_already_trained(self):
        """Check if model is already trained by checking for results file

        Returns:
            Bool: true if trained
        """
        
        already_trained = os.path.exists(self.checkpoint_path) and \
            os.path.exists(self.trainres_path) and \
            os.path.exists(self.valres_path) 
        
        return already_trained

    def save_results(self, epoch, batch_loss, val_results, train_results):
        
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': batch_loss},
                   self.checkpoint_path)

        with open(self.valres_path, 'wb') as fhandle:
            pickle.dump(val_results, fhandle)
        with open(self.trainres_path, 'wb') as fhandle:
            pickle.dump(train_results, fhandle)

    def load_results_and_checkpoint(self):
        
        with open(self.valres_path, 'rb') as fhandle:
            val_results = pickle.load(fhandle)
        with open(self.trainres_path, 'rb') as fhandle:
            train_results = pickle.load(fhandle)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_loss = checkpoint['loss']
        
        return epoch, batch_loss, val_results, train_results

def trainer_object_fn(
    model: torch.nn.Module, 
    dataloaders: Dict, 
    seed: int, 
    log_dir: str,
    epochs: int = 2,  
    learning_rate: float = 1.e-04,
    loss_type: str = 'orig',
    loss_lambda: float = 10**-6,
    eval: Dict = None,
) -> Tuple[float, Any, Dict]:


    # FIXME the only purpose of this function is to rename the argument "eval" to "eval_config". remove this wrapper and make naming consistent

    eval_config = eval
    
    trainer = Trainer(model, 
                      dataloaders, 
                      seed, 
                      log_dir,
                      eval_config,
                      epochs=epochs, 
                      learning_rate=learning_rate,
                      loss_type=loss_type,
                      loss_lambda=loss_lambda)

    return trainer

