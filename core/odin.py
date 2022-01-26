import time
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc

import torch
from torch.autograd import Variable


def evaluate_scores(scores_ID, scores_OOD):
    """calculates classification performance (ROCAUC, FPR@TPR95) based on lists of scores
    
    Returns:
        ROCAUC, fpr95
    """
    labels_in = np.ones(scores_ID.shape)
    labels_out = np.zeros(scores_OOD.shape)
    y = np.concatenate([labels_in, labels_out])
    score = np.concatenate([scores_ID, scores_OOD])
    fpr, tpr, _ = roc_curve(y, score)

    roc_auc = auc(fpr, tpr)
    ii=np.where(tpr>0.95)[0][0]
    
    return roc_auc, fpr[ii]

def predict_scores(net1, device, dataloader, noiseMagnitude1, temper, num_images=1000):
    """Derived from original ODIN code

    """
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # TODO: gradient normalisation depends on dataset, is hardcoded at the moment.
    
    columns = ['temperature', 'epsilon', 'method', 'score']
    df = pd.DataFrame(columns=columns)
    
    t0 = time.time()
    for j, data in enumerate(dataloader):        
        if j == num_images:
            break
        images, _ = data
        
        inputs = Variable(images.to(device), requires_grad = True)
        outputs = net1(inputs)
        

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        
        score = np.max(nnOutputs)
        
        row = {'temperature': 1, 'epsilon': 0, 'method': 'base', 'score': score}
        df = df.append(row, ignore_index=True)
                       
        # Using temperature scaling
        outputs = outputs / temper
	
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).to(device))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        # gradient to {-1, 1}
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        # TODO: same transform as dataset preproc
        # (0.1307,), (0.3081,)
        gradient[0][0] = gradient[0][0]/0.3081
        # gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        # gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        # gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        # TODO why negative sign?
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        
        score = np.max(nnOutputs)
        row = {'temperature': temper, 'epsilon': noiseMagnitude1, 'method': 'odin', 'score': score}
        df = df.append(row, ignore_index=True)
        
        if j % 100 == 99:
            print(f'{j}/{num_images} processed in {time.time()-t0} seconds.')
            t0 = time.time()
            
    return df
 