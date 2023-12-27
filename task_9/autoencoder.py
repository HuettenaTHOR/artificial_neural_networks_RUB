# basic imports
import pickle
import numpy as np
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
# local imports
from model_wrapper import ModelWrapper
                    

def build_model(code_size: int = 32) -> nn.Sequential:
    '''
    This function builds a simple network model.
    
    Parameters
    ----------
    filters :                           The number of filters in the first convolutional layer.
    
    Returns
    ----------
    model :                             The built model.
    '''
    # prepare layers
    layers = [ # ... (a)
              ]
    
    return nn.Sequential(OrderedDict(layers))


if __name__ == '__main__':
    # params
    batch_size = 64
    number_of_epochs = 20
    code_sizes = [32, 64, 128, 256]
    device = 'cuda'
    
    # define data transforms
    transforms = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))])
    # load data
    training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms)
    training_data = training_data.data.clone().detach().float().reshape((60000, 1, 28, 28))/255.
    training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # ... (b)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms)
    test_data = test_data.data.clone().detach().float().reshape((10000, 1, 28, 28))/255.
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # ... (b)
    
    # prepare data (b)
    
    # train networks (c)
    
    