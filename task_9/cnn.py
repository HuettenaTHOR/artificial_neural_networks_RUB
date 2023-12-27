# basic imports
import pickle
import numpy as np
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
# local imports
from model_wrapper import ModelWrapper
                    

def build_model(filters: int = 8) -> nn.Sequential:
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
    layers = \
        [
            ("conv_1", nn.Conv2d(1, filters, kernel_size=3, padding="same")),
            ("max_pool_1", nn.MaxPool2d(2)),
            ("relu_1", nn.ReLU()),
            ("conv_2", nn.Conv2d(filters, filters * 2, kernel_size=3, padding="same")),
            ("max_pool_2", nn.MaxPool2d(2)),
            ("relu_2", nn.ReLU()),
            ("flatten", nn.Flatten()),
            ("dense_1", nn.Linear(7 * 7 * filters * 2, 10)),
        ]
    
    return nn.Sequential(OrderedDict(layers))


if __name__ == '__main__':
    # params
    batch_size = 256
    number_of_epochs = 2
    filters = [4, 8, 12, 16]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # define data transforms
    transforms = Compose([ToTensor()])
    # load data (a)

    # prepare data (a)
    training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms)

    training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # build model and check parameters (a)
    results = {}
    for filter in filters:
        seq_model = build_model(filter)
        optimizer = torch.optim.Adam(seq_model.parameters(), lr=0.001)
        loss = nn.CrossEntropyLoss()
        model = ModelWrapper(seq_model, optimizer, loss, device=device)
        train_pre = model.evaluate(training_data_loader)
        test_pre = model.evaluate(test_data_loader)
        metrics = model.fit(training_data_loader, epochs=number_of_epochs, data_eval=test_data_loader, evaluate=True, verbose=True)
        metrics['training']['loss'] = [train_pre['loss']] + metrics['training']['loss']
        metrics['training']['acc'] = [train_pre['acc']] + metrics['training']['acc']
        metrics['test']['loss'] = [train_pre['loss']] + metrics['test']['loss']
        metrics['test']['acc'] = [train_pre['acc']] + metrics['test']['acc']
        results[filter] = metrics
    # evaluate the model before training (a)

    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    for i, filter in enumerate(filters):
        ax[0, i].plot(results[filter]['training']['loss'], label='training')
        ax[0, i].plot(results[filter]['test']['loss'], label='test')
        ax[0, i].set_title(f'filters: {filter}')
        ax[0, i].legend()

        ax[1, i].plot(results[filter]['training']['acc'], label='training')
        ax[1, i].plot(results[filter]['test']['acc'], label='test')
        ax[1, i].set_title(f'filters: {filter}')
        ax[1, i].legend()

    plt.show()

    
    
    # train models with different numbers of hidden units (c), (d), (e)
    
    