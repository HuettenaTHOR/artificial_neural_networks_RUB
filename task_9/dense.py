# basic imports
import pickle
import numpy as np
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
# local imports
from model_wrapper import ModelWrapper
import matplotlib.pyplot as plt


def build_model(hidden_layer: int = 5) -> nn.Sequential:
    '''
    This function builds a simple network model.
    
    Parameters
    ----------
    hidden_layer :                      The number of units in each hidden layer.
    
    Returns
    ----------
    model :                             The built model.
    '''
    # prepare layers
    layers = [  # ... (c)
        ("hidden_layer_1", nn.Linear(784, hidden_layer)),
        ("relu_1", nn.ReLU()),
        ("hidden_layer_2", nn.Linear(hidden_layer, hidden_layer)),
        ("relu_2", nn.ReLU()),
        ("output", nn.Linear(hidden_layer, 10)),
    ]

    return nn.Sequential(OrderedDict(layers))


if __name__ == '__main__':
    # params
    batch_size = 64
    number_of_epochs = 15
    units = [5, 10, 15, 20]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # define data transforms
    transforms = Compose([ToTensor(), torch.flatten])
    # load data (a)

    training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms)

    training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # prepare data (b)

    # build model and check parameters (c), (d)
    seq_model = build_model()
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    model = ModelWrapper(seq_model, optimizer, loss, device=device)

    train_pre = model.evaluate(training_data_loader)
    test_pre = model.evaluate(test_data_loader)

    metrics = model.fit(training_data_loader, epochs=number_of_epochs, data_eval=test_data_loader, evaluate=True, verbose=True)
    # evaluate the model before training (e)
    metrics['training']['loss'] = [train_pre['loss']] + metrics['training']['loss']
    metrics['training']['acc'] = [train_pre['acc']] + metrics['training']['acc']
    metrics['test']['loss'] = [train_pre['loss']] + metrics['test']['loss']
    metrics['test']['acc'] = [train_pre['acc']] + metrics['test']['acc']

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(metrics['training']['loss'], label='train')
    ax[0].plot(metrics['test']['loss'], label='test')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[1].plot(metrics['training']['acc'], label='train')
    ax[1].plot(metrics['test']['acc'], label='test')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    plt.show()

    # train models with different numbers of hidden units (f), (g)
