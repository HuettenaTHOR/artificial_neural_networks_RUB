# basic imports
from __future__ import annotations

import numpy as np
# torch
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


class ModelWrapper():
    
    def __init__(self, model: nn.Sequential | nn.Module, optimizer: torch.optim.Optimizer, loss: nn.modules.loss._Loss, device: str = 'cpu'):
        '''
        This class wraps around a torch model and provides functions for training and evaluation. 
        
        Parameters
        ----------
        model :                             The network model.
        optimizer :                         The optimizer that will be used for training.
        loss :                              The loss function that will be used for training.
        device :                            The name of the device that the model will stored on (\'cpu\' by default).
        
        Returns
        ----------
        None
        '''
        self.model : nn.Sequential = model
        self.optimizer = optimizer
        self.criterion = loss
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def predict_on_batch(self, batch: np.ndarray | torch.Tensor) -> np.ndarray:
        '''
        This function computes network predictions for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        
        Returns
        ----------
        predictions :                       A batch of network predictions.
        '''
        with torch.inference_mode():
            if type(batch) is np.ndarray:
                batch = torch.tensor(batch, device=self.device)
            else:
                batch = batch.to(device=self.device)
            return self.model(batch).detach().cpu().numpy()
        
    def fit(self, data: DataLoader, epochs: int = 1, data_eval: None | DataLoader = None, evaluate: bool = False, verbose: bool = False) -> dict:
        '''
        This function fits the model on a given data set.
        
        Parameters
        ----------
        data :                              The data set that the model will be fit on.
        epochs :                            The number of epochs that the model will be fit for.
        data_eval :                         An additional test data set that the model will be evaluated on.
        evaluate :                          A flag indicating whether the model should be evaluated after each epoch.
        verbose :                           A flag indicating whether the training progress should be printed to console.
        
        Returns
        ----------
        metrics :                           A dictionary containing the average loss and average accuracy for each training epoch.
        '''
        assert epochs > 0
        metrics = {'training': {'loss': [], 'acc': []}, 'test': {'loss': [], 'acc': []}}
        # ! Implement here !
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(tqdm(data)):
                inputs, labels = batch
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 100 == 0 and verbose:
                    print(f'epoch: {epoch + 1} batch: {i + 1} loss: {running_loss / 100}')
                    running_loss = 0.0
            if evaluate:
                metrics_epoch = self.evaluate(data)
                metrics['training']['loss'].append(metrics_epoch['loss'])
                metrics['training']['acc'].append(metrics_epoch['acc'])
                if data_eval is not None:
                    metrics_epoch = self.evaluate(data_eval)
                    metrics['test']['loss'].append(metrics_epoch['loss'])
                    metrics['test']['acc'].append(metrics_epoch['acc'])
        return metrics
                    
    def evaluate(self, data: DataLoader) -> dict:
        '''
        This function evaluates the model on a given data set.
        
        Parameters
        ----------
        data :                              The data set that the model will be evaluated on.
        
        Returns
        ----------
        metrics :                           A dictionary containing the average loss and average accuracy.
        '''
        metrics = {'loss': [], 'acc': []}
        # ! Implement here !
        for i, batch in enumerate(data):
            with torch.inference_mode():
                inputs, labels = batch
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                predictions = self.model(inputs)
                loss = self.criterion(predictions, labels)
                metrics['loss'].append(loss.item())
                metrics['acc'].append((predictions.argmax(dim=1) == labels).sum().item() / len(labels))

        return {'loss': np.mean(metrics['loss']), 'acc': np.mean(metrics['acc'])}
    
    def get_weights(self) -> list:
        '''
        This function returns the weights of the network.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        weights :                           A list of layer weights.
        '''
        # retrieve params from state_dict and save them as a list of numpy arrays
        weights = list(self.model.state_dict().values())
        for i in range(len(weights)):
            weights[i] = weights[i].cpu().numpy()
            
        return weights
    
    def set_weights(self, weights: list):
        '''
        This function sets the weights of the network.
        
        Parameters
        ----------
        weights :                           A list of layer weights.
        
        Returns
        ----------
        None
        '''
        # prepare a new state_dict
        new_state_dict = self.model.state_dict()
        for i, param in enumerate(new_state_dict):
            new_state_dict[param] = torch.tensor(weights[i], device=self.device)
        # load the state_dict
        self.model.load_state_dict(new_state_dict)
    
    def summary(self) -> dict:
        '''
        This function returns the weight dimensions of each layer as well as the total number of parameters.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        summary :                           A dictionary containing the weight dimensions of each layer and total number of parameters.
        '''
        weight_dims, number_of_parameters = [], 0
        for param in self.model.parameters():
            weight_dims.append(tuple(param.shape))
            number_of_parameters += np.product(param.shape)
        
        return {'weight_dimensions': weight_dims, 'parameters': number_of_parameters}
        