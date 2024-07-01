import torch
import torch.nn as nn

# MLP definition
class MLP(nn.Module):
   # Multi layer perceptron torch model
    def __init__(self, num_variables, num_predictors, num_hidden_units=30, 
                 num_hidden_layers=1, act_fn=nn.ReLU()):
        """Initialize model.

        Args:
            num_variables: number of variables used for prediction
            num_predictors: number of variables to be predicted
            num_hidden_units: Hidden units per layers
            num_hidden_layers: Number of hidden layers
            act_fn: Activation function to use after the hidden layers. Defaults to nn.ReLU
        """
        ####################
        
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = [nn.Linear(num_variables, num_hidden_units), act_fn]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_hidden_units, num_hidden_units))
            layers.append(act_fn)
    
        layers.append(nn.Linear(num_hidden_units, num_predictors))
        
        self.linear_relu_stack = nn.Sequential(*layers)
        ####################

    
    def forward(self, x):
        """Compute model predictions.

        Args:
            x: Tensor of data

        Returns:
            tensor of model prediction
        """
        ####################
        x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred
        ####################