"""
File for all losses used in the Forecast

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum


# Loss function for non negative values
class RMSLELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, predictions, targets):
        return torch.sqrt(self.mse(torch.log(predictions + 1), torch.log(targets + 1)))


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('q', torch.tensor(quantiles).view(1, 1, 1, -1))  # shape [1,1,1,n_quantiles])

    def forward(self, predictions, targets):
        # predictions, targets: [batch, seq, n_targets, n_quantiles]
        diff = predictions - targets.unsqueeze(-1)  # targets: [batch, seq, n_targets, 1]
        ql = (1-self.q)*F.relu(diff) + self.q*F.relu(-diff)
        # Mean over batch and sequence, keep [n_targets, n_quantiles]
        losses = ql.mean(dim=(0, 1))  # shape: [n_targets, n_quantiles]
        return losses.mean() # return single loss item
    
    
class LossTypes(Enum):
    """Defines Loss functions able to use for Forecasting
    """
    QUANTILE = QuantileLoss
    RMSLE = RMSLELoss
    MSE = nn.MSELoss
    L1 = nn.L1Loss