import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def SoftCrossEntropy(input, target, reduction='sum'):
    log_likelihood = -F.log_softmax(input, dim=1)
    batch = input.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class SoftlabelCrossEntropy(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'sum') -> None:
        super(SoftlabelCrossEntropy, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return SoftCrossEntropy(input, target, reduction=self.reduction)
