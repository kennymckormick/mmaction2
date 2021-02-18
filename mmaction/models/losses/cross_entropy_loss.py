import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        loss_cls = F.cross_entropy(cls_score, label, **kwargs)
        return loss_cls


@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@LOSSES.register_module()
class SoftLabelLoss(BaseWeightedLoss):

    def __init__(self, loss_weight=1., temperature=1., kl_ratio=0):
        super().__init__(loss_weight=loss_weight)
        assert 0 <= kl_ratio <= 1
        assert temperature > 0
        self.temperature = temperature
        self.kl_ratio = kl_ratio

    def _forward(self, cls_score, label, **kwargs):
        assert cls_score.shape == label.shape
        label = label / self.temperature
        cls_score = cls_score / self.temperature
        cls_score = nn.LogSoftmax(dim=1)(cls_score)
        label = nn.Softmax(dim=1)(label)
        softce_loss = torch.mean(torch.sum(-label * cls_score, 1))
        kldiv_loss = nn.KLDivLoss(reduction='batchmean')(cls_score, label)
        if not hasattr(self, 'kl_ratio'):
            self.kl_ratio = 0.
        loss = softce_loss * (1 - self.kl_ratio) + kldiv_loss * self.kl_ratio
        loss = loss * (self.temperature**2)
        return loss
