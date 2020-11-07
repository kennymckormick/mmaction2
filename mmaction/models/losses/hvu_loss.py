import torch
import torch.nn.functional as F
from torch import nn

from mmaction.core.evaluation import top_k_accuracy
from ..registry import LOSSES
from .base import BaseWeightedLoss


# Since the label category 'action' can also be single-label, we add the
# option action_ce to let class action use CrossEntropy Loss
# Note that we find action by category_name, so the order must be correct
@LOSSES.register_module()
class HVULoss(BaseWeightedLoss):
    """Calculate the BCELoss for HVU.

    Args:
        categories (tuple[str]): Names of tag categories, tags are organized in
            this order. Default: ['action', 'attribute', 'concept', 'event',
            'object', 'scene'].
        category_nums (tuple[int]): Number of tags for each category. Default:
            (739, 117, 291, 69, 1679, 248).
        category_loss_weights (tuple[float]): Loss weights of categories, it
            applies only if `loss_type == 'individual'`. The loss weights will
            be normalized so that the sum equals to 1, so that you can give any
            positive number as loss weight. Default: (1, 1, 1, 1, 1, 1).
        adjust_by (str): Adjust loss by the loss of one tag_category. If
            adjust_by is set, the final loss_weight doesn't necessarily sum to
            1. Default: None.
        loss_type (str): The loss type we calculate, we can either calculate
            the BCELoss for all tags, or calculate the BCELoss for tags in each
            category. Choices are 'individual' or 'all'. Default: 'all'.
        with_mask (bool): Since some tag categories are missing for some video
            clips. If `with_mask == True`, we will not calculate loss for these
            missing categories. Otherwise, these missing categories are treated
            as negative samples.
        reduction (str): Reduction way. Choices are 'mean' or 'sum'. Default:
            'mean'.
        action_ce (bool): Use CrossEntropy Loss for action. Default: False.
            Only works when loss_type is set as 'individual'. If action_ce is
            set as True, all videos should have the category 'action'.
        loss_weight (float): The loss weight. Default: 1.0.
    """

    def __init__(self,
                 categories=('action', 'attribute', 'concept', 'event',
                             'object', 'scene'),
                 category_nums=(739, 117, 291, 69, 1679, 248),
                 category_loss_weights=(1, 1, 1, 1, 1, 1),
                 adjust_by=None,
                 loss_type='all',
                 with_mask=False,
                 reduction='mean',
                 action_ce=False,
                 loss_weight=1.0):

        super().__init__(loss_weight)
        self.categories = categories
        self.category_nums = category_nums
        self.category_loss_weights = category_loss_weights
        self.adjust_by = adjust_by
        assert self.adjust_by in self.categories

        self.action_ce = action_ce
        assert len(self.category_nums) == len(self.category_loss_weights)

        divide_lw_by = None
        for name, lw in zip(categories, category_loss_weights):
            assert lw >= 0
            if self.adjust_by is not None and name == self.adjust_by:
                assert lw > 0
                divide_lw_by = lw
        if self.adjust_by is not None:
            self.category_loss_weights = [
                x / divide_lw_by for x in category_loss_weights
            ]

        self.loss_type = loss_type
        self.with_mask = with_mask
        self.reduction = reduction
        self.category_startidx = [0]
        for i in range(len(self.category_nums) - 1):
            self.category_startidx.append(self.category_startidx[-1] +
                                          self.category_nums[i])
        assert self.loss_type in ['individual', 'all']
        assert self.reduction in ['mean', 'sum']

    def _forward(self, cls_score, label, mask, category_mask):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            mask (torch.Tensor): The mask of tags. 0 indicates that the
                category of this tag is missing in the label of the video.
            category_mask (torch.Tensor): The category mask. For each sample,
                it's a tensor with length `len(self.categories)`, denotes that
                if the category is labeled for this video.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """

        if self.loss_type == 'all':
            loss_cls = F.binary_cross_entropy_with_logits(
                cls_score, label, reduction='none')
            if self.with_mask:
                w_loss_cls = mask * loss_cls
                w_loss_cls = torch.sum(w_loss_cls, dim=1)
                if self.reduction == 'mean':
                    w_loss_cls = w_loss_cls / torch.sum(mask, dim=1)
                w_loss_cls = torch.mean(w_loss_cls)
                return dict(loss_cls=w_loss_cls)
            else:
                if self.reduction == 'sum':
                    loss_cls = torch.sum(loss_cls, dim=-1)
                return dict(loss_cls=torch.mean(loss_cls))
        elif self.loss_type == 'individual':
            losses = {}
            loss_weights = {}
            for name, num, start_idx in zip(self.categories,
                                            self.category_nums,
                                            self.category_startidx):
                category_score = cls_score[:, start_idx:start_idx + num]
                category_label = label[:, start_idx:start_idx + num]

                if name == 'action' and self.action_ce:
                    cls_score = category_score
                    gt_labels = category_label
                    # check that action label is onehot
                    assert torch.all(
                        torch.isclose(
                            torch.sum(gt_labels, 1),
                            torch.max(gt_labels, 1)[0]))
                    gt_labels = torch.argmax(gt_labels, 1)
                    category_loss = nn.CrossEntropyLoss()(cls_score, gt_labels)
                    # since we use ce loss for action, should also report top1
                    # and top5
                    topk_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                              gt_labels.detach().cpu().numpy(),
                                              (1, 5))
                    losses['action_top1_acc'] = torch.tensor(
                        topk_acc[0], device=cls_score.device)
                    losses['action_top5_acc'] = torch.tensor(
                        topk_acc[1], device=cls_score.device)
                    # If action_ce, the loss is already reduced
                else:
                    category_loss = F.binary_cross_entropy_with_logits(
                        category_score, category_label, reduction='none')
                    if self.reduction == 'mean':
                        category_loss = torch.mean(category_loss, dim=1)
                    elif self.reduction == 'sum':
                        category_loss = torch.sum(category_loss, dim=1)

                idx = self.categories.index(name)
                if self.with_mask:
                    category_mask_i = category_mask[:, idx].reshape(-1)
                    # there should be at least one sample which contains tags
                    # in thie category
                    if torch.sum(category_mask_i) < 0.5:
                        losses[f'{name}_LOSS'] = torch.tensor(.0).cuda()
                        loss_weights[f'{name}_LOSS'] = .0
                        continue
                    category_loss = torch.sum(category_loss * category_mask_i)
                    category_loss = category_loss / torch.sum(category_mask_i)
                else:
                    category_loss = torch.mean(category_loss)
                # We name the loss of each category as 'LOSS', since we only
                # want to monitor them, not backward them. We will also provide
                # the loss used for backward in the losses dictionary
                losses[f'{name}_LOSS'] = category_loss
                loss_weights[f'{name}_LOSS'] = self.category_loss_weights[idx]

            if self.adjust_by is not None:
                basic_loss = losses[f'{self.adjust_by}_LOSS']
                loss_weights_2 = {
                    k: (basic_loss / v).item()
                    for k, v in losses.items()
                }
                new_loss_weights = {
                    k: loss_weights[k] * loss_weights_2[k]
                    for k in loss_weights
                }
                loss_cls = sum(
                    [losses[k] * new_loss_weights[k] for k in losses])
                losses['loss_cls'] = loss_cls
                # the new_loss_weights is printed
                losses.update({
                    k + '_weight': torch.tensor(v).to(losses[k].device)
                    for k, v in new_loss_weights.items()
                })
            else:
                loss_weight_sum = sum(loss_weights.values())
                loss_weights = {
                    k: v / loss_weight_sum
                    for k, v in loss_weights.items()
                }
                loss_cls = sum([losses[k] * loss_weights[k] for k in losses])
                losses['loss_cls'] = loss_cls
                # We also trace the loss weights
                losses.update({
                    k + '_weight': torch.tensor(v).to(losses[k].device)
                    for k, v in loss_weights.items()
                })
            # Note that the loss weights are just for reference.
            return losses
