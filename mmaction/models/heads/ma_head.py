from abc import ABCMeta

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from torch.autograd import Function

from ...core import top_k_accuracy
from ..builder import build_loss
from ..registry import HEADS


class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None


class MLP_Block(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class MLP(nn.Module):

    def __init__(self, layers, scale=1.):
        super().__init__()
        assert len(layers) >= 2
        self.layers = layers
        self.scale = scale
        mlp = []
        for i in range(len(layers) - 2):
            mlp.append(MLP_Block(layers[i], layers[i + 1]))
        mlp.append(nn.Linear(layers[-2], layers[-1]))
        self.mlp = nn.Sequential(*mlp)

    def get_scale(self):
        return self.scale

    def forward(self, x):
        scale = self.get_scale()
        x = ScaleGrad.apply(x, scale)
        return self.mlp(x)

    def init_weights(self, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=std)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


@HEADS.register_module()
class MAHead(nn.Module, metaclass=ABCMeta):
    """Class head for MultiAttribute.

    Args:
        in_channels (int): Number of channels in input feature.

        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 attr_names,
                 in_channels,
                 head_cfg=dict(action=dict(layers=[400], scale=1.)),
                 loss_cfg=dict(action=dict(type='CrossEntropyLoss')),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.attr_names = attr_names
        self.head_cfg = head_cfg
        self.loss_cfg = loss_cfg

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        assert set(head_cfg.keys()) == set(loss_cfg.keys()) == set(attr_names)

        heads = {}
        losses = {}

        for k in self.attr_names:
            head_cfg = self.head_cfg[k]
            loss_cfg = self.loss_cfg[k]
            layers = [self.in_channels] + head_cfg['layers']
            scale = head_cfg['scale']
            heads[k] = MLP(layers=layers, scale=scale)
            losses[k] = build_loss(loss_cfg)
        self.heads = nn.ModuleDict(heads)
        self.losses = nn.ModuleDict(losses)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for head in self.heads.values():
            head.init_weights(self.init_std)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        ret = {}
        for k in self.attr_names:
            head = self.heads[k]
            ret[k] = head(x)
        return ret

    def loss(self, cls_score, gt_label, mask):
        losses = dict()
        losses['loss_cls'] = torch.tensor(.0).cuda()
        for k in self.attr_names:
            score = cls_score[k]
            label = gt_label[k]
            the_mask = mask[k]
            # some asserts about shape
            assert len(label.shape) in [1, 2] and len(the_mask.shape) == 1
            score = score[the_mask == 1]
            label = label[the_mask == 1]
            if score.shape[0] == 0:
                loss = torch.tensor(.0).cuda()
            else:
                loss = self.losses[k](score, label)
            losses[k + '_loss'] = loss
            losses['loss_cls'] += loss

            if len(label.shape) == 1:
                top_k_acc = top_k_accuracy(score.detach().cpu().numpy(),
                                           label.detach().cpu().numpy(),
                                           (1, 5))
                losses[k + '_top1_acc'] = torch.tensor(
                    top_k_acc[0], device=score.device)
                losses[k + '_top5_acc'] = torch.tensor(
                    top_k_acc[1], device=score.device)

        return losses
