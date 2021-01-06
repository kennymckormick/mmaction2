import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


class PPool(nn.Module):
    """Pyramid Pooling for 3D feature, input: N x C x T x H x W, output: N x
    C'.

    Args:
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        sizes (list[int]): Pooled feature sizes.
    """

    def __init__(self, spatial_type='avg', sizes=[1]):
        super().__init__()
        if spatial_type == 'avg':
            self.pools = [nn.AdaptiveAvgPool3d(s) for s in sizes]
        elif spatial_type == 'max':
            self.pools = [nn.AdaptiveMaxPool3d(s) for s in sizes]
        self.pools = nn.ModuleList(self.pools)

    def forward(self, x):
        features = [pool(x) for pool in self.pools]
        features = [feat.view(feat.shape[0], -1) for feat in features]
        feature = torch.cat(features, dim=1)
        return feature


@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 sizes=None,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.sizes = sizes

        self.pool = None
        if self.sizes is not None:
            self.pool = PPool(self.spatial_type, self.sizes)
        elif self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d(1)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if 'real_clip_len' in kwargs:
            ret = []
            real_clip_len = kwargs['real_clip_len']
            assert x.shape[0] == len(real_clip_len)
            for i in range(x.shape[0]):
                clip_len = real_clip_len[i]
                ret.append(self.pool(x[i:i + 1, :, :clip_len]))
            x = torch.cat(ret)
        else:
            x = self.pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
