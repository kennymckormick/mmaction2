import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..registry import BACKBONES
from .resnet3d_slowfast import build_pathway


@BACKBONES.register_module()
class ResNet3dPoseSlowFast(nn.Module):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str): The file path to a pretrained model.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 8.
        rgb_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        pose_pathway (dict): Configuration of fast branch, similar to
            `rgb_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(self,
                 pretrained,
                 speed_ratio=4,
                 channel_ratio=4,
                 lateral_last=False,
                 rgb_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     lateral_infl=1,
                     lateral_activate=[0, 0, 1, 1],
                     base_channels=64,
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 pose_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     in_channels=25,
                     base_channels=32,
                     num_stages=3,
                     out_indices=(0, 1, 2),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False)):
        super().__init__()
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.lateral_last = lateral_last

        if rgb_pathway['lateral']:
            rgb_pathway['speed_ratio'] = speed_ratio
            rgb_pathway['channel_ratio'] = channel_ratio

        if pose_pathway['lateral']:
            pose_pathway['speed_ratio'] = speed_ratio
            pose_pathway['channel_ratio'] = channel_ratio

        self.rgb_path = build_pathway(rgb_pathway)
        self.pose_path = build_pathway(pose_pathway)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch seperately.
            self.rgb_path.init_weights()
            self.pose_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs, heatmap_imgs):
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input data.
            heatmap_imgs (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        # We assume base_channel for RGB and Pose are 64 and 32.
        x_rgb = self.rgb_path.conv1(imgs)
        x_rgb = self.rgb_path.maxpool(x_rgb)
        # N x 64 x 8 x 56 x 56

        x_pose = self.pose_path.conv1(heatmap_imgs)
        x_pose = self.pose_path.maxpool(x_pose)

        x_rgb = self.rgb_path.layer1(x_rgb)
        x_rgb = self.rgb_path.layer2(x_rgb)
        x_pose = self.pose_path.layer1(x_pose)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            x_pose_lateral = self.rgb_path.layer2_lateral(x_pose)

        if hasattr(self.pose_path, 'layer1_lateral'):
            x_rgb_lateral = self.pose_path.layer1_lateral(x_rgb)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer1_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer3(x_rgb)
        x_pose = self.pose_path.layer2(x_pose)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_pose_lateral = self.rgb_path.layer3_lateral(x_pose)

        if hasattr(self.pose_path, 'layer2_lateral'):
            x_rgb_lateral = self.pose_path.layer2_lateral(x_rgb)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer2_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer4(x_rgb)
        x_pose = self.pose_path.layer3(x_pose)

        assert self.lateral_last is False
        return (x_rgb, x_pose)
