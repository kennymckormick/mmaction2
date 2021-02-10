import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp

from ...utils import get_root_logger
from ..registry import BACKBONES


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class AttentionHead(nn.Module):

    def __init__(self,
                 attention_plane=64,
                 attention_channel=1,
                 attention_scaling=False,
                 attention_type='softmax',
                 attention_lowlr=False,
                 feature_dim=2048,
                 head_mode='conv',
                 debug=None):
        super(AttentionHead, self).__init__()
        self.attention_plane = attention_plane
        self.attention_channel = attention_channel
        assert attention_type in ['sigmoid', 'softmax']
        if attention_scaling:
            assert attention_type == 'softmax'

        self.attention_type = attention_type
        self.attention_scaling = attention_scaling
        self.attention_lowlr = attention_lowlr
        self.head_mode = head_mode
        self.feature_dim = feature_dim
        self.debug = debug

        self.layer_name = ('lowlr_feat2att'
                           if self.attention_lowlr else 'feat2att')
        assert feature_dim % attention_channel == 0

        feat2att = []
        feat2att.append(conv3x3(self.attention_plane, self.attention_channel))
        if 'bn' in self.head_mode:
            feat2att.append(nn.BatchNorm2d(self.attention_channel))
        if 'relu' in self.head_mode:
            feat2att.append(nn.ReLU())
        self.add_module(self.layer_name, nn.Sequential(*feat2att))

    def forward(self, att_feature, feature):
        debug_info = {}
        layer = getattr(self, self.layer_name)
        att = layer(att_feature)
        if self.debug:
            debug_info['att_before'] = att

        if self.attention_type == 'softmax':
            att_shape = att.shape[-2:]
            att = att.reshape(att.shape[:-2] + (-1, ))
            att = nn.Softmax(dim=-1)(att)
            att = att.reshape(att.shape[:2] + att_shape)
            if self.attention_scaling:
                coeff = att_shape[-2] * att_shape[-1]
                att = coeff * att
        else:
            att = nn.Sigmoid()(att)

        if self.debug:
            debug_info['att_after'] = att

        repeat_times = self.feature_dim // self.attention_channel
        att = att.repeat(1, repeat_times, 1, 1)
        return att * feature, debug_info


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers
        stride (int): Spatial stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=None,
                   with_cp=False):
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config for norm layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. Default: None.
        act_cfg (dict | None): Config for activate layers. Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        nn.Module: A residual layer for the given config.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = ConvModule(
            inplanes,
            planes * block.expansion,
            kernel_size=1,
            stride=stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module()
class ResNetAtt(nn.Module):
    """ResNetAtt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        dilations (Sequence[int]): Dilation of each stage.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        partial_bn (bool): Whether to use partial bn. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(
            self,
            depth,
            pretrained=None,
            torchvision_pretrain=True,
            in_channels=3,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            style='pytorch',
            frozen_stages=-1,
            # BEGIN OF ATTENTION ARGS
            attention_plane=64,
            attention_channel=1,
            attention_scaling=True,
            attention_type='softmax',
            attention_lowlr=False,
            attention_featdetach=False,
            head_mode='conv',
            debug=False,
            # END OF ATTENTION ARGS
            conv_cfg=dict(type='Conv'),
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            norm_eval=False,
            partial_bn=False,
            with_cp=False):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.frozen_stages = frozen_stages

        self.attention_plane = attention_plane
        self.attention_channel = attention_channel
        self.attention_scaling = attention_scaling
        self.attention_type = attention_type
        self.attention_lowlr = attention_lowlr
        self.attention_featdetach = attention_featdetach
        self.head_mode = head_mode
        self.debug = debug

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp)

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

            if i == 3:
                res_layer = make_res_layer(
                    self.block,
                    self.inplanes,
                    self.attention_plane,
                    num_blocks,
                    stride=1,
                    dilation=dilation,
                    style=self.style,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp)
                layer_name = ('lowlr_att_layer4'
                              if self.attention_lowlr else 'att_layer4')
                self.att_layer_name = layer_name
                self.add_module(self.att_layer_name, res_layer)

            self.inplanes = planes * self.block.expansion

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)
        self.att_head = AttentionHead(
            attention_plane=self.attention_plane * self.block.expansion,
            attention_channel=self.attention_channel,
            attention_scaling=self.attention_scaling,
            attention_type=self.attention_type,
            attention_lowlr=self.attention_lowlr,
            feature_dim=self.feat_dim // 2,
            head_mode=self.head_mode,
            debug=self.debug)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def _load_conv_params(conv, state_dict_tv, module_name_tv,
                          loaded_param_names):
        """Load the conv parameters of resnet from torchvision.

        Args:
            conv (nn.Module): The destination conv module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding conv module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        weight_tv_name = module_name_tv + '.weight'
        conv.weight.data.copy_(state_dict_tv[weight_tv_name])
        loaded_param_names.append(weight_tv_name)

        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            conv.bias.data.copy_(state_dict_tv[bias_tv_name])
            loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn, state_dict_tv, module_name_tv, loaded_param_names):
        """Load the bn parameters of resnet from torchvision.

        Args:
            bn (nn.Module): The destination bn module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding bn module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            param.data.copy_(param_tv)
            loaded_param_names.append(param_tv_name)

        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            # some buffers like num_batches_tracked may not exist
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)

    def _load_torchvision_checkpoint(self, logger=None):
        """Initiate the parameters from torchvision pretrained checkpoint."""
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        loaded_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                if 'downsample' in name:
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                self._load_conv_params(module.conv, state_dict_torchvision,
                                       original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision,
                                     original_bn_name, loaded_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(
                f'These parameters in pretrained checkpoint are not loaded'
                f': {remaining_names}')

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                # torchvision's
                self._load_torchvision_checkpoint(logger)
            else:
                # ours
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        att_res_layer = getattr(self, self.att_layer_name)
        att_feat = x.detach() if self.attention_featdetach else x
        att = att_res_layer(att_feat)
        output, info = self.att_head(att, x)
        output = self.layer4(output)
        if self.debug:
            return output, info
        else:
            return output

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in self.conv1.modules():
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _partial_bn(self):
        logger = get_root_logger()
        logger.info('Freezing BatchNorm2D except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if mode and self.partial_bn:
            self._partial_bn()
