from collections import defaultdict

import torch
import torch.nn as nn

from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class MARecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def __init__(self,
                 backbone,
                 cls_head,
                 attr_names=['action'],
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        assert neck is None
        super(MARecognizer3D, self).__init__(
            backbone=backbone,
            cls_head=cls_head,
            neck=neck,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.attr_names = attr_names
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward_train(self, imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        # only accept 1 seg
        assert imgs.shape[1] == 1
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[:2])

        cls_score = self.cls_head(x)
        # Note: We only keep those in self.attr_names
        cls_score = {
            x: y
            for x, y in cls_score.items() if x in self.attr_names
        }
        gt_label = {
            x: y.squeeze()
            for x, y in kwargs.items() if x in self.attr_names
        }
        mask = {
            x[:-5]: y.squeeze()
            for x, y in kwargs.items()
            if x[:-5] in self.attr_names and x[-5:] == '_mask'
        }
        assert set(cls_score.keys()) == set(self.attr_names)
        assert set(gt_label.keys()) == set(self.attr_names)
        assert set(mask.keys()) == set(self.attr_names)

        loss_cls = self.cls_head.loss(cls_score, gt_label, mask)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            cls_scores = defaultdict(list)
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                x = self.avg_pool(x)
                x = x.reshape(x.shape[:2])
                cls_score = self.cls_head(x)

                for k, v in cls_score.items():
                    cls_scores[k].append(v)

                view_ptr += self.max_testing_views
            cls_score = {k: torch.cat(v) for k, v in cls_scores.items()}

        else:
            x = self.extract_feat(imgs)
            x = self.avg_pool(x)
            x = x.reshape(x.shape[:2])

            cls_score = self.cls_head(x)

        cls_score = {
            k: self.average_clip(v, num_segs)
            for k, v in cls_score.items()
        }
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        cls_score = self._do_test(imgs)
        # We cvt each score to a 1d array
        cls_score = {
            x: y.reshape(-1).cpu().numpy()
            for x, y in cls_score.items() if x in self.attr_names
        }
        return cls_score

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[:2])

        outs = (self.cls_head(x), )
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
