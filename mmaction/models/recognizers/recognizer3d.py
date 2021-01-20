import math

import torch

from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, label, img_metas=None):
        """Defines the computation performed at every call when training."""
        labels = label
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        dynamic = False
        if img_metas is not None and 'real_clip_len' in img_metas[0]:
            real_clip_len = [x['real_clip_len'] for x in img_metas]
            assert self.train_t_stride is not None
            real_clip_len = [
                math.ceil(x / self.train_t_stride) for x in real_clip_len
            ]
            dynamic = True

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        if dynamic:
            cls_score = self.cls_head(x, real_clip_len=real_clip_len)
        else:
            cls_score = self.cls_head(x)

        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, img_metas=None):
        """Defines the computation performed at every call when evaluation and
        testing."""

        dynamic = False
        if img_metas is not None and 'real_clip_len' in img_metas[0]:
            real_clip_len = [x['real_clip_len'] for x in img_metas]
            assert self.test_t_stride is not None
            real_clip_len = [
                math.ceil(x / self.test_t_stride) for x in real_clip_len
            ]
            dynamic = True

        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if dynamic:
            new_real_clip_len = []
            for i in range(len(real_clip_len)):
                new_real_clip_len.extend([real_clip_len[i]] * num_segs)
            real_clip_len = new_real_clip_len

        if self.test_batch is None:
            x = self.extract_feat(imgs)
            if hasattr(self, 'neck'):
                x, _ = self.neck(x)
            if dynamic:
                cls_score = self.cls_head(x, real_clip_len=real_clip_len)
            else:
                cls_score = self.cls_head(x)
        else:
            tot = imgs.shape[0]
            assert num_segs == tot
            ptr = 0
            cls_scores = []
            while ptr < tot:
                batch_imgs = imgs[ptr:ptr + self.test_batch]
                x = self.extract_feat(batch_imgs)
                if hasattr(self, 'neck'):
                    x, _ = self.neck(x)
                assert not dynamic, 'For simplicity'
                cls_scores.append(self.cls_head(x))
                ptr += self.test_batch
            cls_score = torch.cat(cls_scores)

        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score.cpu().numpy()

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
        outs = (self.cls_head(x), )
        return outs
