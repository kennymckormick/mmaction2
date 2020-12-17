import math

from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, img_metas=None):
        """Defines the computation performed at every call when training."""
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

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)

        if dynamic:
            cls_score = self.cls_head(x, real_clip_len=real_clip_len)
        else:
            cls_score = self.cls_head(x)

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
