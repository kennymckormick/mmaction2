import copy as cp
import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES


def combine_quadruple(a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])


def flip_quadruple(a):
    return (1 - a[0] - a[2], a[1], a[2], a[3])


# We assume that results['kp'] is not empty (all equals to 0)
@PIPELINES.register_module()
class PoseCompact:

    def __init__(self,
                 padding=1. / 4.,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=False):
        # hw_ratio can be None, float, or tuple(float)
        self.padding = padding
        self.threshold = threshold

        if isinstance(hw_ratio, float):
            hw_ratio = (hw_ratio, hw_ratio)
        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        # None or tuple of float

        # hw_ratio is height / width
        # The minimum threshold of PoseCompact (#pixel after PoseCompact)
        assert self.padding >= 0

    def __call__(self, results):
        img_shape = results['img_shape']
        h, w = img_shape
        kps = results['kp']
        min_x, min_y, max_x, max_y = np.Inf, np.Inf, -np.Inf, -np.Inf
        for kp in kps:
            # Make NaN zero
            kp[np.isnan(kp)] = 0.

            kp_x = kp[:, :, 0]
            kp_y = kp[:, :, 1]
            # There is at least one legal kp
            if np.sum(kp_x != 0) or np.sum(kp_y != 0):
                min_x = min(min(kp_x[kp_x != 0]), min_x)
                min_y = min(min(kp_y[kp_y != 0]), min_y)
                max_x = max(max(kp_x[kp_x != 0]), max_x)
                max_y = max(max(kp_y[kp_y != 0]), max_y)
            else:
                continue

        # The Compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        box_hwidth = (max_x - min_x) / 2 * (1 + self.padding)
        box_hheight = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            box_hheight = max(self.hw_ratio[0] * box_hwidth, box_hheight)
            box_hwidth = max(1 / self.hw_ratio[1] * box_hheight, box_hwidth)

        min_x, max_x = center[0] - box_hwidth, center[0] + box_hwidth
        min_y, max_y = center[1] - box_hheight, center[1] + box_hheight

        # hot update
        if (not hasattr(self, 'allow_imgpad')) or (not self.allow_imgpad):
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        for kp in kps:
            kp_x = kp[:, :, 0]
            kp_y = kp[:, :, 1]
            kp_x[kp_x != 0] -= min_x
            kp_y[kp_y != 0] -= min_y
        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape
        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results


@PIPELINES.register_module()
class MMPad:

    def __init__(self, hw_ratio=None, padding=0.):
        if isinstance(hw_ratio, float):
            hw_ratio = (hw_ratio, hw_ratio)
        self.hw_ratio = hw_ratio
        self.padding = padding

    # New shape is larger than old shape
    def _pad_kps(self, kps, old_shape, new_shape):
        offset_y = int((new_shape[0] - old_shape[0]) / 2)
        offset_x = int((new_shape[1] - old_shape[1]) / 2)
        offset = np.array([offset_x, offset_y], dtype=np.float32)
        return [kp + offset for kp in kps]

    def _pad_imgs(self, imgs, old_shape, new_shape):
        diff_y, diff_x = new_shape[0] - old_shape[0], new_shape[1] - old_shape[
            1]
        return [
            np.pad(
                img, ((diff_y // 2, diff_y - diff_y // 2),
                      (diff_x // 2, diff_x - diff_x // 2), (0, 0)),
                'constant',
                constant_values=127) for img in imgs
        ]

    def __call__(self, results):
        h, w = results['img_shape']
        h, w = h * (1 + self.padding), w * (1 + self.padding)
        if self.hw_ratio is not None:
            h = max(self.hw_ratio[0] * w, h)
            w = max(1 / self.hw_ratio[1] * h, w)
        h, w = int(h + 0.5), int(w + 0.5)
        if 'kp' in results:
            results['kp'] = self._pad_kps(results['kp'], results['img_shape'],
                                          (h, w))
        # img_shape should be: if not identical to results['img_shape'],
        # at least propotional, just a patch here
        if 'imgs' in results:
            real_img_shape = results['imgs'][0].shape[:2]
            real_h, real_w = real_img_shape
            real_h_ratio = results['img_shape'][0] / real_h
            real_w_ratio = results['img_shape'][1] / real_w
            # almost identical
            # assert np.abs(real_h_ratio - real_w_ratio) < 2e-2

            if real_h == results['img_shape'][0]:
                results['imgs'] = self._pad_imgs(results['imgs'],
                                                 results['img_shape'], (h, w))
            else:
                results['imgs'] = self._pad_imgs(results['imgs'],
                                                 (real_h, real_w),
                                                 (int(h / real_h_ratio + 0.5),
                                                  int(w / real_w_ratio + 0.5)))
        results['img_shape'] = (h, w)
        return results


@PIPELINES.register_module()
class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs" .

    Args:
        size (int): The output size of the images.
    """

    def __init__(self, size):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size

    def _crop_kps(self, kps, crop_bbox):
        return [kp - crop_bbox[:2] for kp in kps]

    def _crop_imgs(self, imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (x_offset / img_w, y_offset / img_h,
                              new_w / img_w, new_h / img_h)
        crop_quadruple = combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple

        if 'kp' in results:
            results['kp'] = self._crop_kps(results['kp'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "imgs", "img_shape", "crop_bbox",
    added or modified keys are "imgs", "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback to maximized CenterCrop
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (left / img_w, top / img_h, new_w / img_w,
                              new_h / img_h)
        crop_quadruple = combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'kp' in results:
            results['kp'] = self._crop_kps(results['kp'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range})')
        return repr_str


@PIPELINES.register_module()
class Resize:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale, keep_ratio=False, interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
                # Then we automatically set keep_ratio as True
                keep_ratio = True
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def nchannel_resize(self, img, new_w, new_h):
        num_channels = img.shape[2]
        idx, imgs = 0, []
        while idx < num_channels:
            if idx + 3 <= num_channels:
                cur_img = mmcv.imresize(
                    img[:, :, idx:idx + 3], (new_w, new_h),
                    interpolation=self.interpolation)
                idx += 3
            else:
                cur_img = mmcv.imresize(
                    img[:, :, idx], (new_w, new_h),
                    interpolation=self.interpolation)
                cur_img = cur_img[..., np.newaxis]
                idx += 1
            imgs.append(cur_img)

        img = np.concatenate(imgs, axis=2)
        return img

    def _resize_kps(self, kps, scale_factor):
        return [kp * self.scale_factor for kp in kps]

    def _resize_imgs(self, imgs, new_w, new_h, modality):
        # If MM, modality can be a list
        if modality == 'RGB' or 'RGB' in modality:
            return [
                mmcv.imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in imgs
            ]
        elif modality in ['RGBFlow', 'PoTion', 'Heatmap']:
            return [self.nchannel_resize(img, new_w, new_h) for img in imgs]

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor
        modality = results['modality']

        if 'kp' in results:
            results['kp'] = self._resize_kps(results['kp'], self.scale_factor)

        if 'imgs' in results:
            results['imgs'] = self._resize_imgs(results['imgs'], new_w, new_h,
                                                modality)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class RandomRescale:
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation='bilinear'):
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        assert mmcv.is_tuple_of(scale_range, int)
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation)
        results = resize(results)

        results['short_edge'] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "flip_direction".

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """

    def __init__(self, flip_ratio=0.5, left=[], right=[]):
        self.flip_ratio = flip_ratio
        self.left = left
        self.right = right
        assert len(self.left) == len(self.right)
        self.direction = 'horizontal'

    def _flip_imgs(self, imgs, modality):
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # 1st Frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        elif modality == 'RGBFlow':
            # 4th channel of each frame is flow-x
            for i in range(lt):
                imgs[i][..., 3] = mmcv.iminvert(imgs[i][..., 3])
        elif modality in ['PoTion', 'Heatmap']:
            if modality == 'PoTion':
                assert lt == 1
            new_order = list(range(imgs[0].shape[-1]))
            for left, right in zip(self.left, self.right):
                new_order[left] = right
                new_order[right] = left
            imgs = [img[..., new_order] for img in imgs]
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):

        for kp, kpscore in zip(kps, kpscores):
            kp[:, :, 0] = img_width - kp[:, :, 0]

            new_order = list(range(kp.shape[1]))
            for left, right in zip(self.left, self.right):
                new_order[left] = right
                new_order[right] = left
            kp[:] = kp[:, new_order]
            kpscore[:] = kpscore[:, new_order]

        return kps, kpscores

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False

        results['flip'] = flip
        results['flip_direction'] = self.direction

        img_width = results['img_shape'][1]
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        results['crop_quadruple'] = flip_quadruple(crop_quadruple)
        modality = results['modality']

        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'], modality)
            if 'kp' in results and 'kpscore' in results:
                kp, kpscore = self._flip_kps(results['kp'], results['kpscore'],
                                             img_width)
                results['kp'], results['kpscore'] = kp, kpscore

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(flip_ratio={self.flip_ratio})'
        return repr_str


@PIPELINES.register_module()
class HeatmapFlipTest:

    def __init__(self, left=[], right=[]):
        self.left = left
        self.right = right

    def __call__(self, results):
        results_ = cp.deepcopy(results)
        flip = Flip(flip_ratio=1, left=self.left, right=self.right)
        results_ = flip(results)
        results['imgs'] = np.concatenate([results['imgs'], results_['imgs']])
        return results


@PIPELINES.register_module()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality in ['RGB', 'RGBFlow', 'Heatmap'] or 'RGB' in modality:
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        elif modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        else:
            raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(RandomCrop):
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", and "img_shape".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (left / img_w, top / img_h, new_w / img_w,
                              new_h / img_h)
        crop_quadruple = combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple

        if 'kp' in results:
            results['kp'] = self._crop_kps(results['kp'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class ThreeCrop(object):
    """Crop images into three crops. Works for image only now !!!!

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str
