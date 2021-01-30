import copy as cp
import io
import os.path as osp
import pickle
import warnings

import cv2
import mmcv
import numpy as np
from mmcv.fileio import FileClient

from ..registry import PIPELINES
from .augmentations import Flip


@PIPELINES.register_module()
class SampleFrames(object):
    """Sample frames from the video.

    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class UniformSampleFrames:

    def __init__(self, clip_len, num_clips=1, test_mode=False, seed=255):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames, clip_len):
        assert self.num_clips == 1
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset
        return inds

    def _get_test_clips(self, num_frames, clip_len):
        np.random.seed(self.seed)
        if num_frames < clip_len:
            # Then we use a simple strategy
            if num_frames < self.num_clips:
                start_inds = list(range(self.num_clips))
            else:
                start_inds = [
                    i * num_frames // self.num_clips
                    for i in range(self.num_clips)
                ]
            inds = np.concatenate(
                [np.arange(i, i + clip_len) for i in start_inds])
        elif clip_len <= num_frames < clip_len * 2:
            all_inds = []
            for i in range(self.num_clips):
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
                all_inds.append(inds)
            inds = np.concatenate(all_inds)
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            all_inds = []
            for i in range(self.num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            inds = np.concatenate(all_inds)
        return inds

    def __call__(self, results):
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class WeightedUniformSampleFrames(UniformSampleFrames):
    # Only works for OpenPose-18P
    def __init__(self,
                 face=0.1,
                 torso=0.2,
                 limb=0.3,
                 mppolicy='max',
                 hard_thre=None,
                 remain_score=False,
                 **kwargs):
        super(WeightedUniformSampleFrames, self).__init__(**kwargs)
        self.weights = dict(face=face, torso=torso, limb=limb)
        self.mppolicy = mppolicy
        self.hard_thre = hard_thre
        self.remain_score = remain_score
        assert self.mppolicy in ['max', 'mean'] or \
            isinstance(self.mppolicy, float)
        self.kpsubset = dict(
            torso=[0, 1, 2, 8, 5, 11],
            limb=[3, 4, 6, 7, 9, 10, 12, 13],
            face=[14, 15, 16, 17])

    def _get_segments(self, kpscore, clip_len):
        face = self.kpsubset['face']
        torso = self.kpsubset['torso']
        limb = self.kpsubset['limb']
        kpscore = kpscore.astype(np.float32)
        if self.hard_thre is not None:
            kpscore = kpscore >= self.hard_thre
        score = np.sum(kpscore[..., face], axis=-1) * self.weights['face'] + \
            np.sum(kpscore[..., torso], axis=-1) * self.weights['torso'] + \
            np.sum(kpscore[..., limb], axis=-1) * self.weights['limb']
        if self.mppolicy == 'max':
            score = np.max(score, axis=0)
        elif self.mppolicy == 'mean':
            score = np.mean(score, axis=0)
        elif isinstance(self.mppolicy, float):
            max_score = np.max(score, axis=0)
            if not self.remain_score:
                num_person = np.sum(max_score > 0.01, axis=0)
                num_person -= 1
                score = max_score * (1 + num_person * self.mppolicy)
            else:
                sum_score = np.sum(score, axis=0)
                sum_score = sum_score - max_score
                score = max_score + self.mppolicy * sum_score
        # This is score used for sampling
        end = np.zeros(clip_len + 1, dtype=np.int32)
        end_value = np.zeros(clip_len + 1)
        score_bin = np.sum(score) / clip_len
        ptr, summ = 1, 0
        for i in range(len(score)):
            summ += score[i]
            if summ > ptr * score_bin + 0.01:
                end[ptr] = i
                end_value[ptr] = summ
                ptr += 1
        end[ptr] = i + 1
        end_value[ptr] = summ
        while ptr < clip_len:
            ptr += 1
            end[ptr] = i + 1
            end_value[ptr] = summ
        return end, score

    def _get_clips_given_segments(self, end, score, mode='train'):
        if mode == 'test':
            np.random.seed(self.seed)
        lt = len(end) - 1
        indices = []
        for i in range(lt):
            ind_range = list(range(end[i], end[i + 1]))
            prob = score[end[i]:end[i + 1]]
            prob = prob / np.sum(prob)
            inds = np.random.choice(
                ind_range, size=self.num_clips, replace=True, p=prob)
            indices.append(inds)
        indices = np.stack(indices).T
        indices = indices.reshape(-1)
        return indices

    def __call__(self, results):
        num_frames = results['total_frames']
        kpscore = results['kpscore']
        clip_len = self.clip_len
        end, score = self._get_segments(kpscore, clip_len)

        end_legal = True
        for i in range(clip_len):
            if end[i] == end[i + 1]:
                end_legal = False
                break
        score_legal = True
        for i in range(clip_len):
            if np.sum(score[end[i]:end[i + 1]]) < 0.01:
                score_legal = False
                break
        legal = end_legal and score_legal

        if self.test_mode:
            if num_frames <= 2 * self.clip_len or not legal:
                inds = self._get_test_clips(num_frames, self.clip_len)
            else:
                inds = self._get_clips_given_segments(end, score, mode='test')
        else:
            if num_frames <= 2 * self.clip_len or not legal:
                inds = self._get_train_clips(num_frames, self.clip_len)
            else:
                inds = self._get_clips_given_segments(end, score)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class MMUniformSampleFrames(UniformSampleFrames):
    # Here, clip_len is a dictionary: key: modality_name, value: clip_len
    # We assume it is RGB, Pose & start_index is hardcoded
    # MM is abbrev of multi-modality

    def __call__(self, results):
        num_frames = results['total_frames']
        modalities = []
        for modality, clip_len in self.clip_len.items():
            if self.test_mode:
                inds = self._get_test_clips(num_frames, clip_len)
            else:
                inds = self._get_train_clips(num_frames, clip_len)
            inds = np.mod(inds, num_frames)
            results[f'{modality}_inds'] = inds.astype(np.int)
            modalities.append(modality)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        if not isinstance(results['modality'], list):
            # should override
            results['modality'] = modalities
        return results


@PIPELINES.register_module()
class DecordInit(object):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def _get_videoreader(self, filename):
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=1)
        return container

    def __call__(self, results):
        """Perform the Decord initiation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        results['video_reader'] = self._get_videoreader(results['filename'])
        results['total_frames'] = len(results['video_reader'])
        return results


@PIPELINES.register_module()
class DecordDecode(object):
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".
    """

    def __init__(self, **kwargs):
        pass

    def _decord_load_frames(self, container, inds):
        # very interesting, for some videos, the #channel is 4
        frame_dict = {
            idx: container[idx].asnumpy()[:, :, :3]
            for idx in np.unique(inds)
        }
        imgs = [frame_dict[idx] for idx in inds]
        return imgs

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # Generate frame index mapping in order
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class RawFrameDecode(object):
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def _load_frames(self, frame_dir, filename_tmpl, modality, frame_inds):
        mmcv.use_backend('cv2')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        imgs = list()
        for frame_idx in frame_inds:
            if modality == 'RGB':
                filepath = osp.join(frame_dir, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(frame_dir,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(frame_dir,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError('RawFrameDecode: Modality '
                                          f'{modality} not supported')
        return imgs

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        frame_dir = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset
        imgs = self._load_frames(frame_dir, filename_tmpl, modality,
                                 frame_inds)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class PoTionDecode(object):
    """Load and decode frames with given indices.

    Required keys are "filename", added or modified keys are "imgs",
    "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``PoTionDecode``.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(results['filename'])
        img = np.load(io.BytesIO(bytes))
        results['imgs'] = [img]
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        return results


# Support Pose w. MultiPerson
# Decide to make `per_frame_box` optional
@PIPELINES.register_module()
class PoseDecode(object):
    """Load and decode pose with given indices.

    Required keys are "frame_inds", "kp". "kpscore" and "per_frame_box" are
    optional.
    """

    def __init__(self,
                 random_drop=False,
                 random_seed=1,
                 drop_per_nframe=16,
                 drop_njoints=1,
                 manipulate_joints=[7, 8, 9, 10, 13, 14, 15, 16]):
        self.random_drop = random_drop
        self.random_seed = random_seed
        self.drop_per_nframe = drop_per_nframe
        self.drop_njoints = drop_njoints
        self.manipulate_joints = manipulate_joints

    def _drop_kpscore(self, kpscores):
        for kpscore in kpscores:
            lt = kpscore.shape[0]
            for tidx in range(lt):
                if np.random.random() < 1. / self.drop_per_nframe:
                    jidxs = np.random.choice(
                        self.manipulate_joints,
                        size=self.drop_njoints,
                        replace=False)
                    for jidx in jidxs:
                        kpscore[tidx, jidx] = 0.
        return kpscores

    def _load_kp(self, kp, frame_inds):
        return [x[frame_inds].astype(np.float32) for x in kp]

    # drop kpscore happens outside
    def _load_kpscore(self, kpscore, frame_inds):
        return [x[frame_inds].astype(np.float32) for x in kpscore]

    def _load_pose_box(self, pose_box, frame_inds):
        return [x[frame_inds].astype(np.float32) for x in pose_box]

    def _load_compact_heatmap(self, compact_heatmap, frame_inds):
        return [[x[ind] for ind in frame_inds] for x in compact_heatmap]

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # If no 'frame_inds' in results, set 'frame_inds' as range(num_frames)
        # by default
        if self.random_drop:
            np.random.seed(self.random_seed)
            assert 'kpscore' in results, 'for simplicity'

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'kpscore' in results:
            assert results['num_person'] == len(results['kpscore'])
            if self.random_drop:
                results['kpscore'] = self._drop_kpscore(results['kpscore'])

            results['kpscore'] = self._load_kpscore(results['kpscore'],
                                                    frame_inds)

        if 'kp' in results:
            assert results['num_person'] == len(results['kp'])
            results['kp'] = self._load_kp(results['kp'], frame_inds)

        if 'pose_box' in results:
            assert results['num_person'] == len(results['pose_box'])
            results['pose_box'] = self._load_pose_box(results['pose_box'],
                                                      frame_inds)

        if 'compact_heatmap' in results:
            assert results['num_person'] == len(results['compact_heatmap'])
            results['compact_heatmap'] = self._load_compact_heatmap(
                results['compact_heatmap'], frame_inds)

        return results


@PIPELINES.register_module()
class MMDecode(DecordInit, DecordDecode, RawFrameDecode, PoseDecode):
    # rgb_type in ['video', 'frame']
    def __init__(self, io_backend='disk', rgb_type='frame', **kwargs):
        self.io_backend = io_backend
        assert rgb_type in ['frame', 'video']
        self.rgb_type = rgb_type
        self.kwargs = kwargs
        self.file_client = None

    # def _decode_rgb(self, frame_dir)
    def __call__(self, results):
        for mod in results['modality']:
            if results[f'{mod}_inds'].ndim != 1:
                results[f'{mod}_inds'] = np.squeeze(results[f'{mod}_inds'])
            frame_inds = results[f'{mod}_inds']
            if mod == 'RGB':
                if self.rgb_type == 'video':
                    video_reader = self._get_videoreader(results['frame_dir'])
                    imgs = self._decord_load_frames(video_reader, frame_inds)
                    del video_reader
                else:
                    # + 1 for RawframeDataset
                    imgs = self._load_frames(results['frame_dir'],
                                             results['filename_tmpl'], 'RGB',
                                             frame_inds + 1)
                results['imgs'] = imgs
            elif mod == 'Pose':
                assert 'kp' in results
                if 'kpscore' not in results:
                    kpscore = [
                        np.ones(kp.shape[:-1], dtype=np.float32)
                        for kp in results['kp']
                    ]
                    results['kpscore'] = kpscore
                results['kp'] = self._load_kp(results['kp'], frame_inds)
                results['kpscore'] = self._load_kp(results['kpscore'],
                                                   frame_inds)
            else:
                raise NotImplementedError(f'MMDecode: Modality {mod} not '
                                          'supported')

        return results


@PIPELINES.register_module()
class LoadFile:
    """Load a pickle file given filename & update to results."""

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        assert 'filename' in results
        filename = results.pop('filename')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(filename)
        data = pickle.loads(bytes)
        data.update(results)
        return data


@PIPELINES.register_module()
class LoadKineticsPose(object):
    """Load and decode pose with given indices.

    Required keys are "filename".
    """

    # squeeze (Remove those frames that w/o. keypoints)
    # kp2keep (The list of keypoint ids to keep)
    def __init__(self,
                 io_backend='disk',
                 squeeze=True,
                 kp2keep=None,
                 kpscore_thre=0,
                 **kwargs):
        self.io_backend = io_backend
        self.squeeze = squeeze
        self.kp2keep = kp2keep
        self.kpscore_thre = kpscore_thre
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        assert 'filename' in results
        filename = results.pop('filename')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(filename)
        data = pickle.loads(bytes)
        data.update(results)

        def mapinds(inds):
            uni = np.unique(inds)
            mapp = {x: i for i, x in enumerate(uni)}
            inds = [mapp[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        num_person = data['num_person']
        num_frame = data['num_frame']
        frame_inds = list(data.pop('frame_inds'))
        person_inds = list(data.pop('person_inds'))

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            num_frame = np.max(frame_inds) + 1

        # need write back
        data['num_frame'] = num_frame
        data['total_frames'] = num_frame

        kps = data['kp']
        if self.kp2keep is not None:
            kps = kps[:, self.kp2keep]
        h, w = data['img_shape']
        kps[:, :, 0] *= w
        kps[:, :, 1] *= h

        num_kp = kps.shape[1]
        new_kp = np.zeros([num_person, num_frame, num_kp, 2], dtype=np.float16)
        new_kpscore = np.zeros([num_person, num_frame, num_kp],
                               dtype=np.float16)

        for frame_ind, person_ind, kp in zip(frame_inds, person_inds, kps):
            new_kp[person_ind, frame_ind] = kp[:, :2]
            kpscore = kp[:, 2]
            # Apply kpscore threshold
            kpscore[kpscore < self.kpscore_thre] = 0.
            new_kpscore[person_ind, frame_ind] = kpscore

        data['kp'] = new_kp
        data['kpscore'] = new_kpscore
        return data


@PIPELINES.register_module()
class RawRGBFlowDecode(object):
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend('cv2')

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']
        assert modality == 'RGBFlow'

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        # Only for trimmed video recognition
        assert offset == 0

        img_prefix = 'img'
        x_prefix = 'x'
        y_prefix = 'y'
        if 'filename_prefix' in results:
            filename_prefix = results['filename_prefix']
            img_prefix = filename_prefix['img']
            x_prefix = filename_prefix['x']
            y_prefix = filename_prefix['y']

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            # flow frame idx may be different from RGB index
            flow_frame_idx = (
                frame_idx - results['start_index']
            ) / results['total_frames'] * results['flow_total_frames']
            flow_frame_idx = int(flow_frame_idx) + results['start_index']

            # load RGB frame
            pair = []
            filepath = osp.join(directory,
                                filename_tmpl.format(img_prefix, frame_idx))
            img_bytes = self.file_client.get(filepath)
            # Get frame with channel order RGB directly.
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            pair.append(cur_frame)
            # load Flow frames
            x_filepath = osp.join(
                directory, filename_tmpl.format(x_prefix, flow_frame_idx))
            y_filepath = osp.join(
                directory, filename_tmpl.format(y_prefix, flow_frame_idx))
            x_img_bytes = self.file_client.get(x_filepath)
            x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
            y_img_bytes = self.file_client.get(y_filepath)
            y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')

            x_frame = x_frame[:, :, np.newaxis]
            y_frame = y_frame[:, :, np.newaxis]
            pair.extend([x_frame, y_frame])

            hs = [x.shape[0] for x in pair]
            ws = [x.shape[1] for x in pair]
            min_h = min(hs)
            min_w = min(ws)
            pair = [x[:min_h, :min_w] for x in pair]

            img = np.concatenate(pair, axis=2)
            imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class FrameSelector(RawFrameDecode):
    """Deprecated class for ``RawFrameDecode``."""

    def __init__(self, *args, **kwargs):
        warnings.warn('"FrameSelector" is deprecated, please switch to'
                      '"RawFrameDecode"')
        super().__init__(*args, **kwargs)


@PIPELINES.register_module()
class ConvertCompactHeatmap:

    def __init__(self,
                 shortedge=100,
                 compact=False,
                 padding=1. / 4,
                 hw_ratio=None):
        # The compact operation should be conducted by group
        self.shortedge = shortedge
        self.compact = compact
        self.padding = padding
        if isinstance(hw_ratio, float):
            hw_ratio = (hw_ratio, hw_ratio)
        # We allow imgpad always
        self.hw_ratio = hw_ratio
        # This threshold (eps) is decided when we generate heatmaps
        # If active area is too small, we do not do copmact op
        self.eps = 3e-3
        self.threshold = 10

    def _convert_pose_box(self, compact_heatmap, pose_box):
        new_compact_heatmap = []
        for item in compact_heatmap:
            heatmap, pos = item[0], item[1]
            h, w = heatmap.shape
            # [x, y, w, h]
            quadruple = [
                pos[0] / pos[2], pos[1] / pos[3], w / pos[2], h / pos[3]
            ]
            this_pose_box = [
                pose_box[0] + quadruple[0] * pose_box[2],
                pose_box[1] + quadruple[1] * pose_box[3],
                pose_box[2] * quadruple[2], pose_box[3] * quadruple[3]
            ]
            new_compact_heatmap.append((heatmap, this_pose_box))
            # print(heatmap.shape)
        return new_compact_heatmap

    def __call__(self, results):
        img_shape = results['img_shape']
        old_se = min(img_shape)
        scale_factor = self.shortedge / old_se

        new_h, new_w = int(img_shape[0] * scale_factor), int(img_shape[1] *
                                                             scale_factor)
        new_shape = (new_h, new_w)

        results['img_shape'] = new_shape
        # May have multiple lists here
        results['pose_box'] = [scale_factor * x for x in results['pose_box']]
        results['per_frame_box'] = [
            scale_factor * x for x in results['per_frame_box']
        ]

        new_heatmaps = []
        num_frame = results['pose_box'][0].shape[0]
        num_joints = len(results['compact_heatmap'][0][0])
        for i in range(num_frame):
            new_heatmaps.append(
                np.zeros([new_h, new_w, num_joints], dtype=np.float32))

        for i in range(results['num_person']):
            compact_heatmap = results['compact_heatmap'][i]
            pose_box = results['pose_box'][i]

            for j in range(num_frame):
                heatmap, box = compact_heatmap[j], pose_box[j]
                num_joints = len(heatmap)
                # print(num_joints)
                new_compact_heatmap = self._convert_pose_box(heatmap, box)
                new_heatmap = new_heatmaps[j]
                # print(len(new_compact_heatmap))
                # We use a corase mapping here
                for k in range(num_joints):
                    # [x, y, w, h]
                    # print(new_compact_heatmap)
                    heatmap, box = new_compact_heatmap[k]
                    if not (heatmap.shape[0] and heatmap.shape[1]):
                        continue
                    heatmap = heatmap.astype(np.float32)
                    # print(len(heatmap))
                    # print(heatmap.shape, box.shape)
                    box = [int(x + 0.5) for x in box]
                    if not (box[2] and box[3]):
                        continue
                    heatmap = cv2.resize(heatmap, (box[2], box[3]))
                    x_offset, y_offset = max(0, -box[0]), max(0, -box[1])
                    st_x, st_y = box[0] + x_offset, box[1] + y_offset
                    ed_x, ed_y = min(box[0] + box[2],
                                     new_w), min(box[1] + box[3], new_h)
                    box_width = ed_x - st_x
                    box_height = ed_y - st_y
                    if box_width <= 0 or box_height <= 0:
                        continue
                    patch = heatmap[y_offset:y_offset + box_height,
                                    x_offset:x_offset + box_width]
                    # print(np.max(patch), st_y, ed_y, st_x, ed_x)
                    original_patch = new_heatmap[st_y:ed_y, st_x:ed_x, k]

                    original_patch[:] = np.maximum(original_patch, patch)
        heatmaps = new_heatmaps
        results['imgs'] = heatmaps
        results['modality'] = 'Heatmap'

        # print(self.compact)
        # Now that we get heatmaps, We can make them compact if needed,
        # note that it will not be conducted by group (right choice, not now)

        if self.compact:
            min_x, min_y, max_x, max_y = new_w - 1, new_h - 1, 0, 0
            for frame in heatmaps:
                # find th e
                for x in range(min_x):
                    if np.any(frame[:, x] > self.eps):
                        min_x = x
                        break
                for x in range(new_w - 1, max_x, -1):
                    if np.any(frame[:, x] > self.eps):
                        max_x = x
                        break
                for y in range(min_y):
                    if np.any(frame[y] > self.eps):
                        min_y = y
                        break
                for y in range(new_h - 1, max_y, -1):
                    if np.any(frame[y] > self.eps):
                        max_y = y
                        break
            if (max_x - min_x < self.threshold) or (max_y - min_y <
                                                    self.threshold):
                return results

            center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
            box_hwidth = (max_x - min_x) / 2 * (1 + self.padding)
            box_hheight = (max_y - min_y) / 2 * (1 + self.padding)
            if self.hw_ratio is not None:
                box_hheight = max(self.hw_ratio[0] * box_hwidth, box_hheight)
                box_hwidth = max(1 / self.hw_ratio[1] * box_hheight,
                                 box_hwidth)
            st_x, ed_x = center[0] - box_hwidth, center[0] + box_hwidth
            st_y, ed_y = center[1] - box_hheight, center[1] + box_hheight
            st_x, ed_x, st_y, ed_y = [int(x) for x in [st_x, ed_x, st_y, ed_y]]

            pad_left = 0 if st_x >= 0 else -st_x
            pad_right = 0 if ed_x <= new_w else ed_x - new_w
            pad_top = 0 if st_y >= 0 else -st_y
            pad_bottom = 0 if ed_y <= new_h else ed_y - new_h

            if st_x < 0:
                ed_x -= st_x
                st_x = 0
            if st_y < 0:
                ed_y -= st_y
                st_y = 0

            cropped_shape = (ed_y - st_y, ed_x - st_x)
            old_se = min(cropped_shape)
            scale_factor = self.shortedge / old_se
            new_h, new_w = int(cropped_shape[0] *
                               scale_factor), int(cropped_shape[1] *
                                                  scale_factor)
            new_shape = (new_h, new_w)
            results['img_shape'] = new_shape

            for f_idx in range(len(heatmaps)):
                frame = heatmaps[f_idx]
                frame = np.pad(
                    frame,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    'constant',
                    constant_values=0)
                frame = frame[st_y:ed_y, st_x:ed_x]
                frame = cv2.resize(frame, (new_w, new_h))
                heatmaps[f_idx] = frame
            results['imgs'] = heatmaps

        return results


# By default, the returned tensor is (T x num_clips) x H x W x C
# If the double option is set as True, the returned tensor will be
# (2 x T x num_clips) x H x W x C
@PIPELINES.register_module()
class GeneratePoseTarget(object):

    def __init__(
            self,
            sigma=0.6,
            use_score=False,
            with_kp=True,
            with_limb=False,
            # scale the size of generated kp heatmap
            scaling=1.,
            left=[],
            right=[],
            skeletons=[],
            double=False):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.scaling = scaling
        self.double = double
        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left = left
        self.right = right
        self.skeletons = skeletons

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 2, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 2, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # return an empty heatmap (since not in the image)
            if not (len(x) and len(y)):
                return heatmap
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            start, end = np.array(start), np.array(end)
            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 2, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 2, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                return heatmap

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            d2_start = ((x - start[0])**2 + (y - start[1])**2)
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)
            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    # sigma should have already been adjusted
    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        heatmaps = []
        if self.with_kp:
            num_kp = kps[0].shape[0]
            for i in range(num_kp):
                heatmaps.append(
                    self.generate_a_heatmap(img_h, img_w,
                                            [kp[i] for kp in kps], sigma,
                                            [value[i]
                                             for value in max_values]))

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = [kp[start_idx] for kp in kps]
                ends = [kp[end_idx] for kp in kps]
                start_values = [value[start_idx] for value in max_values]
                end_values = [value[end_idx] for value in max_values]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def gen_an_aug(self, results):
        all_kps = results['kp']
        kp_shape = results['kp'][0].shape
        num_person = results['num_person']

        if 'kpscore' in results:
            all_kpscores = results['kpscore']
        else:
            all_kpscores = [np.ones(kp_shape[:-1]) for i in range(num_person)]

        num_frame = kp_shape[0]

        if self.sigma is not None:
            sigma_ratio = np.ones(num_frame, dtype=np.float32)
            sigmas = self.sigma * sigma_ratio
        else:
            sigmas = [None] * num_frame

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps = [kp * self.scaling for kp in all_kps]

        imgs = []
        for i in range(num_frame):
            # We list shape of each item in the list
            kps = [kps[i] for kps in all_kps]  # num_kp x 2
            kpscores = [kpscores[i] for kpscores in all_kpscores]  # num_kp
            sigma = sigmas[i]  # 1

            num_kps = kpscores[0].shape[0]
            max_values = [np.ones(num_kps)] * results['num_person']
            if self.use_score:
                max_values = kpscores
            # just keep appending
            imgs.append(
                self.generate_heatmap(img_h, img_w, kps, sigma, max_values))

        return imgs

    def __call__(self, results):
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'

        if not self.double:
            results[key] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(flip_ratio=1, left=self.left, right=self.right)
            results_ = flip(results_)
            results[key] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        return results


# The Input will be a feature map ((N x T) x H x W x K), The output will be
# a 2D map: (N x H x W x [K * (2C + 1)])
# N is #clips x #crops, K is num_kpt
@PIPELINES.register_module()
class Heatmap2Potion:

    def __init__(self, C, option='full'):
        self.C = C
        self.option = option
        self.eps = 1e-4
        assert isinstance(C, int)
        assert C >= 2
        assert self.option in ['U', 'N', 'I', 'full']

    def __call__(self, results):
        assert results['modality'] == 'Pose'
        heatmaps = results['imgs']

        if 'clip_len' in results:
            clip_len = results['clip_len']
        else:
            # Just for Video-PoTion generation
            clip_len = heatmaps.shape[0]

        C = self.C
        heatmaps = heatmaps.reshape((-1, clip_len) + heatmaps.shape[1:])

        # t in {0, 1, 2, ..., clip_len - 1}
        def idx2color(t):
            st = np.zeros(C, dtype=np.float32)
            ed = np.zeros(C, dtype=np.float32)
            if t == clip_len - 1:
                ed[C - 1] = 1.
                return ed
            val = t / (clip_len - 1) * (C - 1)
            bin_idx = int(val)
            val = val - bin_idx
            st[bin_idx] = 1.
            ed[bin_idx + 1] = 1.
            return (1 - val) * st + val * ed

        heatmaps_wcolor = []
        for i in range(clip_len):
            color = idx2color(i)
            heatmap = heatmaps[:, i]
            heatmap = heatmap[..., None]
            heatmap = np.matmul(heatmap, color[None, ])
            heatmaps_wcolor.append(heatmap)

        # The shape of each element is N x H x W x K x C
        heatmap_S = np.sum(heatmaps_wcolor, axis=0)
        # The shape of U_norm is N x 1 x 1 x K x C
        U_norm = np.max(
            np.max(heatmap_S, axis=1, keepdims=True), axis=2, keepdims=True)
        heatmap_U = heatmap_S / (U_norm + self.eps)
        heatmap_I = np.sum(heatmap_U, axis=-1, keepdims=True)
        heatmap_N = heatmap_U / (heatmap_I + 1)
        if self.option == 'U':
            heatmap = heatmap_U
        elif self.option == 'I':
            heatmap = heatmap_I
        elif self.option == 'N':
            heatmap = heatmap_N
        elif self.option == 'full':
            heatmap = np.concatenate([heatmap_U, heatmap_I, heatmap_N],
                                     axis=-1)

        # Reshape the heatmap to 4D
        heatmap = heatmap.reshape(heatmap.shape[:3] + (-1, ))
        results['imgs'] = heatmap
        return results
