import copy as cp
import io
import os
import os.path as osp
import pickle
import shutil
import warnings

import cv2
import mmcv
import numpy as np
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..registry import PIPELINES
from .augmentations import PoseFlip


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
class UntrimmedSampleFrames(object):
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 16.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
    """

    def __init__(self, clip_len=1, frame_interval=16, start_index=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.start_index = start_index

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + self.start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results


@PIPELINES.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        sample_range (int): Total sample range for dense sample.
            Default: 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Default: 10.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 sample_range=64,
                 num_sample_positions=10,
                 temporal_jitter=False,
                 out_of_bound_opt='loop',
                 test_mode=False):
        super().__init__(
            clip_len,
            frame_interval,
            num_clips,
            temporal_jitter,
            out_of_bound_opt=out_of_bound_opt,
            test_mode=test_mode)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets


@PIPELINES.register_module()
class UniformSampleFrames:

    def __init__(self,
                 clip_len,
                 sample_ratio=-1,
                 num_clips=1,
                 test_mode=False,
                 random_seed=255):
        self.clip_len = clip_len
        self.sample_ratio = sample_ratio
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.random_seed = random_seed

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
        np.random.seed(self.random_seed)
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
        if self.sample_ratio <= 0:
            clip_len = self.clip_len
        else:
            clip_len = int(num_frames / self.sample_ratio)
            # self.clip_len is the upper bound
            clip_len = min(clip_len, self.clip_len)

        if self.test_mode:
            inds = self._get_test_clips(num_frames, clip_len)
        else:
            inds = self._get_train_clips(num_frames, clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int)
        results['clip_len'] = self.clip_len

        if self.test_mode:
            results['clip_len'] = clip_len
        else:
            if self.sample_ratio > 0:
                results['real_clip_len'] = clip_len

        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class SampleProposalFrames(SampleFrames):
    """Sample frames from proposals in the video.

    Required keys are "total_frames" and "out_proposals", added or
    modified keys are "frame_inds", "frame_interval", "num_clips",
    'clip_len' and 'num_proposals'.

    Args:
        clip_len (int): Frames of each sampled output clip.
        body_segments (int): Number of segments in course period.
        aug_segments (list[int]): Number of segments in starting and
            ending period.
        aug_ratio (int | float | tuple[int | float]): The ratio
            of the length of augmentation to that of the proposal.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        test_interval (int): Temporal interval of adjacent sampled frames
            in test mode. Default: 6.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        mode (str): Choose 'train', 'val' or 'test' mode.
            Default: 'train'.
    """

    def __init__(self,
                 clip_len,
                 body_segments,
                 aug_segments,
                 aug_ratio,
                 frame_interval=1,
                 test_interval=6,
                 temporal_jitter=False,
                 mode='train'):
        super().__init__(
            clip_len,
            frame_interval=frame_interval,
            temporal_jitter=temporal_jitter)
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.test_interval = test_interval

    def _get_train_indices(self, valid_length, num_segments):
        """Get indices of different stages of proposals in train mode.

        It will calculate the average interval for each segment,
        and randomly shift them within offsets between [0, average_duration].
        If the total number of frames is smaller than num segments, it will
        return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (valid_length + 1) // num_segments
        if avg_interval > 0:
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = base_offsets + np.random.randint(
                avg_interval, size=num_segments)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_val_indices(self, valid_length, num_segments):
        """Get indices of different stages of proposals in validation mode.

        It will calculate the average interval for each segment.
        If the total number of valid length is smaller than num segments,
        it will return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in validation mode.
        """
        if valid_length >= num_segments:
            avg_interval = valid_length / float(num_segments)
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_proposal_clips(self, proposal, num_frames):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices in the proposal's three
        stages: starting, course and ending stage.

        Args:
            proposal (object): The proposal object.
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0
        valid_length = duration - ori_clip_len

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        if self.mode == 'train':
            starting_offsets = self._get_train_indices(valid_starting_length,
                                                       self.aug_segments[0])
            course_offsets = self._get_train_indices(valid_length,
                                                     self.body_segments)
            ending_offsets = self._get_train_indices(valid_ending_length,
                                                     self.aug_segments[1])
        elif self.mode == 'val':
            starting_offsets = self._get_val_indices(valid_starting_length,
                                                     self.aug_segments[0])
            course_offsets = self._get_val_indices(valid_length,
                                                   self.body_segments)
            ending_offsets = self._get_val_indices(valid_ending_length,
                                                   self.aug_segments[1])
        starting_offsets += valid_starting
        course_offsets += start_frame
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        return offsets

    def _get_train_clips(self, num_frames, proposals):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices of each proposal, and then
        assemble them.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list): Proposals fetched.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        clip_offsets = []
        for proposal in proposals:
            proposal_clip_offsets = self._get_proposal_clips(
                proposal[0][1], num_frames)
            clip_offsets = np.concatenate(
                [clip_offsets, proposal_clip_offsets])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        It will calculate sampled frame indices based on test interval.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        return np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=np.int)

    def _sample_clips(self, num_frames, proposals):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list | None): Proposals fetched.
                It is set to None in test mode.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.mode == 'test':
            clip_offsets = self._get_test_clips(num_frames)
        else:
            assert proposals is not None
            clip_offsets = self._get_train_clips(num_frames, proposals)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        out_proposals = results.get('out_proposals', None)
        clip_offsets = self._sample_clips(total_frames, out_proposals)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        start_index = results['start_index']
        frame_inds = np.mod(frame_inds, total_frames) + start_index

        results['frame_inds'] = np.array(frame_inds).astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = (
            self.body_segments + self.aug_segments[0] + self.aug_segments[1])
        if self.mode in ['train', 'val']:
            results['num_proposals'] = len(results['out_proposals'])

        return results


@PIPELINES.register_module()
class PyAVInit(object):
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initiation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results


@PIPELINES.register_module()
class PyAVDecode(object):
    """Using pyav to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread

    def __call__(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max indice to make early stop
        max_inds = max(results['frame_inds'])
        i = 0
        for frame in container.decode(video=0):
            if i > max_inds + 1:
                break
            imgs.append(frame.to_rgb().to_ndarray())
            i += 1

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['imgs'] = [imgs[i % len(imgs)] for i in results['frame_inds']]

        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread})'
        return repr_str


@PIPELINES.register_module()
class DecordInit(object):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the Decord initiation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
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
        frame_dict = {
            idx: container[idx].asnumpy()
            for idx in np.unique(frame_inds)
        }

        imgs = [frame_dict[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class OpenCVInit(object):
    """Using OpenCV to initalize the video_reader.

    Required keys are "filename", added or modified keys are "new_path",
    "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        random_string = get_random_string()
        thread_id = get_thread_id()
        self.tmp_folder = osp.join(get_shm_dir(),
                                   f'{random_string}_{thread_id}')
        os.mkdir(self.tmp_folder)

    def __call__(self, results):
        """Perform the OpenCV initiation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        shutil.rmtree(self.tmp_folder)


@PIPELINES.register_module()
class OpenCVDecode(object):
    """Using OpenCV to decode the video.

    Required keys are "video_reader", "filename" and "frame_inds", added or
    modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Perform the OpenCV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
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
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

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

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # If no 'frame_inds' in results, set 'frame_inds' as range(num_frames)
        # by default

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'per_frame_box' in results:
            assert results['num_person'] == len(results['per_frame_box'])
            # the three items are lists
            # for storing, we use fp16, here we can convert them to float32
            results['per_frame_box'] = [
                x[frame_inds].astype(np.float32)
                for x in results['per_frame_box']
            ]
        if 'kpscore' in results:
            assert results['num_person'] == len(results['kpscore'])
            results['kpscore'] = [
                x[frame_inds].astype(np.float32) for x in results['kpscore']
            ]

        if 'kp' in results:
            assert results['num_person'] == len(results['kp'])
            results['kp'] = [
                x[frame_inds].astype(np.float32) for x in results['kp']
            ]

        if 'pose_box' in results:
            assert results['num_person'] == len(results['pose_box'])
            results['pose_box'] = [
                x[frame_inds].astype(np.float32) for x in results['pose_box']
            ]

        if 'compact_heatmap' in results:
            assert results['num_person'] == len(results['compact_heatmap'])
            results['compact_heatmap'] = [[x[ind] for ind in frame_inds]
                                          for x in results['compact_heatmap']]

        return results


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
                 **kwargs):
        self.io_backend = io_backend
        self.squeeze = squeeze
        self.kp2keep = kp2keep
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
            new_kpscore[person_ind, frame_ind] = kp[:, 2]

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
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

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
class LoadLocalizationFeature(object):
    """Load Video features for localizer with given video_name list.

    Required keys are "video_name" and "data_prefix",
    added or modified keys are "raw_feature".

    Args:
        raw_feature_ext (str): Raw feature file extension.  Default: '.csv'.
    """

    def __init__(self, raw_feature_ext='.csv'):
        valid_raw_feature_ext = ('.csv', )
        if raw_feature_ext not in valid_raw_feature_ext:
            raise NotImplementedError
        self.raw_feature_ext = raw_feature_ext

    def __call__(self, results):
        """Perform the LoadLocalizationFeature loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        data_prefix = results['data_prefix']

        data_path = osp.join(data_prefix, video_name + self.raw_feature_ext)
        raw_feature = np.loadtxt(
            data_path, dtype=np.float32, delimiter=',', skiprows=1)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results


@PIPELINES.register_module()
class GenerateLocalizationLabels(object):
    """Load video label for localizer with given video_name list.

    Required keys are "duration_frame", "duration_second", "feature_frame",
    "annotations", added or modified keys are "gt_bbox".
    """

    def __call__(self, results):
        """Perform the GenerateLocalizationLabels loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@PIPELINES.register_module()
class LoadProposals(object):
    """Loading proposals with given proposal results.

    Required keys are "video_name"
    added or modified keys are 'bsp_feature', 'tmin', 'tmax',
    'tmin_score', 'tmax_score' and 'reference_temporal_iou'.

    Args:
        top_k (int): The top k proposals to be loaded.
        pgm_proposals_dir (str): Directory to load proposals.
        pgm_features_dir (str): Directory to load proposal features.
        proposal_ext (str): Proposal file extension. Default: '.csv'.
        feature_ext (str): Feature file extension. Default: '.npy'.
    """

    def __init__(self,
                 top_k,
                 pgm_proposals_dir,
                 pgm_features_dir,
                 proposal_ext='.csv',
                 feature_ext='.npy'):
        self.top_k = top_k
        self.pgm_proposals_dir = pgm_proposals_dir
        self.pgm_features_dir = pgm_features_dir
        valid_proposal_ext = ('.csv', )
        if proposal_ext not in valid_proposal_ext:
            raise NotImplementedError
        self.proposal_ext = proposal_ext
        valid_feature_ext = ('.npy', )
        if feature_ext not in valid_feature_ext:
            raise NotImplementedError
        self.feature_ext = feature_ext

    def __call__(self, results):
        """Perform the LoadProposals loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        proposal_path = osp.join(self.pgm_proposals_dir,
                                 video_name + self.proposal_ext)
        if self.proposal_ext == '.csv':
            pgm_proposals = np.loadtxt(
                proposal_path, dtype=np.float32, delimiter=',', skiprows=1)

        pgm_proposals = np.array(pgm_proposals[:self.top_k])
        tmin = pgm_proposals[:, 0]
        tmax = pgm_proposals[:, 1]
        tmin_score = pgm_proposals[:, 2]
        tmax_score = pgm_proposals[:, 3]
        reference_temporal_iou = pgm_proposals[:, 5]

        feature_path = osp.join(self.pgm_features_dir,
                                video_name + self.feature_ext)
        if self.feature_ext == '.npy':
            bsp_feature = np.load(feature_path).astype(np.float32)

        bsp_feature = bsp_feature[:self.top_k, :]

        results['bsp_feature'] = bsp_feature
        results['tmin'] = tmin
        results['tmax'] = tmax
        results['tmin_score'] = tmin_score
        results['tmax_score'] = tmax_score
        results['reference_temporal_iou'] = reference_temporal_iou

        return results


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
        # print(self.compact)
        # Now that we get heatmaps, We can make them compact if needed,
        # note that it will be conducted by group
        if self.compact:
            # print('called')
            clip_len = results['clip_len']

            assert len(new_heatmaps) % clip_len == 0
            num_clips = len(new_heatmaps) // clip_len
            for i in range(num_clips):
                # Do not use per_frame_box
                st_frame, ed_frame = i * clip_len, (i + 1) * clip_len
                min_x, min_y, max_x, max_y = new_w - 1, new_h - 1, 0, 0
                for f_idx in range(st_frame, ed_frame):
                    # find th e
                    for x in range(min_x):
                        if np.any(heatmaps[f_idx][:, x] > self.eps):
                            min_x = x
                            break
                    for x in range(new_w - 1, max_x, -1):
                        if np.any(heatmaps[f_idx][:, x] > self.eps):
                            max_x = x
                            break
                    for y in range(min_y):
                        if np.any(heatmaps[f_idx][y] > self.eps):
                            min_y = y
                            break
                    for y in range(new_h - 1, max_y, -1):
                        if np.any(heatmaps[f_idx][y] > self.eps):
                            max_y = y
                            break
                if (max_x - min_x < self.threshold) or (max_y - min_y <
                                                        self.threshold):
                    continue
                center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                box_hwidth = (max_x - min_x) / 2 * (1 + self.padding)
                box_hheight = (max_y - min_y) / 2 * (1 + self.padding)
                if self.hw_ratio is not None:
                    box_hheight = max(self.hw_ratio[0] * box_hwidth,
                                      box_hheight)
                    box_hwidth = max(1 / self.hw_ratio[1] * box_hheight,
                                     box_hwidth)
                st_x, ed_x = center[0] - box_hwidth, center[0] + box_hwidth
                st_y, ed_y = center[1] - box_hheight, center[1] + box_hheight
                st_x, ed_x, st_y, ed_y = [
                    int(x) for x in [st_x, ed_x, st_y, ed_y]
                ]

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

                for f_idx in range(st_frame, ed_frame):
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
        results['modality'] = 'Heatmap'
        return results


# By default, the returned tensor is (T x num_clips) x H x W x C
# If the double option is set as True, the returned tensor will be
# (2 x T x num_clips) x H x W x C
@PIPELINES.register_module()
class GeneratePoseTarget(object):

    def __init__(self,
                 sigma=2,
                 use_score=False,
                 with_kp=True,
                 with_limb=False,
                 left=[],
                 right=[],
                 skeletons=[],
                 double=False):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
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

        if 'real_clip_len' in results:
            num_pad = results['clip_len'] - results['real_clip_len']
            for i in range(num_pad):
                imgs.append(np.zeros_like(imgs[0]))

        return imgs

    def __call__(self, results):
        if not self.double:
            results['imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = PoseFlip(flip_ratio=1, left=self.left, right=self.right)
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
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
