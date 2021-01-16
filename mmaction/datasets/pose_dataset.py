import copy
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core import mean_class_accuracy, top_k_accuracy
from ..utils import get_root_logger
from .base import BaseDataset
from .registry import DATASETS

# In our design, the field of an annotations includes:
#     frame_dir: name of the video (can be extend for RGB + Pose later)
#     num_frame: number of frames
#     num_person: number of persons
#     label: action label
#     per_frame_box: list[num_person] of ndarray[num_frame * 4]. (optional),
#         follows [x, y, w, h].
#     kp: list[num_person] of ndarray[num_frame * num_kp * 2] (for each person,
#         all 0 if not exists)
#     kpscore: list[num_person] of ndarray[num_frame * num_kp] (for each
#         person, score should between [0, 1], -1 if sure that the kp not
#         exists)
#     img_shape, original_shape: The same. The img shape.


@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    num_frame, label, per_frame_box, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        **kwargs: Keyword arguments for ``BaseDataset``.
        byfreq (bool): Optional. If set, will set sample_freq for the dataset.
        power (float): Optional. Power of class frequency. Default: 1.
        valid_norm_range (tuple[float]): Normalize probability by number of
            valid frames. Only support linear now.
    """

    def __init__(self, ann_file, pipeline, **kwargs):
        additional_args = [
            'valid_ratio', 'valid_frame', 'byfreq', 'power', 'valid_norm_range'
        ]
        add_kwargs = {}
        for arg in additional_args:
            if arg in kwargs:
                add_kwargs[arg] = kwargs.pop(arg)

        super().__init__(
            ann_file, pipeline, start_index=0, modality='Pose', **kwargs)

        # Thresholding Training Examples
        if 'valid_ratio' in add_kwargs:
            ratio = add_kwargs['valid_ratio']
            assert isinstance(ratio, float)
            # Perform thresholding
            self.video_infos = [
                x for x in self.video_infos
                if x['num_valid'] / x['num_frame'] >= ratio
            ]

        if 'valid_frame' in add_kwargs:
            valid_frame = add_kwargs['valid_frame']
            assert isinstance(valid_frame, int)
            # Perform thresholding
            self.video_infos = [
                x for x in self.video_infos if x['num_valid'] >= valid_frame
            ]

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

        if 'byfreq' in add_kwargs and add_kwargs['byfreq']:
            power = 1.
            if 'power' in add_kwargs:
                power = add_kwargs['power']
            label_num, label_freq = self._label_freq(power=power)
            if 'valid_norm_range' in add_kwargs:
                valid_norm_range = add_kwargs['valid_norm_range']
                assert mmcv.is_tuple_of(valid_norm_range, float)
                assert valid_norm_range[0] < valid_norm_range[1]
                assert valid_norm_range[0] > 0
                # It will set intra_class_freq for each video in video_infos
                self.get_intra_class_freq(valid_norm_range)
                sample_freq = [
                    label_freq[x['label']] * x['intra_class_freq']
                    for x in self.video_infos
                ]
            else:
                sample_freq = [
                    label_freq[x['label']] / label_num[x['label']]
                    for x in self.video_infos
                ]
            self.sample_freq = np.array(sample_freq, dtype=np.float32)

    def get_intra_class_freq(self, valid_norm_range=(0.7, 1.3)):
        """Linear Only Now."""
        bkts = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            bkts[label].append(item)
        for k, videos in bkts.items():
            min_valid = min([x['num_valid'] for x in videos])
            max_valid = max([x['num_valid'] for x in videos])

            def get_freq(num_valid):
                pos = (num_valid - min_valid) / (max_valid - min_valid)
                range_diff = valid_norm_range[1] - valid_norm_range[0]
                return valid_norm_range[0] + range_diff * pos

            intra_class_freq = [get_freq(x['num_valid']) for x in videos]
            Z = sum(intra_class_freq)
            intra_class_freq = [x / Z for x in intra_class_freq]
            for video, freq in zip(videos, intra_class_freq):
                video['intra_class_freq'] = freq

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        for i, item in enumerate(data):
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            # will do that in prepare_frames, cuz there may be only a
            # 'filename' in annofile
            if 'num_frame' in item:
                item['total_frames'] = item['num_frame']
        return data

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # if 'filename' in results:
        #     results.update(mmcv.load(results['filename']))
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # if 'filename' in results:
        #     results.update(mmcv.load(results['filename']))
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        """Evaluation in rawframe dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (tuple[int]): K value for top_k_accuracy metric.
                Defaults: (1, 5).
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if not isinstance(topk, (int, tuple)):
            raise TypeError(
                f'topk must be int or tuple of int, but got {type(topk)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results
