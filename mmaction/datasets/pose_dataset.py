import copy
import os.path as osp

import mmcv
from mmcv.utils import print_log

from ..core import mean_class_accuracy, top_k_accuracy
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
    """

    def __init__(self, ann_file, pipeline, **kwargs):
        super().__init__(
            ann_file, pipeline, start_index=0, modality='Pose', **kwargs)

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
        if 'filename' in results:
            results.update(mmcv.load(results['filename']))
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        if 'filename' in results:
            results.update(mmcv.load(results['filename']))
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
