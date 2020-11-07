import copy
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core import mean_average_precision, mean_class_accuracy, top_k_accuracy
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class HVUDataset(BaseDataset):
    """HVU dataset, which supports the recognition tags of multiple categories.
    Accept both video annotation files or rawframe annotation files.

    The dataset loads videos or raw frames and applies specified transforms to
    return a dict containing the frame tensors and other information.

    The ann_file is a json file with multiple dictionaries, and each dictionary
    indicates a sample video with the filename and tags, the tags are organized
    as different categories. Example of a video dictionary:

    .. code-block:: txt

        {
            'filename': 'gD_G1b0wV5I_001015_001035.mp4',
            'label': {
                'concept': [250, 131, 42, 51, 57, 155, 122],
                'object': [1570, 508],
                'event': [16],
                'action': [180],
                'scene': [206]
            }
        }

    Example of a rawframe dictionary:

    .. code-block:: txt

        {
            'frame_dir': 'gD_G1b0wV5I_001015_001035',
            'total_frames': 61
            'label': {
                'concept': [250, 131, 42, 51, 57, 155, 122],
                'object': [1570, 508],
                'event': [16],
                'action': [180],
                'scene': [206]
            }
        }


    Args:
        ann_file (str): Path to the annotation file, should be a json file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        tag_categories (list[str]): List of category names of tags.
        tag_category_nums (list[int]): List of number of tags in each category.
        filename_tmpl (str | None): Template for each filename. If set to None,
            video dataset is used. Default: None.
        action_ce (bool): Use cross entropy loss for 'action', if set as True,
            will use Top-k accuracy and mean_class_accuracy to evaluate action.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 tag_categories,
                 tag_category_nums,
                 filename_tmpl=None,
                 action_ce=False,
                 **kwargs):
        assert len(tag_categories) == len(tag_category_nums)
        self.tag_categories = tag_categories
        self.tag_category_nums = tag_category_nums
        self.filename_tmpl = filename_tmpl
        self.num_categories = len(self.tag_categories)
        self.num_tags = sum(self.tag_category_nums)
        self.category2num = dict(zip(tag_categories, tag_category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] +
                                  self.tag_category_nums[i])
        self.category2startidx = dict(zip(tag_categories, self.start_idx))
        self.start_index = kwargs.pop('start_index', 0)
        self.action_ce = action_ce
        self.dataset_type = None
        super().__init__(
            ann_file, pipeline, start_index=self.start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.json')
        return self.load_json_annotations()

    def load_json_annotations(self):
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)

        video_info0 = video_infos[0]
        assert ('filename' in video_info0) != ('frame_dir' in video_info0)
        path_key = 'filename' if 'filename' in video_info0 else 'frame_dir'
        self.dataset_type = 'video' if path_key == 'filename' else 'rawframe'
        if self.dataset_type == 'rawframe':
            assert self.filename_tmpl is not None

        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value

            # We will convert label to torch tensors in the pipeline
            video_infos[i]['categories'] = self.tag_categories
            video_infos[i]['category_nums'] = self.tag_category_nums
            if self.dataset_type == 'rawframe':
                video_infos[i]['filename_tmpl'] = self.filename_tmpl
                video_infos[i]['start_index'] = self.start_index
                video_infos[i]['modality'] = self.modality

        return video_infos

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='mean_average_precision',
                 metric_options=None,
                 logger=None):
        """Evaluation in HVU Video Dataset. We only support evaluating mAP for
        each tag categories. Since some tag categories are missing for some
        videos, we can not evaluate mAP for all tags.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'mean_average_precision'.
            metric_options (dict | None): Dict for metric options.
                Default: None.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]

        # There should be only one metric in the metrics list:
        # 'mean_average_precision'
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric == 'mean_average_precision'

        gt_labels = [ann['label'] for ann in self.video_infos]

        eval_results = {}
        for i, category in enumerate(self.tag_categories):

            start_idx = self.category2startidx[category]
            num = self.category2num[category]
            preds = [
                result[start_idx:start_idx + num]
                for video_idx, result in enumerate(results)
                if category in gt_labels[video_idx]
            ]
            gts = [
                gt_label[category] for gt_label in gt_labels
                if category in gt_label
            ]

            if category == 'action' and self.action_ce:
                # should be single label
                assert sum([1 for x in gts if len(x) == 1]) == len(gts)
                action_gt_labels = [x[0] for x in gts]
                topk = (1, 5)
                top_k_acc = top_k_accuracy(preds, action_gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'action_top{k}_acc'] = acc
                    log_msg.append(f'\naction_top{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)

                mean_acc = mean_class_accuracy(preds, action_gt_labels)
                eval_results['action_mean_class_accuracy'] = mean_acc
                log_msg = f'\naction_mean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
            else:
                gts = [self.label2array(num, item) for item in gts]

                mAP = mean_average_precision(preds, gts)
                eval_results[f'{category}_mAP'] = mAP
                log_msg = f'\n{category}_mAP\t{mAP:.4f}'
                print_log(log_msg, logger=logger)

        return eval_results
