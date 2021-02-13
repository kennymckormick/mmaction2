import copy
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core import mean_average_precision, mean_class_accuracy, top_k_accuracy
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class MADataset(BaseDataset):
    """MultiAttrDataset, which supports the recognition tags of multiple
    categories. Accept both video annotation files or rawframe annotation
    files.

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
        tag_label_types (list[str]): The types of tag labels, include 'single',
            'multi', 'soft'.
        filename_tmpl (str | None): Template for each filename. If set to None,
            video dataset is used. Default: None.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    # What if:
    # 1. type(label) is int: still int
    # 2. type(label) is list of int: convert to tensor
    # 3. type(label) is nd
    def __init__(self,
                 ann_file,
                 pipeline,
                 tag_categories,
                 tag_category_nums,
                 tag_label_types,
                 filename_tmpl=None,
                 random_pick_multi=False,
                 multi2soft=False,
                 **kwargs):
        assert len(tag_categories) == len(tag_category_nums)
        self.tag_categories = tag_categories
        self.tag_category_nums = {
            cate: num
            for cate, num in zip(tag_categories, tag_category_nums)
        }
        self.ok_types = ['single', 'multi', 'soft']
        for type in tag_label_types:
            assert type in self.ok_types
        self.tag_label_types = {
            cate: type
            for cate, type in zip(tag_categories, tag_label_types)
        }
        self.filename_tmpl = filename_tmpl
        self.num_categories = len(self.tag_categories)
        self.random_pick_multi = random_pick_multi

        self.start_index = kwargs.pop('start_index', 0)
        self.dataset_type = None
        self.multi2soft = multi2soft
        super().__init__(
            ann_file, pipeline, start_index=self.start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
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

    # The attr sample_by_class is ignored
    def prepare_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        for cate_name in self.tag_categories:
            if cate_name not in results:
                results[cate_name + '_mask'] = 0
                if self.tag_label_types[cate_name] == 'single':
                    results[cate_name] = -1
                elif self.tag_label_types[cate_name] in ['multi', 'soft']:
                    num = self.tag_category_nums[cate_name]
                    results[cate_name] = torch.ones(num) * -1.
                continue
            else:
                results[cate_name + '_mask'] = 1

            label = results[cate_name]
            if self.tag_label_types[cate_name] == 'single':
                # Just some assertion
                assert isinstance(label, int)
                assert label < self.tag_category_nums[cate_name]
            elif self.tag_label_types[cate_name] == 'multi':
                # Note that the sum for onehot here >= 1.0
                assert mmcv.is_list_of(label, int)
                if self.random_pick_multi:
                    results[cate_name] = int(np.random.choice(label))
                elif self.multi2soft:
                    soft = torch.zeros(self.tag_category_nums[cate_name])
                    soft[label] = 1.
                    soft /= len(label)
                    results[cate_name] = soft
                else:
                    onehot = torch.zeros(self.tag_category_nums[cate_name])
                    onehot[label] = 1.
                    results[cate_name] = onehot
            elif self.tag_label_types[cate_name] == 'soft':
                assert isinstance(label, np.ndarray)
                label = label.astype(np.float32)
                assert label.shape == (self.tag_category_nums[cate_name], )
                results[cate_name] = torch.from_numpy(label)

        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        return self.prepare_frames(idx)

    def prepare_test_frames(self, idx):
        return self.prepare_frames(idx)

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics=None,
                 metric_options=None,
                 logger=None):
        """Evaluation in HVU Video Dataset. We only support evaluating mAP for
        each tag categories. Since some tag categories are missing for some
        videos, we can not evaluate mAP for all tags.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Not used.
            metric_options (dict | None): Not used.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """
        # The metric for 'single' is top_k_accuracy and mean_class_accuracy
        # The metric for 'multi' is mean_average_precision

        # results should be a list of dict
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        eval_results = OrderedDict()
        for cate in self.tag_categories:
            cate_type = self.tag_label_types[cate]

            valid = [(cate in x) for x in self.video_infos]

            preds = [result[cate] for result, f in zip(results, valid) if f]
            gts = [item[cate] for item, f in zip(self.video_infos, valid) if f]

            # Will Return Top-1 and Top-5
            if cate_type == 'soft':
                continue

            if cate_type == 'multi':
                cate_num = self.tag_category_nums[cate]
                gts = [self.label2array(cate_num, label) for label in gts]
                if len(gts):
                    mAP = mean_average_precision(preds, gts)
                    eval_results[f'{cate}_mAP'] = mAP
                    log_msg = f'\n{cate}_mAP\t{mAP:.4f}'
                    print_log(log_msg, logger=logger)
            elif cate_type in ['soft', 'single']:
                topk = (1, 5)
                top_k_acc = top_k_accuracy(preds, gts, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'{cate}_top{k}_acc'] = acc
                    log_msg.append(f'\n{cate}_top{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)

                mean_acc = mean_class_accuracy(preds, gts)
                eval_results[f'{cate}_mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)

        return eval_results
