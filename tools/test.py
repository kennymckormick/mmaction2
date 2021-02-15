import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmaction.apis import multi_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    if output_config.get('out', None):
        out = output_config['out']
        # make sure the dirname of the output path exists
        mmcv.mkdir_or_exist(osp.dirname(out))
        _, suffix = osp.splitext(out)
        # No need to consider AVA here
        assert suffix[1:] in file_handlers, (
            'The format of the output '
            'file should be json, pickle or yaml')

    assert out.endswith('.pkl'), 'for simplicity'

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        if args.average_clips is not None:
            cfg.test_cfg.average_clips = args.average_clips

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # No fp16 supports
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    assert distributed
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    rank, _ = get_dist_info()
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))

    if 'type' in cfg.data.test:
        dataset_type = cfg.data.test.type
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        data_loader = build_dataloader(dataset, **dataloader_setting)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
        if rank == 0:
            if output_config.get('out', None):
                out = output_config['out']
                print(f'\nwriting results to {out}')
                dataset.dump_results(outputs, **output_config)
            if dataset_type == 'MADataset':
                eval_res = dataset.evaluate(outputs)
                for name, val in eval_res.items():
                    print(f'{name}: {val:.04f}')
            elif eval_config:
                eval_res = dataset.evaluate(outputs, **eval_config)
                for name, val in eval_res.items():
                    print(f'{name}: {val:.04f}')
    else:
        for dataset_name, dataset in cfg.data.test.items():
            dataset_type = dataset.type
            dataset.test_mode = True
            dataset = build_dataset(dataset, dict(test_mode=True))
            data_loader = build_dataloader(dataset, **dataloader_setting)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
            if rank == 0:
                if output_config.get('out', None):
                    out = output_config['out']
                    out = out.replace('.pkl', f'_{dataset_name}.pkl')
                    print(f'\nwriting results to {out}')
                    dataset.dump_results(outputs, **output_config)
                # which is only used for MA Dataset
                assert dataset_type == 'MADataset'
                eval_res = dataset.evaluate(outputs)
                for name, val in eval_res.items():
                    print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
