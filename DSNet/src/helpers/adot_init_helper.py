import argparse
import logging
import random
from pathlib import Path
import easydict

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('--model', type=str, default = 'anchor-based',
                        choices=('anchor-based', 'anchor-free'))

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--nms-thresh', type=float, default=0.5) #default0.5

    # inference
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=10) #default 15->eunsun custom=5로 변경
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)

    # common model config
    parser.add_argument('--base-model', type=str, default='attention',
                        choices=['attention', 'lstm', 'linear', 'bilstm',
                                 'gcn'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)

    # anchor based
    parser.add_argument('--neg-sample-ratio', type=float, default=2.0)
    parser.add_argument('--incomplete-sample-ratio', type=float, default=1.0)
    parser.add_argument('--pos-iou-thresh', type=float, default=0.6)
    parser.add_argument('--neg-iou-thresh', type=float, default=0.0)
    parser.add_argument('--incomplete-iou-thresh', type=float, default=0.3)
    parser.add_argument('--anchor-scales', type=int, nargs='+',
                        default=[4, 8, 16, 32]) #default=[4, 8, 16, 32]

    # anchor free
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    parser.add_argument('--cls-loss', type=str, default='focal',
                        choices=['focal', 'cross-entropy'])
    parser.add_argument('--reg-loss', type=str, default='soft-iou',
                        choices=['soft-iou', 'smooth-l1'])

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    print('--1--')
    # args = parser.parse_args()
    args = easydict.EasyDict({
      "model":  'anchor-based',
      "divice": 'cuda',
      "seed": 12345, 
      "splits": [],
      "max-epoch": 300,
      "model-dir" : '../models/model',
      '--log-file': 'log.txt',
      "lr": 5e-5,
      "weight-decay": 1e-5,
      "lambda-reg": 1.0,
      "nms-thresh": 0.5, 
      "sample-rate": 10, 
      "base-model": 'attention',
      "num-head": 8, 
      "num-feature": 1024,
      "num-hidden": 128,
      "neg-sample-ratio": 2.0,
      "incomplete-sample-ratio": 1.0,
      "pos-iou-thresh": 0.6,
      "neg-iou-thresh": 0.0,
      "incomplete-iou-thresh": 0.3,
      "anchor-scales": [4, 8, 16, 32]})
        
    print('--2--')
    return args
