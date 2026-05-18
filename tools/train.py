"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import datetime

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)    #各种yml组成字典cfg
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)       # DetSolver
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':
    #CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port=9909 --nproc_per_node=2 tools/train.py  
    
    parser = argparse.ArgumentParser()
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    # priority 0
    parser.add_argument('-c', '--config', type=str, default='./configs/gqdetr/gqdetr_fasternet_6x.yml')       #gqdetr_mbLarge_6x
    parser.add_argument('-r', '--resume', type=str, 
                        default='', help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, default='', help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default= 42, help='exp reproducibility')     #0
    parser.add_argument('--use-amp', action='store_true', default=True, help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, 
                        default=f'/home/fyl/workspace_fyl/GQ-light/exps_light/{time_str}', help='output directoy')
    parser.add_argument('--summary-dir', type=str, 
                        default=f'/home/fyl/workspace_fyl/GQ-light/exps_light/{time_str}', help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()
    

    main(args)
