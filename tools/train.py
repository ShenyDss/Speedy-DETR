import sys
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.yaml_config import *
from src.solver import *
import src.misc.dist as dist


import argparse



def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['detection']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='../configs/', type=str, )
    parser.add_argument('--n_head', '-h', default= 3, type=int, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--f_b_ratio', '-a', type=float,)
    parser.add_argument('--distance', '-d', type=str,)
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False, )
    parser.add_argument('--amp', action='store_true', default=False, )

    args = parser.parse_args()

    main(args)

