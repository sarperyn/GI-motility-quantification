import argparse
import yaml

def train_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_id',type=str, default='exp/0')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--lr', type=float, default='3e-3')
    parser.add_argument('--bs', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args

def test_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir',type=str, default='/home/syurtseven/gsoc/scripts/results')
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--sample_size', type=int, default=30)
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()
    return args