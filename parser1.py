import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection')
    parser.add_argument('-D', '--dataset', type=str, default='IO',
                        choices=['REENTRANCY', 'TIMESTAMP','IO'])
    parser.add_argument('-M', '--model', type=str, default='gcn',
                        choices=['gcn', 'gcet'])
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--lr_decay_steps', type=str, default='1,3')
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('-d', '--dropout', type=float, default=0.03)
    parser.add_argument('-f', '--filters', type=str, default='64,64,64')
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-t', '--threads', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=1180)
    parser.add_argument('--shuffle_nodes', action='store_true', default=True)
    parser.add_argument('-F', '--folds', default=5, choices=[3, 5, 10])
    parser.add_argument('--alpha', type=float, default=0.1)

    return parser.parse_args()
