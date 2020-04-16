import argparse

from feature import run_feature
from model import run_model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--res_file', type=str)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    data_dir = args.data_dir
    res_file = args.res_file
    run_feature(data_dir)
    run_model(res_file)


if __name__ == '__main__':
    main()
