import sys
import translate
import argparse, pathlib
from ffnn import ffnn
from rnn import rnn
from cnn import cnn
from ae import ae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=pathlib.Path,
        help='Include CSV file using --file arg'
    )
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--rnn', action='store_true')
    parser.add_argument('--ae', action='store_true')
    parser.add_argument('--ffnn', action='store_true')
    args = parser.parse_args()

    if not args.file:
        sys.exit('Include dataset to run network using: --file <PATH>')

    x_train, y_train = translate.run(args.file)

    if args.ffnn:
        ffnn(x_train, y_train)
    elif args.rnn:
        rnn(x_train, y_train)
    elif args.cnn:
        cnn(x_train, y_train)
    elif args.ae:
        ae(x_train, y_train)
    else:
        sys.exit('Include network type: [--cnn, --rnn, --ae, --ffnn]')
