import sys
import translate
import argparse, pathlib
from ffnn import ffnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=pathlib.Path,
        help='Include CSV file using --file arg'
    )
    args = parser.parse_args()

    if not args.file:
        sys.exit('Include dataset to run network')

    x_train, y_train = translate.run(args.file)

    ffnn(x_train, y_train)
