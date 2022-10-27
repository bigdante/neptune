import sys
import os

BASE_PATH = "/raid/liuxiao/nell_data/uncoref_ids"


def to_split(src, tgt, span):
    ids = [line for line_idx, line in enumerate(open(f'{BASE_PATH}/{src}.jsonl')) if span[0] <= line_idx < span[1]]
    with open(f'{BASE_PATH}/{tgt}.jsonl', 'w') as f:
        for _id in ids:
            f.write(_id)


if __name__ == '__main__':
    _src, _tgt = sys.argv[1], sys.argv[3]
    _span = [int(v) for v in sys.argv[2].split('-')]
    to_split(_src, _tgt, _span)
    # os.remove(f'/raid/liuxiao/NePtune1.0/log/iterations/{_tgt}')
