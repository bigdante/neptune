import json

from data_object import BaseSentence

from tqdm import tqdm
from collections import OrderedDict

BLOCK = 218323


def get_unextracted_ids():
    data = json.load(open('/home/liuxiao/NePtune1.0/data/sorted_invalid_blocks_gt_10.json'))
    id_to_number = OrderedDict((item[0], item[1]) for item in data)
    all_ids = 0
    marker, last_id, cnt = False, None, 0
    out = open(f'/home/liuxiao/NePtune1.0/data/ids/0.json', 'w')
    for sentence in tqdm(BaseSentence.objects.no_cache()):
        if all_ids > 0 and all_ids % BLOCK == 0:
            out = open(f'/home/liuxiao/NePtune1.0/data/ids/{all_ids // BLOCK}.json', 'w')
        if str(sentence.id) in id_to_number:
            marker = True
            last_id = str(sentence.id)
            # all_ids.append(str(sentence.id))
            out.write(str(sentence.id) + '\n')
            cnt += 1
            all_ids += 1
        if marker:
            if cnt < id_to_number[last_id]:
                out.write(str(sentence.id) + '\n')
                cnt += 1
                all_ids += 1
            else:
                marker = False
                cnt = 0
    print("Unextracted sentences:", all_ids)


if __name__ == '__main__':
    get_unextracted_ids()
