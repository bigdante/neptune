import json
from tqdm import tqdm

train_path = "./Re-DocRED-main/data/train_revised.json"
dev_path = "./Re-DocRED-main/data/dev_revised.json"
test_path = "./Re-DocRED-main/data/test_revised.json"
rel_info = "./Re-DocRED-main/data/rel_info.json"
'''
    按照原来的数据，如果h r t成立，那么h和r下面所有的mentions，一一配对也应该都成立
'''

def data_preprocess(path, data_type):
    data = json.load(open(path))
    ori_data = []
    samples = []
    relations = json.load(open(rel_info))
    descripts = []
    for index, item in tqdm(enumerate(data)):
        descript = " ".join([" ".join(sen) for sen in item.get("sents")])
        descripts.append(descript)
        ori_samples = []
        labels = item.get('labels', [])
        for label in labels:
            hs = item.get("vertexSet")[label['h']]
            ts = item.get("vertexSet")[label['t']]
            relation = label['r']
            for h in hs:
                for t in ts:
                    s = (
                        index,
                        h['name'],
                        t['name'],
                        relation
                    )
                    ori_samples.append(s)
        ori_samples = list(set(ori_samples))
        ori_data += ori_samples
        for i, vertex_i in enumerate(item.get("vertexSet")):
            for j, vertex_j in enumerate(item.get("vertexSet")):
                for vi in vertex_i:
                    for vj in vertex_j:
                        for r in list(relations.keys()):
                            s = (
                                index,
                                vi['name'],
                                vj['name'],
                                r
                            )
                            if s in ori_samples:
                                in_ori = True
                            else:
                                in_ori = False
                            samples.append((
                                index,
                                vi['name'],
                                vj['name'],
                                r,
                                in_ori
                            ))

    print("saveing......")
    json.dump(list(set(ori_data)), open(f"./ori_{data_type}_samples.json", "w"), indent=4)
    json.dump(list(set(samples)), open(f"./{data_type}_samples.json", "w"), indent=4)
    json.dump(descripts, open(f"./{data_type}_descripts.json", "w"), indent=4)
    print("save done")


if __name__ == '__main__':
    data_preprocess(train_path, "train")
    data_preprocess(dev_path, "dev")
