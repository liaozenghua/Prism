import json
import pickle
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)

def read_jsonl(fname):
    with open(fname, 'r') as json_file:
        json_list = list(json_file)
    dev_set = []
    for (i, json_str) in enumerate(json_list):
        try:
            result = json.loads(json_str)
            dev_set.append(result)
        except Exception as e:
            print(i)
            print(json_str)
            print(e)
            raise Exception('end')
    return dev_set

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def read_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)