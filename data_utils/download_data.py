import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import jsonlines
def sort_and_return_indices(lst):
    # Use sorted() to sort the list and enumerate() to get the index and value pairs
    sorted_indices = sorted(enumerate(lst), key=lambda x: -x[1])
    # Extract the indices from the sorted list of index-value pairs
    indices = [index for index, _ in sorted_indices]
    return indices


def get_Ultra_HH(split: str, silent: bool = False, cache_dir: str = None):
    data = defaultdict(lambda: defaultdict(list))
    dataset = datasets.load_dataset('openbmb/UltraFeedback',split=split)
    for row in dataset:
        # print(row['source'])
        # print(row['instruction'])
        prompt=row['instruction']
        overscore_l=[i['overall_score'] for i in row['completions']]
        # print(overscore_l)
        index=sort_and_return_indices(overscore_l)
        if len(index)>=3:
            # data[prompt]['demonstration']=row['completions'][index[0]]['response']
            chosen=row['completions'][index[1]]['response']
            rejected=row['completions'][index[-1]]['response']
            responses=[chosen,rejected]
            n_responses = len(data[prompt]['responses'])
            n_demonstration = len(data[prompt]['sft_target'])
            # data[prompt]['pairs'].append((n_responses, n_responses + 1, n_demonstration))
            # data[prompt]['responses'].extend(responses)
            data[prompt]['sft_target'] = row['completions'][index[0]]['response']
            data[prompt]['chosen']=chosen
            data[prompt]['reject']=rejected
    return data

ds_prefernce = get_Ultra_HH("train[:20000]")
ds_prefernce = dict(ds_prefernce)

f = open("data/output_path.json","w")
writer = jsonlines.Writer(f)
for i in ds_prefernce.keys():
    writer.write({"demon_prompt":i,"demon":ds_prefernce[i]["sft_target"],"chosen":ds_prefernce[i]["chosen"], "rejected":ds_prefernce[i]["reject"]})
# ds_prefernce.to_json(f"data/split_prefer_prompt.json")