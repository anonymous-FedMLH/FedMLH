import numpy as np
from collections import defaultdict
import argparse

# TODO: need to be changed when changing dataset
parser = argparse.ArgumentParser()
parser.add_argument("--file_loc", default='./data/train.txt', type=str)
parser.add_argument("--write_loc", default='./data/train_feature_hash_1000', type=str)
parser.add_argument('--num_features', type=int, default=1000, help="number of hashed input features")
parser.add_argument('--seed', type=int, default=101, help='random seed (default: 101)')

args = parser.parse_args()
num_features = args.num_features
file = args.file_loc
write_loc = args.write_loc


def load_feature_lookup(n_features):
    return np.load('../lookup/feature_hash/bucket_B_'+str(n_features)+'.npy')

lookup = load_feature_lookup(num_features)
new_file_array = []

with open(file, 'r', encoding='utf-8') as f:
    file_array = f.readlines()

for i, line in enumerate(file_array):
    itms = line.strip().split()
    new_itm = []
    idxs = [int(itm.split(':')[0]) for itm in itms[1:]]
    try:
        vals = [float(itm.split(':')[1]) for itm in itms[1:]]
    except:
        continue
    val_dict = defaultdict(float)
    # accumulate the values hashed into the same bucket
    for i, val in zip(idxs, vals):
        hash_i = lookup[i]
        val_dict[hash_i] += val
    for key in val_dict:
        new_itm += [str(key) + ':' + str(val_dict[key])]
    itms[1:] = new_itm
    new_itm = ' '.join(itms)
    new_itm = new_itm +'\n'
    new_file_array += [new_itm]

## Write new_file_array into text file
with open(write_loc, 'w', encoding='utf-8') as f:
    f.writelines(new_file_array)