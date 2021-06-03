from itertools import islice
import numpy as np
from sklearn.utils import murmurhash3_32 as mmh3
import argparse
from multiprocessing import Pool
import time

# TODO: need to be changed when changing dataset
parser = argparse.ArgumentParser()
parser.add_argument("--n_cores", default=32, type=int)
parser.add_argument("--B", default=800, type=int)
parser.add_argument("--write_loc", default='../lookup/feature_hash', type=str)
parser.add_argument('--num_features', type=int, default=5000, help="number of input features")
parser.add_argument('--seed', type=int, default=101, help='random seed (default: 101)')

args = parser.parse_args()
num_features = args.num_features
B = args.B


bucket_order = np.zeros(num_features, dtype=int)
#
for i in range(num_features):
    bucket = mmh3(i, seed=args.seed) % B
    bucket_order[i] = bucket
    #
np.save(args.write_loc+'/bucket_B_'+str(B)+'.npy',bucket_order)

