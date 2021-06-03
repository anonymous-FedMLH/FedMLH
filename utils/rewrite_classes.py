import numpy as np
import argparse

# TODO: need to be changed when changing dataset
parser = argparse.ArgumentParser()
parser.add_argument("--label_count", default='../lookup/label_frequency.npy', type=str)
parser.add_argument("--lookup_table_loc", default='../lookup/b_500/', type=str)
parser.add_argument('--num_hash_table', type=int, default=4, help="number of hash tables")
parser.add_argument('--R', type=int, default=500, help="hash table size")
parser.add_argument('--seed', type=int, default=101, help='random seed (default: 101)')

args = parser.parse_args()
nhash = args.num_hash_table
R = args.R
label_count = np.load(args.label_count)

lookup_tables = []
for i in range(nhash):
    lookup_tables += [np.load(args.lookup_table_loc+'bucket_order_'+str(i)+'.npy')]
lookup_tables = np.stack(lookup_tables)

bucket_count = np.zeros((nhash, R))
for i in range(R):
    for j in range(nhash):
        bucket_count[j, i] = np.sum(label_count[lookup_tables[j]==i])


class_bucket_prop = np.zeros((lookup_tables.shape[1]))
bucket_class_prop = np.zeros((lookup_tables.shape[1]))
for i in range(len(class_bucket_prop)):
    bucket_i = 0
    for j in range(nhash):
        bucket_i += bucket_count[j][lookup_tables[j,i]]
    class_bucket_prop[i] += (nhash*label_count[i])/bucket_i
    if class_bucket_prop[i]==0:
        bucket_class_prop[i] = 1
    else:
        bucket_class_prop[i] = 1/class_bucket_prop[i]


## write class_bucket_prop to file
np.save('../lookup/class_bucket_prop.npy', class_bucket_prop)
np.save('../lookup/bucket_class_prop.npy', bucket_class_prop)