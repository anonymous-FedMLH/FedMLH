import argparse

def args_parser_fedmach():
    parser = argparse.ArgumentParser()
    # TODO: need to be changed when changing dataset
    parser.add_argument('--epochs', type=int, default=30, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_classes', type=int, default=30938, help="number of output classes")
    parser.add_argument('--n_train', type=int, default=14146, help="number of total training data")
    parser.add_argument('--n_test', type=int, default=6616, help='number of total test data')
    parser.add_argument('--n_threads', type=int, default=32, help="number of threads")
    parser.add_argument('--frac', type=float, default=0.4, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")

    parser.add_argument('--step_per_epoch', type=int, default=1, help="the number of steps per local epoch (should be changed during runtime)")
    parser.add_argument('--total_test_steps', type=int, default=1, help="the number of steps during test (should be changed during runtime)")

    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--algorithm', type=str, default='fedmach', help='algorithm name')
    parser.add_argument('--feature_hash', action='store_true', default=False, help='whether perform feature hashing')
    parser.add_argument('--num_hash_features', type=int, default=1000, help='number of hashed features')  
    # other arguments
    parser.add_argument('--B', type=int, default=500, help = "hash table size")
    parser.add_argument('--R', type=int, default=4, help = "number of repetitions")
    parser.add_argument('--dataset', type=str, default='Wiki10', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--freq_thres', type=float, default=0.01, help="frequency threshold of frequent classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args


def args_parser_fedavg():
    parser = argparse.ArgumentParser()

    # TODO: need to be changed when changing dataset
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_classes', type=int, default=30938, help="number of output classes")
    parser.add_argument('--n_train', type=int, default=14146, help="number of total training data")
    parser.add_argument('--n_test', type=int, default=6616, help='number of total test data')
    parser.add_argument('--n_threads', type=int, default=32, help="number of threads")

    parser.add_argument('--frac', type=float, default=0.4, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")

    parser.add_argument('--step_per_epoch', type=int, default=1, help="the number of steps per local epoch (should be changed during runtime)")
    parser.add_argument('--total_test_steps', type=int, default=1, help="the number of steps during test (should be changed during runtime)")

    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--algorithm', type=str, default='fedavg', help='algorithm name')
    parser.add_argument('--model_avg', action='store_true', default=False, help='ensemble the models')
    parser.add_argument('--num_model_avg', type=int, default=1, help='number of ensembled models')
    parser.add_argument('--feature_hash', action='store_true', default=False, help='whether perform feature hashing')
    parser.add_argument('--num_hash_features', type=int, default=1000, help='number of hashed features')   
    parser.add_argument('--reweight_feature', action='store_true', default=False, help='whether reweight features')
    parser.add_argument('--reweight_equal', action='store_true', default=False, help='whether reweight to equal counts')
    parser.add_argument('--feature_weight_loc', type=str, default='./lookup/bucket_class_prop.npy', help='feature weight file')
    parser.add_argument('--dim1', type=int, default=400, help='dimension of first hidden layer') 
    parser.add_argument('--dim2', type=int, default=400, help='dimension of second hidden layer')   
    # other arguments
    parser.add_argument('--dataset', type=str, default='Wiki10', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--freq_thres', type=float, default=0.01, help="frequency threshold of frequent classes")

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args