import numpy as np
import pickle
from tqdm import tqdm

in_file = './data/run_opt_on_obman_test.pkl'
runs = pickle.load(open(in_file, 'rb'))
print('Loaded {} len {}'.format(in_file, len(runs)))

for idx,data in enumerate(tqdm(runs)):
    gt_ho = data['gt_ho']
    in_ho = data['in_ho']
    out_ho = data['out_ho']
    print(out_ho)