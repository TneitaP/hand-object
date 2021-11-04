import numpy as np
import pickle
from tqdm import tqdm
from contactopt.util import calc_l2_err


if __name__ == "__main__":

    in_file = './data/run_opt_on_obman_test.pkl'
    runs = pickle.load(open(in_file, 'rb'))
    print('Loaded {} len {}'.format(in_file, len(runs)))
    for idx,data in enumerate(tqdm(runs)):
        gt_ho = data['gt_ho']
        in_ho = data['in_ho']
        out_ho = data['out_ho']
        mpjpe = 100 * calc_l2_err(gt_ho['hand_joints3d_gt'].squeeze(0).detach().cpu().numpy(), 
                                    out_ho['hand_joints3d_out'], axis=1) 
        print(mpjpe)
        print("gt_ho:",gt_ho['hand_joints3d_gt'])
        print("in_ho:",in_ho['hand_joints3d_aug'])
        print("out_ho:",out_ho['hand_joints3d_out'])
