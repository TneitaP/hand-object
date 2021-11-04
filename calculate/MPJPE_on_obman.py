import cv2
import numpy as np
import pickle
from tqdm import tqdm
from contactopt.util import calc_l2_err, save_obj, vis_Joints3D


if __name__ == "__main__":

    in_file = './data/run_opt_on_obman_test.pkl'
    runs = pickle.load(open(in_file, 'rb'))
    print('Loaded {} len {}'.format(in_file, len(runs)))
    for idx,data in enumerate(tqdm(runs)):
        gt_ho = data['gt_ho']
        in_ho = data['in_ho']
        out_ho = data['out_ho']
        save_obj()
        # gt_ho['oriImage'].save('C:/Users/zbh/Desktop/hand_mesh/'+ str(idx) +'_hand.jpg')
        # save_obj(gt_ho['anno_verts'], 'C:/Users/zbh/Desktop/hand_mesh/'+ str(idx) +'_hand_anno.obj')
        # save_obj(gt_ho['hand_verts_gt'].squeeze(0).detach().cpu().numpy(), 'C:/Users/zbh/Desktop/hand_mesh/'+ str(idx) +'_hand_gt.obj')
        # save_obj(out_ho['hand_verts_out'], 'C:/Users/zbh/Desktop/hand_mesh/'+ str(idx) +'_hand_out.obj')
        # mpjpe = 100 * calc_l2_err(gt_ho['hand_joints3d_gt'].squeeze(0).detach().cpu().numpy(), 
        #                             out_ho['hand_joints3d_out']/10.0, axis=1) 
        # vis_Joints3D(gt_ho['hand_joints3d_gt'].squeeze(0).detach().cpu().numpy(),vis=True)
        # vis_Joints3D(out_ho['hand_joints3d_out'],vis=True)

        # print(mpjpe)
        # print("gt_ho:",gt_ho['hand_joints3d_gt'])
        # print("in_ho:",in_ho['hand_joints3d_aug'])
        # print("out_ho:",out_ho['hand_joints3d_out'])
