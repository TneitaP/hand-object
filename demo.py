# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append('./')
import numpy as np
import trimesh
import json
from plyfile import PlyData
import contactopt.util as util
import contactopt.arguments as arguments
from contactopt.hand_object import HandObject
from contactopt.run_contactopt import run_contactopt
import transforms3d.quaternions as txq

def pose_matrix(pose):
  T = np.eye(3)
  T[:3, 3]  = pose['translation']
  T[:3, :3] = txq.quat2mat(pose['rotation'])
  return T

def create_demo_dataset():
    obj_mesh = trimesh.load('../ContactPose/data/contactpose_data/full1_handoff/banana/banana.obj') 
    with open('../ContactPose/data/contactpose_data/full1_handoff/banana/mano_fits_15.json') as json_file:  # Load mano parameters , include beta 、pose and trans
        mano_params = json.load(json_file)

    # Initialize the HandObject class with the given mano parameters and object mesh.
    # Note that pose must be represented using the 15-dimensional PCA space
    ho_gt = HandObject()  #Initialize the HandObject class   np.linalg.inv(mutils.pose_matrix(p['mTc']))
    ho_gt.load_from_mano_params(hand_beta=mano_params[1]["betas"], hand_pose=mano_params[1]["pose"], hand_trans=mano_params[1]["mTc"],
                                  obj_faces=obj_mesh.faces, obj_verts=obj_mesh.vertices)   #读取handobject（）类下的load_from_mano_params函数
                                                                                           #通过参数估计ho_pred

    aug_trans = 0.05   #参考官方代码生成手物初始姿态的过程，即随机初始化
    aug_rot = 0.1
    aug_pca = 0.5
    ho_pred = HandObject()
    aug_t = np.random.randn(3) * aug_trans
    aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(15) * aug_pca)).astype(np.float32)
    ho_pred.load_from_ho(ho_gt, aug_p, aug_t)

    new_sample = dict()   #define a new params named new_sample, with type of dict
    new_sample['ho_aug'] = ho_pred
    new_sample['ho_gt'] = ho_gt

    # Select the random object vertices which will be sampled,在[0,len(ho_gt.obj_verts))之间随机采样，输出2048个采样值
    new_sample['obj_sampled_idx'] = np.random.randint(0, len(ho_gt.obj_verts), util.SAMPLE_VERTS_NUM) #SAMPLE_VERTS_NUM = 2048

    # Calculate hand and object features. The network uses these for improved performance.
    #通过得出的2048个物体mesh顶点，来提高手和物体的特点，#'hand_feats_aug'：【778，25】，obj_feats_aug【2048，25】
    new_sample['hand_feats_aug'], new_sample['obj_feats_aug'] = ho_pred.generate_pointnet_features(new_sample['obj_sampled_idx'])

    return [new_sample]     # Return a dataset of length 1


if __name__ == '__main__':
    dataset = create_demo_dataset()
    args = arguments.run_contactopt_parse_args()   #read some default params

    defaults = {'lr': 0.01,
                'n_iter': 250,
                'w_cont_hand': 2.5,
                'sharpen_thresh': -1,
                'ncomps': 15,
                'w_cont_asym': 2,
                'w_opt_trans': 0.3,
                'w_opt_rot': 1,
                'w_opt_pose': 1.0,
                'caps_rad': 0.001,
                'cont_method': 0,
                'caps_top': 0.0005,
                'caps_bot': -0.001,
                'w_pen_cost': 320,
                'pen_it': 0,
                'rand_re': 8,
                'rand_re_trans': 0.02,
                'rand_re_rot': 5,
                'w_obj_rot': 0,
                'vis_method': 1}

    for k in defaults.keys():    #更新args中的一些默认参数
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]

    args.test_dataset = dataset
    args.split = 'user'

    run_contactopt(args)    #将所有参数包括dataset全部送入run_contactopt()函数内部
