# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append("./")
from contactopt.loader import ContactDBDataset
from contactopt.deepcontact_net import DeepContactNet
import glob
import argparse
from contactopt.optimize_pose import optimize_pose
from contactopt.visualize import show_optimization
import pickle
from contactopt.hand_object import HandObject
import contactopt.util as util
from tqdm import tqdm
import contactopt.arguments as arguments
import time
import torch
import os
from torch.utils.data import DataLoader
import pytorch3d
import numpy as np


def get_newest_checkpoint():
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    :return: Model with loaded weights
    """
    list_of_files = glob.glob('./model/*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    model = DeepContactNet()
    model.load_state_dict(torch.load(latest_file))
    return model


def run_contactopt(args):
    """
    Actually run ContactOpt approach. Estimates target contact with DeepContact,
    then optimizes it. Performs random restarts if selected.
    Saves results to a pkl file.
    :param args: input settings
    读取输入的设置：args，先使用DeepContact估计一个接触，然后对其进行优化，并保存为.pkl文件
    """
    print('Running split', args.split)
    dataset = ContactDBDataset(args.train_dataset, min_num_cont=args.min_cont)  #数据读取
    shuffle = args.vis or args.partial > 0 #判断是否重新打乱数据
    print('Shuffle:', shuffle)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=6, collate_fn=ContactDBDataset.collate_fn)
    #：collate_fn：表示的是如何读取样本，可以自己定义函数来准确的说明想要实现的功能
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #cuda or cpu

    model = get_newest_checkpoint()   #get pre-defined model weight ,and model is DeepContactNet
    model.to(device)
    model.eval()

    all_data = list()

    for idx, data in enumerate(tqdm(test_loader)):
        data_gpu = util.dict_to_device(data, device)  #数据转到gpu
        batch_size = data['obj_sampled_idx'].shape[0]

        if args.split != 'fine':
            with torch.no_grad():
                #将手顶点【1，778，3】，手特征【1，778，25】，物体采样的顶点【1，2048，3】，物体采样特征【1，2048，25】
                # 送入网络，输出contact_hand【1，778，10】和contact_obj【1，2048，10】
                network_out = model(data_gpu['hand_verts_aug'], data_gpu['hand_feats_aug'], data_gpu['obj_sampled_verts_aug'], data_gpu['obj_feats_aug'])
              #Finds the highest softmax for each class，变成二维度，然后变成三维度
                hand_contact_target = util.class_to_val(network_out['contact_hand']).unsqueeze(2)
                obj_contact_target = util.class_to_val(network_out['contact_obj']).unsqueeze(2)
        else:
            hand_contact_target = data_gpu['hand_contact_gt']
            obj_contact_target = util.batched_index_select(data_gpu['obj_contact_gt'], 1, data_gpu['obj_sampled_idx'])

        if args.sharpen_thresh > 0: # If flag, sharpen contact
            print('Sharpening')
            obj_contact_target = util.sharpen_contact(obj_contact_target, slope=2, thresh=args.sharpen_thresh)
            hand_contact_target = util.sharpen_contact(hand_contact_target, slope=2, thresh=args.sharpen_thresh)

        if args.rand_re > 1:    # If we desire random restarts，#data_gpu['hand_mTc_aug']：【1，4，4】
            mtc_orig = data_gpu['hand_mTc_aug'].detach().clone()
            print('Doing random optimization restarts')
            best_loss = torch.ones(batch_size) * 100000

            for re_it in range(args.rand_re):
                # Add noise to hand translation and rotation
                data_gpu['hand_mTc_aug'] = mtc_orig.detach().clone()
                random_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(torch.randn((batch_size, 3), device=device) * args.rand_re_rot / 180 * np.pi, 'ZYX')
                # Convert rotations given as Euler angles in radians to rotation matrices.
                data_gpu['hand_mTc_aug'][:, :3, :3] = torch.bmm(random_rot_mat, data_gpu['hand_mTc_aug'][:, :3, :3])#矩阵乘法
                data_gpu['hand_mTc_aug'][:, :3, 3] += torch.randn((batch_size, 3), device=device) * args.rand_re_trans

                #姿势优化
                cur_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
                                           w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis, ncomps=args.ncomps,
                                           w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans, w_opt_pose=args.w_opt_pose,
                                           w_opt_rot=args.w_opt_rot,
                                           caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                           caps_on_hand=args.caps_hand,
                                           contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                           w_obj_rot=args.w_obj_rot, pen_it=args.pen_it)
                if re_it == 0:
                    out_pose = torch.zeros_like(cur_result[0])
                    out_mTc = torch.zeros_like(cur_result[1])
                    obj_rot = torch.zeros_like(cur_result[2])
                    opt_state = cur_result[3]

                loss_val = cur_result[3][-1]['loss']  #获得的loss值
                for b in range(batch_size):
                    if loss_val[b] < best_loss[b]:
                        best_loss[b] = loss_val[b]
                        out_pose[b, :] = cur_result[0][b, :]  #【1，18】
                        out_mTc[b, :, :] = cur_result[1][b, :, :]#【1，4，4】
                        obj_rot[b, :, :] = cur_result[2][b, :, :]#【1，3，3】

                # print('Loss, re', re_it, loss_val)
                # print('Best loss', best_loss)
        else:
            result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
                                   w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis, ncomps=args.ncomps,
                                   w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans, w_opt_pose=args.w_opt_pose,
                                   w_opt_rot=args.w_opt_rot,
                                   caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                   caps_on_hand=args.caps_hand,
                                   contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                   w_obj_rot=args.w_obj_rot, pen_it=args.pen_it)
            out_pose, out_mTc, obj_rot, opt_state = result

    #将物体mesh的顶点数量，由固定的2048变为原来数量
        obj_contact_upscale = util.upscale_contact(data_gpu['mesh_aug'], data_gpu['obj_sampled_idx'], obj_contact_target)

        for b in range(obj_contact_upscale.shape[0]):    # Loop over batch
            gt_ho = HandObject()
            in_ho = HandObject()
            out_ho = HandObject()
            gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'], b)
            in_ho.load_from_batch(data['hand_beta_aug'], data['hand_pose_aug'], data['hand_mTc_aug'], hand_contact_target, obj_contact_upscale, data['mesh_aug'], b)
            out_ho.load_from_batch(data['hand_beta_aug'], out_pose, out_mTc, data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], b, obj_rot=obj_rot)
            # out_ho.calc_dist_contact(hand=True, obj=True)
            
            all_data.append({'gt_ho': gt_ho, 'in_ho': in_ho, 'out_ho': out_ho})

        if args.vis:
            show_optimization(data, opt_state, hand_contact_target.detach().cpu().numpy(), obj_contact_upscale.detach().cpu().numpy(),
                              is_video=args.video, vis_method=args.vis_method)

        if idx >= args.partial > 0:   # Speed up for eval
            break

    out_file = './data/optimized_{}.pkl'.format(args.split)
    print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    f =  open(out_file, 'wb')
    pickle.dump(all_data,f)
    f.close()
    

if __name__ == '__main__':
    util.hack_filedesciptor()
    args = arguments.run_contactopt_parse_args()
    if args.split == 'aug':     # Settings defaults for Perturbed ContactPose
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.0,
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 2,
                    'w_opt_trans': 0.3,
                    'w_opt_rot': 1.0,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 0,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 8,
                    'rand_re_trans': 0.04,
                    'rand_re_rot': 5,
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'im' or args.split == 'demo':    # Settings defaults for image-based pose estimates
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
    elif args.split == 'fine':  # Settings defaults for small-scale refinement
        defaults = {'lr': 0.003,
                    'n_iter': 250,
                    'w_cont_hand': 0,
                    'sharpen_thresh': 0.3,
                    'ncomps': 15,
                    'w_cont_asym': 4,
                    'w_opt_trans': 0.03,
                    'w_opt_rot': 1.0,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 5,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 1,
                    'rand_re_trans': 0.00,
                    'rand_re_rot': 0,
                    'w_obj_rot': 0,
                    'vis_method': 5}
    for k in defaults.keys():   # Override arguments that have not been manually set with defaults
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]

    print(args)

    start_time = time.time()
    run_contactopt(args)
    print('Elapsed time:', time.time() - start_time)

