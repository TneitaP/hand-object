import pytorch3d
from pytorch3d.structures import Meshes
import torch
import numpy as np
from tqdm import tqdm
from load_obman import load_obman
from contactopt import util
from contactopt.run_contactopt import get_newest_checkpoint
from contactopt.optimize_pose import optimize_pose
from contactopt.hand_object import HandObject
from contactopt.arguments import run_contactopt_on_obman_parse_args

def generate_pointnet_features(necessary_param, obj_sampled_idx):
    """Calculates per-point features for pointnet. DeepContact uses these features"""
    obj_mesh = Meshes(verts=[torch.Tensor(necessary_param["obj_verts"])], faces=[torch.Tensor(necessary_param["obj_faces"])])
    hand_mesh = Meshes(verts=[torch.Tensor(necessary_param["hand_verts"])], faces=[torch.Tensor(util.get_mano_closed_faces())])

    obj_sampled_verts_tensor = obj_mesh.verts_padded()[:, obj_sampled_idx, :]
    _, _, obj_nearest = pytorch3d.ops.knn_points(obj_sampled_verts_tensor, hand_mesh.verts_padded(), K=1, return_nn=True)  # Calculate on object
    _, _, hand_nearest = pytorch3d.ops.knn_points(hand_mesh.verts_padded(), obj_sampled_verts_tensor, K=1, return_nn=True)  # Calculate on hand

    obj_normals = obj_mesh.verts_normals_padded()
    obj_normals = torch.nn.functional.normalize(obj_normals, dim=2, eps=1e-12)    # Because buggy mistuned value in Pytorch3d, must re-normalize
    norms = torch.sum(obj_normals * obj_normals, dim=2)  # Dot product
    obj_normals[norms < 0.8] = 0.6   # TODO hacky get-around when normal finding fails completely
    obj_normals_aug = obj_normals.detach().squeeze().numpy()
    
    obj_sampled_verts = necessary_param["obj_verts"][obj_sampled_idx, :]
    obj_sampled_normals = obj_normals[0, obj_sampled_idx, :].detach().numpy()
    hand_normals = hand_mesh.verts_normals_padded()[0, :, :].detach().numpy()

    hand_centroid = np.mean(necessary_param["hand_verts"], axis=0)
    obj_centroid = np.mean(necessary_param["obj_verts"], axis=0)

    # Hand features
    hand_one_hot = np.ones((necessary_param["hand_verts"].shape[0], 1))
    hand_vec_to_closest = hand_nearest.squeeze().numpy() - necessary_param["hand_verts"]
    hand_dist_to_closest = np.expand_dims(np.linalg.norm(hand_vec_to_closest, 2, 1), axis=1)
    hand_dist_along_normal = np.expand_dims(np.sum(hand_vec_to_closest * hand_normals, axis=1), axis=1)
    hand_dist_to_joint = np.expand_dims(necessary_param["hand_verts"], axis=1) - np.expand_dims(necessary_param["hand_joints"], axis=0)   # Expand for broadcasting
    hand_dist_to_joint = np.linalg.norm(hand_dist_to_joint, 2, 2)
    hand_dot_to_centroid = np.expand_dims(np.sum((necessary_param["hand_verts"] - obj_centroid) * hand_normals, axis=1), axis=1)

    # Object features
    obj_one_hot = np.zeros((obj_sampled_verts.shape[0], 1))
    obj_vec_to_closest = obj_nearest.squeeze().numpy() - obj_sampled_verts
    obj_dist_to_closest = np.expand_dims(np.linalg.norm(obj_vec_to_closest, 2, 1), axis=1)
    obj_dist_along_normal = np.expand_dims(np.sum(obj_vec_to_closest * obj_sampled_normals, axis=1), axis=1)
    obj_dist_to_joint = np.expand_dims(obj_sampled_verts, axis=1) - np.expand_dims(necessary_param["hand_joints"], axis=0)   # Expand for broadcasting ,#self.joints(21,3)
    obj_dist_to_joint = np.linalg.norm(obj_dist_to_joint, 2, 2)
    obj_dot_to_centroid = np.expand_dims(np.sum((obj_sampled_verts - hand_centroid) * obj_sampled_normals, axis=1), axis=1)

    # hand_feats = np.concatenate((hand_one_hot, hand_normals, hand_vec_to_closest, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
    # obj_feats = np.concatenate((obj_one_hot, obj_sampled_normals, obj_vec_to_closest, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)
    hand_feats = np.concatenate((hand_one_hot, hand_dot_to_centroid, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
    obj_feats = np.concatenate((obj_one_hot, obj_dot_to_centroid, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)

    return hand_feats, obj_feats,obj_normals_aug

def prepare_param(pose_dataset,img_idx):
    necessary_param = dict()
    data_gpu = dict()

    hand_verts3d = pose_dataset.get_verts3d(img_idx)
    hand_joints3d = pose_dataset.get_joints3d(img_idx)
    hand_faces = pose_dataset.get_faces3d(img_idx)
    hand_poses = pose_dataset.get_pca(img_idx)
    hand_shapes = pose_dataset.get_shape(img_idx)
    hand_mTc = pose_dataset.get_mTc(img_idx)
    obj_verts3d, obj_faces = pose_dataset.get_obj_verts_faces(img_idx)
    necessary_param["hand_verts"] = hand_verts3d
    necessary_param["hand_joints"] = hand_joints3d
    necessary_param["hand_faces"] = hand_faces
    necessary_param["obj_verts"] = obj_verts3d
    necessary_param["obj_faces"] = obj_faces

    obj_sampled_idx = np.random.randint(0, len(necessary_param["obj_verts"]), 2048) 
    hand_feats_aug, obj_feats_aug,obj_normals_aug = generate_pointnet_features(necessary_param , obj_sampled_idx)

    data_gpu['hand_verts_aug'] = torch.from_numpy(necessary_param["hand_verts"]).unsqueeze(0).cuda().float()
    data_gpu['hand_feats_aug'] = torch.from_numpy(hand_feats_aug).unsqueeze(0).cuda().float()
    data_gpu['obj_feats_aug'] = torch.from_numpy(obj_feats_aug).unsqueeze(0).cuda().float()
    data_gpu['obj_sampled_idx'] = torch.from_numpy(obj_sampled_idx).unsqueeze(0).cuda().long()
    data_gpu['obj_sampled_verts_aug'] = torch.Tensor(obj_verts3d)[torch.Tensor(obj_sampled_idx).long(), :].unsqueeze(0).cuda().float()
    data_gpu['hand_pose_aug'] = torch.from_numpy(hand_poses).unsqueeze(0).cuda()
    data_gpu['hand_beta_aug'] = torch.from_numpy(hand_shapes).unsqueeze(0).cuda()
    data_gpu['hand_mTc_aug'] = torch.from_numpy(hand_mTc).unsqueeze(0).cuda()

    data_gpu['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded([torch.Tensor(obj_normals_aug).cuda()], pad_value=-1)
    data_gpu['mesh_aug'] = Meshes(verts=[torch.Tensor(obj_verts3d)], faces=[torch.Tensor(obj_faces)])

    return data_gpu

def run_opt_on_obman(args):
    pose_dataset = load_obman(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = get_newest_checkpoint()   #get pre-defined model weight ,and model is DeepContactNet
    model.to(device)
    model.eval()
    
    for i in tqdm(range(0, args.img_nb, args.img_step)):
        batch_size = 1

        img_idx = args.img_idx + i
        data_gpu = prepare_param(pose_dataset,img_idx)
        
        network_out = model(data_gpu['hand_verts_aug'], data_gpu['hand_feats_aug'], data_gpu['obj_sampled_verts_aug'], data_gpu['obj_feats_aug'])
        hand_contact_target = util.class_to_val(network_out['contact_hand']).unsqueeze(2)
        obj_contact_target = util.class_to_val(network_out['contact_obj']).unsqueeze(2)

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
    #     obj_contact_upscale = util.upscale_contact(data_gpu['mesh_aug'], data_gpu['obj_sampled_idx'], obj_contact_target)

    #     for b in range(obj_contact_upscale.shape[0]):    # Loop over batch
    #         gt_ho = HandObject()
    #         in_ho = HandObject()
    #         out_ho = HandObject()
    #         gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'], b)
    #         in_ho.load_from_batch(data['hand_beta_aug'], data['hand_pose_aug'], data['hand_mTc_aug'], hand_contact_target, obj_contact_upscale, data['mesh_aug'], b)
    #         out_ho.load_from_batch(data['hand_beta_aug'], out_pose, out_mTc, data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], b, obj_rot=obj_rot)
    #         # out_ho.calc_dist_contact(hand=True, obj=True)
            
    #         all_data.append({'gt_ho': gt_ho, 'in_ho': in_ho, 'out_ho': out_ho})

    #     if args.vis:
    #         show_optimization(data, opt_state, hand_contact_target.detach().cpu().numpy(), obj_contact_upscale.detach().cpu().numpy(),
    #                           is_video=args.video, vis_method=args.vis_method)

    #     if idx >= args.partial > 0:   # Speed up for eval
    #         break

    # out_file = './data/optimized_{}.pkl'.format(args.split)
    # print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    # f =  open(out_file, 'wb')
    # pickle.dump(all_data,f)
    # f.close()


if __name__ == "__main__":

    args = run_contactopt_on_obman_parse_args()
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
    for k in defaults.keys():   # Override arguments that have not been manually set with defaults
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]


    run_opt_on_obman(args)
    