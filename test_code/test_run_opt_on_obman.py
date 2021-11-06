import pytorch3d
from pytorch3d.structures import Meshes
import torch
import numpy as np
from tqdm import tqdm
import pickle
from loader.load_obman import load_obman
from contactopt import util
from contactopt.run_contactopt import get_newest_checkpoint
from contactopt.optimize_pose import optimize_obman_pose
from contactopt.arguments import run_contactopt_on_obman_parse_args
from manopth.manolayer import ManoLayer

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

    hand_feats = np.concatenate((hand_one_hot, hand_dot_to_centroid, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
    obj_feats = np.concatenate((obj_one_hot, obj_dot_to_centroid, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)

    return hand_feats, obj_feats,obj_normals_aug

def run_mano_on_obman(aug_hand_pose,aug_hand_beta,aug_hand_mTc):
    """Runs forward_mano, computing the hand vertices and joints based on pose/beta parameters.
        Handles numpy-pytorch-numpy conversion"""
    if aug_hand_pose.shape[0] == 48:   # Special case when we're loading GT honnotate
        #mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang", use_pca=True, center_idx=None, flat_hand_mean=True)
        mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang",use_pca=True, ncomps=45, side='right', flat_hand_mean=True)
    else:   # Everything else
        mano_model = ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

    pose_tensor = torch.Tensor(aug_hand_pose).unsqueeze(0)
    beta_tensor = torch.Tensor(aug_hand_beta).unsqueeze(0)
    tform_tensor = torch.Tensor(aug_hand_mTc).unsqueeze(0)
    mano_verts, mano_joints = util.forward_mano(mano_model, pose_tensor, beta_tensor, [tform_tensor])
    hand_verts = mano_verts.squeeze().detach().numpy()
    hand_joints = mano_joints.squeeze().detach().numpy()

    return hand_verts,hand_joints


def prepare_param(pose_dataset,img_idx):
    necessary_param = dict()
    data_gpu_gt = dict()
    data_gpu = dict()

    img = pose_dataset.get_image(img_idx)
    hand_verts3d = pose_dataset.get_verts3d(img_idx)
    hand_joints3d = pose_dataset.get_joints3d(img_idx)
    anno_poses = pose_dataset.get_pca(img_idx)
    hand_poses = np.zeros(48)
    hand_poses[3:] = anno_poses
    hand_shapes = pose_dataset.get_shape(img_idx)
    hand_mTc = pose_dataset.get_mTc(img_idx)
    obj_verts3d, obj_faces = pose_dataset.get_obj_verts_faces(img_idx)
    
    """hand_gt"""
    #根据pose的前18维，重新生成had_vert和hand_joints
    hand_verts_gt,hand_joints3d_gt = run_mano_on_obman(hand_poses, hand_shapes , hand_mTc)
    #util.save_obj(hand_verts_gt,'C:/Users/zbh/Desktop/hand_anno_pose_mano_in.obj')
    data_gpu_gt['oriImage'] = img
    data_gpu_gt['anno_verts'] = hand_verts3d
    data_gpu_gt['anno_joints'] = hand_joints3d
    data_gpu_gt['anno_poses'] = anno_poses
    data_gpu_gt['hand_verts_gt'] = torch.from_numpy(hand_verts_gt).unsqueeze(0).cuda().float()
    data_gpu_gt['hand_joints3d_gt'] = torch.from_numpy(hand_joints3d_gt).unsqueeze(0).cuda().float()
    data_gpu_gt['hand_pose_gt'] = torch.from_numpy(hand_poses).unsqueeze(0).cuda()
    data_gpu_gt['hand_beta_gt'] = torch.from_numpy(hand_shapes).unsqueeze(0).cuda()
    data_gpu_gt['hand_mTc_gt'] = torch.from_numpy(hand_mTc).unsqueeze(0).cuda()

    """hand_aug"""
    aug_trans = 0.05
    aug_rot = 0.1
    aug_pca = 0.5
    aug_t = np.random.randn(3) * aug_trans
    aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(45) * aug_pca)).astype(np.float32)

    #perturbed pose
    tmp_hand_pose = np.array(hand_poses)
    tmp_hand_pose += aug_p
    hand_pose_aug = tmp_hand_pose

    #perturbed mTc
    tmp_hand_mTc = np.array(hand_mTc)
    tmp_hand_mTc[:3,3] += aug_t
    hand_mTc_aug = tmp_hand_mTc

    hand_beta_aug = np.array(hand_shapes)   #beta 参数始终保持不变
    hand_verts_aug,hand_joints_aug = run_mano_on_obman(hand_pose_aug,hand_beta_aug,hand_mTc_aug)
    #util.save_obj(hand_verts_aug,'C:/Users/zbh/Desktop/hand_anno_pose_mano_aug.obj')
    data_gpu['hand_verts_aug'] = torch.from_numpy(hand_verts_aug).unsqueeze(0).cuda().float()
    data_gpu['hand_joints3d_aug'] = torch.from_numpy(hand_joints_aug).unsqueeze(0).cuda().float()
    data_gpu['hand_pose_aug'] = torch.from_numpy(hand_pose_aug).unsqueeze(0).cuda()
    data_gpu['hand_beta_aug'] = torch.from_numpy(hand_shapes).unsqueeze(0).cuda()
    data_gpu['hand_mTc_aug'] = torch.from_numpy(hand_mTc_aug).unsqueeze(0).cuda()

    obj_sampled_idx = np.random.randint(0, len(obj_verts3d), 2048)
    necessary_param["hand_verts"] = hand_verts_aug
    necessary_param["hand_joints"] = hand_joints_aug
    necessary_param["obj_verts"] = obj_verts3d
    necessary_param["obj_faces"] = obj_faces 
    hand_feats_aug, obj_feats_aug,obj_normals_aug = generate_pointnet_features(necessary_param , obj_sampled_idx)

    data_gpu['hand_feats_aug'] = torch.from_numpy(hand_feats_aug).unsqueeze(0).cuda().float()
    data_gpu['obj_feats_aug'] = torch.from_numpy(obj_feats_aug).unsqueeze(0).cuda().float()
    data_gpu['obj_sampled_idx'] = torch.from_numpy(obj_sampled_idx).unsqueeze(0).cuda().long()
    data_gpu['obj_sampled_verts_aug'] = torch.Tensor(obj_verts3d)[torch.Tensor(obj_sampled_idx).long(), :].unsqueeze(0).cuda().float()
    data_gpu['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded([torch.Tensor(obj_normals_aug).cuda()], pad_value=-1)
    data_gpu['mesh_aug'] = Meshes(verts=[torch.Tensor(obj_verts3d).cuda()], faces=[torch.Tensor(obj_faces).cuda()])

    return data_gpu_gt , data_gpu

    
def load_from_batch(hand_beta, hand_pose, hand_mTc, obj_mesh):
    data_gpu_out = dict()
    """Generate HO object from a torch dataloader batch"""
    obj_verts = obj_mesh.verts_list()[0]
    hand_beta = hand_beta.squeeze(0).detach().cpu().numpy()
    hand_pose = hand_pose.squeeze(0).detach().cpu().numpy()
    hand_mTc = hand_mTc.squeeze(0).detach().cpu().numpy()
    hand_verts_out,hand_joints3d_out = run_mano_on_obman(hand_pose , hand_beta , hand_mTc)

    data_gpu_out['hand_pose_out'] = hand_pose
    data_gpu_out['hand_beta_out'] = hand_beta
    data_gpu_out['hand_mTc_out'] = hand_mTc
    data_gpu_out['hand_verts_out'] = hand_verts_out
    data_gpu_out['hand_joints3d_out'] = hand_joints3d_out
    data_gpu_out['obj_verts_out'] = obj_verts

    return data_gpu_out

def run_opt_on_obman(args):
    pose_dataset = load_obman(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = get_newest_checkpoint()   #get pre-defined model weight ,and model is DeepContactNet
    model.to(device)
    model.eval()
    all_data = list()
    samples =  pickle.load(open('data/perturbed_contactpose_train.pkl','rb'))
    for i in tqdm(range(0, args.img_nb, args.img_step)):
        batch_size = 1

        img_idx = args.img_idx + i
        #data_gpu_gt , data_gpu = prepare_param(pose_dataset,img_idx)
        data_gpu_gt = samples[i]['ho_gt']
        data_gpu = samples[i]['ho_aug']
        obj_sampled_idx = samples[i]['obj_sampled_idx']
        obj_sampled_verts_aug = torch.Tensor(data_gpu.obj_verts)[torch.Tensor(obj_sampled_idx).long(), :].unsqueeze(0).cuda().float()
        head_feats_aug =  samples[i]['hand_feats_aug']
        obj_feats_aug = samples[i]['obj_feats_aug']
        with torch.no_grad():
            """ Code related to section 3.1, to estimate the target contact on hand and on object"""
            network_out = model(
                    torch.from_numpy(data_gpu.hand_verts).unsqueeze(0).cuda().float(),
                    torch.from_numpy(head_feats_aug).unsqueeze(0).cuda().float(), 
                    obj_sampled_verts_aug,
                    torch.from_numpy(obj_feats_aug).unsqueeze(0).cuda().float()
                    )
            hand_contact_target = util.class_to_val(network_out['contact_hand']).unsqueeze(2)
            obj_contact_target = util.class_to_val(network_out['contact_obj']).unsqueeze(2)

        if args.sharpen_thresh > 0: # If flag, sharpen contact
            print('Sharpening')
            obj_contact_target = util.sharpen_contact(obj_contact_target, slope=2, thresh=args.sharpen_thresh)
            hand_contact_target = util.sharpen_contact(hand_contact_target, slope=2, thresh=args.sharpen_thresh)
        
        data_new = dict()
        data_new['hand_pose_aug'] = torch.from_numpy(data_gpu.hand_pose).unsqueeze(0).cuda().float()
        data_new['hand_beta_aug'] =  torch.from_numpy(data_gpu.hand_beta).unsqueeze(0).cuda().float()
        data_new['hand_verts_aug'] =  torch.from_numpy(data_gpu.hand_verts).unsqueeze(0).cuda().float()
        #data_new['hand_mTc_aug'] = data_gpu.hand_mTc
        data_new['mesh_aug'] =  Meshes(verts=[torch.Tensor(data_gpu.obj_verts).cuda()], faces=[torch.Tensor(data_gpu.obj_faces).cuda()])
        data_new ['obj_sampled_idx'] =  torch.from_numpy(obj_sampled_idx).unsqueeze(0).cuda().long()
        data_new['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded([torch.Tensor(data_gpu.obj_normals).cuda()], pad_value=-1)
        data_new['obj_sampled_verts_aug'] =  obj_sampled_verts_aug
        if args.rand_re > 1:    # If we desire random restarts，#data_gpu['hand_mTc_aug']：【1，4，4】
            mtc_orig = data_gpu.hand_mTc
            print('Doing random optimization restarts')
            best_loss = torch.ones(batch_size) * 100000

            for re_it in range(args.rand_re):
                # Add noise to hand translation and rotation
                data_gpu.hand_mTc_aug = torch.from_numpy(mtc_orig).unsqueeze(0).cuda().float()
                random_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(torch.randn((batch_size, 3), device=device) * args.rand_re_rot / 180 * np.pi, 'ZYX')
                # Convert rotations given as Euler angles in radians to rotation matrices.
                data_gpu.hand_mTc_aug[:, :3, :3] = torch.bmm(random_rot_mat, data_gpu.hand_mTc_aug[:, :3, :3])#矩阵乘法
                data_gpu.hand_mTc_aug[:, :3, 3] += torch.randn((batch_size, 3), device=device) * args.rand_re_trans
                #util.save_obj(data_gpu['hand_verts_aug'].squeeze(0).detach().cpu().numpy(), 'C:/Users/zbh/Desktop/hand_aug.obj')
                #姿势优化
                data_new['hand_mTc_aug'] = data_gpu.hand_mTc_aug
                mano_model = ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=args.ncomps, side='right', flat_hand_mean=True).to(device)
                cur_result = optimize_obman_pose(mano_model, data_new, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
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
                print("第"+str(re_it)+"次：",loss_val)
                for b in range(batch_size):
                    if loss_val[b] < best_loss[b]:
                        best_loss[b] = loss_val[b]
                        out_pose[b, :] = cur_result[0][b, :]  #【1，18】
                        out_mTc[b, :, :] = cur_result[1][b, :, :]#【1，4，4】
                        obj_rot[b, :, :] = cur_result[2][b, :, :]#【1，3，3】

                # out_verts, _ = run_mano_on_obman(
                #     out_pose.squeeze(0).detach().cpu().numpy() , 
                #     data_gpu['hand_beta_aug'].squeeze(0).detach().cpu().numpy() ,                     
                #     out_mTc.squeeze(0).detach().cpu().numpy())
                # #util.save_obj(out_verts, 'C:/Users/zbh/Desktop/after_hand_aug.obj')
        data_gpu_out = load_from_batch(data_new['hand_beta_aug'], out_pose, out_mTc, data_new['mesh_aug'])
        all_data.append({'gt_ho': data_gpu_gt, 'in_ho': data_new, 'out_ho': data_gpu_out})

    out_file = './data/run_opt_on_obman_{}.pkl'.format(args.split)
    print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    f =  open(out_file, 'wb')
    pickle.dump(all_data,f)
    f.close()


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
                'cont_method': 5,   #original value 0
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
    