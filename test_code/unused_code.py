import torch
import pytorch3d
from pytorch3d.structures import Meshes

def trans_pkl_from_contactpose(samples):
    data_gpu_gt = dict()
    data_gpu = dict()

    """data_gpu_gt"""
    data_gpu_gt['hand_verts_gt'] = torch.from_numpy(samples['ho_gt'].hand_verts).unsqueeze(0).cuda().float()
    data_gpu_gt['hand_joints3d_gt'] = torch.from_numpy(samples['ho_gt'].hand_joints).unsqueeze(0).cuda().float()
    data_gpu_gt['hand_pose_gt'] = torch.from_numpy(samples['ho_gt'].hand_pose).unsqueeze(0).cuda()
    data_gpu_gt['hand_beta_gt'] = torch.from_numpy(samples['ho_gt'].hand_beta).unsqueeze(0).cuda()
    data_gpu_gt['hand_mTc_gt'] = torch.from_numpy(samples['ho_gt'].hand_mTc).unsqueeze(0).cuda()

    """ data_gpu """
    data_gpu['hand_verts_aug'] = torch.from_numpy(samples['ho_aug'].hand_verts).unsqueeze(0).cuda().float()    #
    data_gpu['hand_joints3d_aug'] = torch.from_numpy(samples['ho_aug'].hand_joints).unsqueeze(0).cuda().float()
    data_gpu['hand_pose_aug'] = torch.from_numpy(samples['ho_aug'].hand_pose).unsqueeze(0).cuda().float()  #
    data_gpu['hand_beta_aug'] = torch.from_numpy(samples['ho_aug'].hand_beta).unsqueeze(0).cuda().float()   #
    data_gpu['hand_mTc_aug'] = torch.from_numpy(samples['ho_aug'].hand_mTc).unsqueeze(0).cuda().float()
    data_gpu['hand_feats_aug'] = torch.from_numpy(samples['hand_feats_aug']).unsqueeze(0).cuda().float()
    data_gpu['obj_feats_aug'] = torch.from_numpy(samples['obj_feats_aug']).unsqueeze(0).cuda().float()
    data_gpu ['obj_sampled_idx'] =  torch.from_numpy(samples['obj_sampled_idx']).unsqueeze(0).cuda().long()

    data_gpu['mesh_aug'] =  Meshes(verts=[torch.Tensor(samples['ho_aug'].obj_verts).cuda()], faces=[torch.Tensor(samples['ho_aug'].obj_faces).cuda()])
    data_gpu['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded([torch.Tensor(samples['ho_aug'].obj_normals).cuda()], pad_value=-1)
    data_gpu['obj_sampled_verts_aug'] =  torch.Tensor(samples['ho_aug'].obj_verts)[torch.Tensor(samples['obj_sampled_idx']).long(), :].unsqueeze(0).cuda().float()

    return data_gpu_gt , data_gpu