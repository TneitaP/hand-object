from manopth import manolayer
import numpy as np
from open3d import io as o3dio
import json
import torch
from contactopt import util

with open("../../HOinter_data/Obman/Scene_inMeter_test/test_00000000_02992529_6682bf5d835701abe1a8044199c77d84_1/paramano_right.json","r") as param:
    param = json.load(param)
hand_file = "../../HOinter_data/Obman/Scene_inMeter_test/test_00000000_02992529_6682bf5d835701abe1a8044199c77d84_1/hand_right.ply"
hand_mesh = o3dio.read_triangle_mesh(hand_file)
obj_file = "../../HOinter_data/Obman/Scene_inMeter_test/test_00000000_02992529_6682bf5d835701abe1a8044199c77d84_1/object.ply"
obj_mesh = o3dio.read_triangle_mesh(obj_file)

hand_poses = np.array(param['pose'],dtype=np.float32)
hand_shapes = np.array(param['betas'],dtype=np.float32)
trans = np.array(param['trans'],dtype=np.float32)
hand_mTc = np.array(param['hTm'],dtype=np.float32).reshape(4,4)
obj_verts3d = np.array(obj_mesh.vertices, dtype=np.float32)     # Keep as floats since torch uses floats
hand_verts3d = np.array(hand_mesh.vertices, dtype=np.float32)     # Keep as floats since torch uses floats
util.save_obj(obj_verts3d, 'C:/Users/zbh/Desktop/222/'+ str(0) +'_obj_gt_ori.obj')
trans1 = trans.copy()
tmp0 = trans[0]
tmp1 = trans[1]
tmp2 = trans[2]
# trans1[0] = 5*tmp0
# trans1[1] = np.abs(5*tmp1)
# trans1[2] = 0.8*tmp2
obj_verts = (obj_verts3d - trans1.transpose())
util.save_obj(obj_verts, 'C:/Users/zbh/Desktop/222/'+ str(0) +'_obj_gt_chage.obj')
util.save_obj(hand_verts3d, 'C:/Users/zbh/Desktop/222/'+ str(0) +'_hand_gt_ori.obj')

if hand_poses.shape[0] == 48:   # Special case when we're loading GT honnotate
    #mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang", use_pca=True, center_idx=None, flat_hand_mean=True)
    mano_model = manolayer.ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang",use_pca=True, ncomps=45, side='right', flat_hand_mean=True)
else:   # Everything else
    mano_model = manolayer.ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

pose_tensor = torch.Tensor(hand_poses).unsqueeze(0)
beta_tensor = torch.Tensor(hand_shapes).unsqueeze(0)
tform_tensor = torch.Tensor(hand_mTc).unsqueeze(0)
mano_trans = torch.Tensor(trans).unsqueeze(0)
mano_verts, mano_joints = util.forward_mano(mano_model, pose_tensor, beta_tensor, [tform_tensor],mano_trans)
hand_verts = mano_verts.squeeze().detach().numpy()
hand_joints = mano_joints.squeeze().detach().numpy()
util.save_obj(hand_verts, 'C:/Users/zbh/Desktop/222/'+ str(0) +'_hand_out_trans.obj')

mano_verts, mano_joints = util.forward_mano(mano_model, pose_tensor, beta_tensor, [tform_tensor])
hand_verts = mano_verts.squeeze().detach().numpy()
hand_joints = mano_joints.squeeze().detach().numpy()
util.save_obj(hand_verts, 'C:/Users/zbh/Desktop/222/'+ str(0) +'_hand_out.obj')