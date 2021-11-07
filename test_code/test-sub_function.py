import numpy as np

from loader.run_opt_on_obman import generate_pointnet_features, run_mano_on_obman

def function__generate_pointnet_features():
    necessary_param = dict()
    necessary_param["hand_verts"] = np.loadtxt('C:/Users/zbh/Desktop/test/hand_verts.txt',dtype = np.float32)
    necessary_param["hand_joints"] = np.loadtxt('C:/Users/zbh/Desktop/test/hand_joints.txt',dtype = np.float32)
    necessary_param["obj_verts"] = np.loadtxt('C:/Users/zbh/Desktop/test/obj_verts.txt',dtype = np.float32)
    necessary_param["obj_faces"] = np.loadtxt('C:/Users/zbh/Desktop/test/obj_faces.txt',dtype = int)

    obj_sampled_idx = np.loadtxt('C:/Users/zbh/Desktop/test/obj_sampled_idx.txt',dtype=int)

    result1 = np.loadtxt('C:/Users/zbh/Desktop/test/hand_feats_aug.txt')
    result2 = np.loadtxt('C:/Users/zbh/Desktop/test/obj_feats_aug.txt')
    result3 = np.loadtxt('C:/Users/zbh/Desktop/test/obj_normals.txt')

    hand_feats_aug, obj_feats_aug,obj_normals_aug = generate_pointnet_features(necessary_param, obj_sampled_idx)

    print(result1 - hand_feats_aug)
    print(result2 - obj_feats_aug)
    print(result3 - obj_normals_aug)

def function__run_mano_on_obman():
    aug_hand_pose = np.loadtxt('C:/Users/zbh/Desktop/test/hand_pose.txt')
    aug_hand_beta = np.loadtxt('C:/Users/zbh/Desktop/test/hand_beta.txt')
    aug_hand_mTc = np.loadtxt('C:/Users/zbh/Desktop/test/hand_mTc.txt')

    result1 = np.loadtxt('C:/Users/zbh/Desktop/test/hand_verts.txt') 
    result2 = np.loadtxt('C:/Users/zbh/Desktop/test/hand_joints.txt') 
    hand_verts,hand_joints = run_mano_on_obman(aug_hand_pose,aug_hand_beta,aug_hand_mTc)

    print(result1 - hand_verts)
    print(result2 - hand_joints)

if __name__ == "__main__":
    
    #function__generate_pointnet_features()
    function__run_mano_on_obman()