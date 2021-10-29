import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from genericpath import exists
import pickle
from loader.ContactPose.contactpose_dataset import get_object_names, ContactPose
from contactopt.util import vis_Joints3D,vis_2d_pose
import eval
def get_samples():
    """
    Gets all participants and objects from ContactPose
    Cuts out grasps with two hands or grasps using left hand
    :return: list of (participant_num, intent, object_name, ContactPose_object)
    """
    samples = []
    print('Reading ContactPose dataset')
    for participant_id in tqdm(range(1, 2)):
        for intent in ['handoff', 'use']:
            print(intent)
            for object_name in get_object_names(participant_id, intent):                
                cp = ContactPose(participant_id, intent, object_name, load_mano=False)
                if cp._valid_hands != [1]:  # If anything else than just the right hand, remove
                    #print(object_name)   #delete "hand"、"palm_print"   以及  handoff:bowl、utah_teapot;   use:banana、bowl、camera、ps_controller、water_bottle
                    continue              
                samples.append((participant_id, intent, object_name, cp))
    print('Valid ContactPose samples:', len(samples))
    return samples


       

if __name__ == "__main__":
    samples = get_samples()
    in_file = './data/optimized_{}.pkl'.format('single_perturbed')
    runs = pickle.load(open(in_file, 'rb'))
    print('Loaded {} len {}'.format(in_file, len(runs)))
    
    all_data = []
    for idx in range(len(samples)):
        p_num,object_name,cp = samples[idx][0], samples[idx][1], samples[idx][2]

        all_data.append(100*eval.process_sample(runs[idx], idx)[1]['unalign_hand_joints'])   #参考源代码中求MPJPE的参数
        print("The MPJPE in {} is : {}".format(os.path.join(str(p_num),object_name,cp),all_data[-1]))

    MPJPE = np.array(all_data).mean()
    print("Final MPJPE in contactpose is : {}".format(MPJPE))

