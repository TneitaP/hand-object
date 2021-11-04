import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from genericpath import exists
from loader.ContactPose.contactpose_dataset import get_object_names, ContactPose

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

def computeIoU(mask_gt_path,mask_fine_path):

    obj_IOU = 0
    handR_IOU = 0

    #for i in range(len(mask_gt_path)):
    for frame_idx, mask_path in enumerate(tqdm(mask_gt_path)):
        mask_gt = cv2.imread(mask_path)
        mask_fine = cv2.imread(mask_fine_path[frame_idx])

        #取出物体和手的各个通道，并重新排列
        obj_mask_gt = mask_gt[:,:,0].reshape(-1,1)
        obj_mask_fine = mask_fine[:,:,0].reshape(-1,1)
        handR_mask_gt = mask_gt[:,:,2].reshape(-1,1)
        hand_mask_fine = mask_fine[:,:,2].reshape(-1,1)

        obj_intersection = np.sum(np.logical_and(obj_mask_gt,obj_mask_fine))
        obj_allarea= np.sum(np.logical_or(obj_mask_gt,obj_mask_fine))
        obj_iou = obj_intersection / obj_allarea
        obj_IOU += obj_iou

        handR_intersection = np.sum(np.logical_and(handR_mask_gt,hand_mask_fine))
        handR_allarea = np.sum(np.logical_or(handR_mask_gt,hand_mask_fine))
        handR_iou = handR_intersection / handR_allarea
        handR_IOU += handR_iou

    return obj_IOU/len(mask_gt_path),handR_IOU/len(mask_fine_path)

def load_mask():
    samples = get_samples()   #目前只适用于右手

    mask_gt = []
    mask_fine = []
    for idx in range(len(samples)):
        for camera_name in ('kinect2_left', 'kinect2_right', 'kinect2_middle'):
            image_root = os.path.join(samples[idx][3].data_tmp_dir,"mask_full",camera_name,"color").replace("images","mask")
            if not exists(image_root):
                continue
            for frame_idx, image_name in enumerate(tqdm(next(os.walk(image_root))[2])):
                gt_root = os.path.join(image_root,image_name)
                fine_root = gt_root.replace("mask","mask_perturbed")   #change here can relize "gt-fine" or "gt-perturbed",choices["mask_fine","mask_perturbed"]
                mask_gt.append(gt_root)
                mask_fine.append(fine_root)
    return mask_gt,mask_fine         

if __name__ == "__main__":
    mask_gt = []
    mask_fine = []
    mask_gt,mask_fine = load_mask()
    if len(mask_gt)!=len(mask_fine):
        raise AssertionError
    obj_IOU, handR_IOU = computeIoU(mask_gt,mask_fine)
    print(obj_IOU, handR_IOU)

