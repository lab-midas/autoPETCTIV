import numpy as np
import pandas as pd
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys
import os
import glob
import argparse
from scipy import integrate

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gt_path", required=True, help="Path to the nifti GT labels folder")
parser.add_argument("-p", "--prediction_path", required=False, help="Path to the nifti Pred labels folder")
args = parser.parse_args()


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg

def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score

def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol

def compute_interactive_metrics(nii_gt_paths, nii_pred_paths):
    assert len(nii_gt_paths) == 11 and len(nii_pred_paths) == 11 # 10 clicks + 1 initial prediction
    all_dice, all_fpv, all_fnv = [], [], []
    for (nii_gt_path, nii_pred_path) in zip(nii_gt_paths, nii_pred_paths):
        dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)
        all_dice.append(dice_sc)
        all_fpv.append(false_pos_vol)
        all_fnv.append(false_neg_vol)
    dice_auc = integrate.cumulative_trapezoid(all_dice, np.arange(11))[-1]
    fpv_auc = integrate.cumulative_trapezoid(all_fpv, np.arange(11))[-1]
    fnv_auc = integrate.cumulative_trapezoid(all_fnv, np.arange(11))[-1]
    dice_final = all_dice[-1]
    fpv_final = all_fpv[-1]
    fnv_final = all_fnv[-1]

    return dice_auc, fpv_auc, fnv_auc, dice_final, fpv_final, fnv_final

def get_metrics(gt_folder_path, pred_folder_path):
    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    with open("metrics.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        
        for nii_gt_path in glob.glob(os.path.join(gt_folder_path, '*.nii.gz')): 
            nii_pred_path = os.path.join(pred_folder_path, os.path.basename(nii_gt_path))
            if not os.path.exists(nii_pred_path):
                print(f"Missing prediction for: {os.path.basename(nii_gt_path)}")
                continue
            nii_gt_path = plb.Path(nii_gt_path)
            nii_pred_path = plb.Path(nii_pred_path)
            dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)
            writer.writerow([
                os.path.basename(nii_gt_path),
                dice_sc,
                false_pos_vol,
                false_neg_vol
            ])

if __name__ == "__main__":

    gt_folder_path = args.gt_path
    pred_folder_path = args.prediction_path
    get_metrics(gt_folder_path,pred_folder_path)

    #Aggregate metrics
    df = pd.read_csv('metrics.csv')
    numeric_cols = ['dice_sc', 'false_pos_vol', 'false_neg_vol']
    metrics = df[numeric_cols]
    print(metrics.mean())
