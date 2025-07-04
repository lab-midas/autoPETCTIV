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

def false_pos_pix(ground_truth: np.ndarray, prediction: np.ndarray):
    """Count the number of false positive pixel, which do not overlap with the ground truth, based on the prediction
    and ground truth arrays.
    Returns zero if the prediction array is empty.
    Args:
        prediction (np.ndarray): The predicted array.
        ground_truth (np.ndarray): The ground truth array.
    returns:
        float: The number of false positive pixel which do not overlap with the ground truth.

    """
    
    if prediction.sum() == 0:
        return 0

    connected_components = cc3d.connected_components(
        prediction.astype(int), connectivity=18
    )
    false_positives = 0

    for idx in range(1, connected_components.max() + 1):
        component_mask = np.isin(connected_components, idx)
        if (component_mask * ground_truth).sum() == 0:
            false_positives += component_mask.sum()

    return false_positives

def false_neg_pix(ground_truth: np.ndarray, prediction: np.ndarray):
    """Count the number of false negative pixel, which do not overlap with the ground truth, based on the prediction
    and ground truth arrays.
    Returns nan if the ground truth array is empty.
    Args:
        prediction (np.ndarray): The predicted array.
        ground_truth (np.ndarray): The ground truth array.
    Returns:
        float: The number of false negative pixel, which do not overlap with the prediction.

    """
    
    if ground_truth.sum() == 0:
        return np.nan

    gt_components = cc3d.connected_components(ground_truth.astype(int), connectivity=18)
    false_negatives = 0

    for component_id in range(1, gt_components.max() + 1):
        component_mask = np.isin(gt_components, component_id)
        if (component_mask * prediction).sum() == 0:
            false_negatives += component_mask.sum()

    return false_negatives

def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate the Dice score between the prediction and ground truth arrays.
    Returns nan if the ground truth array is empty.
    Args:
        prediction (np.ndarray): The predicted array.
        ground_truth (np.ndarray): The ground truth array.
    Returns:
        float: The Dice score between the prediction and ground truth arrays.

    """
    if ground_truth.sum() == 0:
        return np.nan

    intersection = (ground_truth * prediction).sum()
    union = ground_truth.sum() + prediction.sum()
    dice_score = 2 * intersection / union

    return dice_score


def compute_metrics(nii_gt_path, nii_pred_path):
    # compute metrics core for a single test case
    # input: path of NIfTI segmentation file (ground truth and prediction)
    # output: Dice score, false positive volume, false negative volume
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol

    num_classes = np.unique(gt_array)
    # For Task 2, the ground truth masks have multi-labels.
    # For this case the dice will be computed for each individual class/label and average over the image.
    if len(num_classes)>2:
        dices = []
        for cls in num_classes:
            true_cls = (gt_array == cls).astype(np.uint8)
            pred_cls = (pred_array == cls).astype(np.uint8)
            dices.append(dice_score(true_cls,pred_cls))
        dice_sc = np.mean(dices)
    else:
        dice_sc = dice_score(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol

def compute_interactive_metrics(nii_gt_paths, nii_pred_paths, n_clicks):
    # compute the interactive metrics for a single test case after n_clicks interactive segmentations
    # input: path of NIfTI segmentation files (ground truth and prediction) after n_clicks iterations in the form of a list:
    #        nii_gt_paths = [nii_gt_path1, nii_gt_path2, ...] or nii_gt_path (str)
    #        nii_pred_paths = [nii_pred_path0, nii_pred_path1, ..., nii_pred_pathn] with n=n_clicks: 0=no click, 1=single foreground and background click, 2=two foreground and background clicks, ...
    # output: Dice score, false positive volume, false negative volume
    assert len(nii_pred_paths) == n_clicks 
    if isinstance(str(nii_gt_paths), str):
        nii_gt_paths = [str(nii_gt_paths) for _ in range(n_clicks)]
    all_dice, all_fpv, all_fnv = [], [], []
    for (nii_gt_path, nii_pred_path) in zip(nii_gt_paths, nii_pred_paths):
        dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)
        all_dice.append(dice_sc)
        all_fpv.append(false_pos_vol)
        all_fnv.append(false_neg_vol)
    dice_auc = integrate.cumulative_trapezoid(all_dice, np.arange(n_clicks))[-1]
    fpv_auc = integrate.cumulative_trapezoid(all_fpv, np.arange(n_clicks))[-1]
    fnv_auc = integrate.cumulative_trapezoid(all_fnv, np.arange(n_clicks))[-1]
    dice_final = all_dice[-1]
    fpv_final = all_fpv[-1]
    fnv_final = all_fnv[-1]

    return dice_auc, fpv_auc, fnv_auc, dice_final, fpv_final, fnv_final

def get_metrics(gt_folder_path, pred_folder_path):
    # Get metrics for a single test case and aggregate in a non-interactive setting (using directly predicted mask from all 10 click pairs—foreground and background)
    # input: path of NIfTI segmentation files (ground truth and prediction)
    # output: Dice score, false positive volume, false negative volume
    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    with open("metrics.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        
        # loop over all test cases
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

def get_interactive_metrics(gt_folder_path, pred_folder_path, n_clicks):
    # Get metrics for a single test case and aggregate in an interactive setting
    # input: path of NIfTI segmentation files (ground truth and prediction)
    # output: Dice score AUC, false positive volume AUC, false negative volume AUC, Dice score final, false positive volume final, false negative volume final
    csv_header = ['gt_name', 'dice_auc', 'fpv_auc', 'fnv_auc', 'dice_final', 'fpv_final', 'fnv_final']
    with open("metrics_interactive.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        
        # loop over all test cases
        for nii_gt_path in glob.glob(os.path.join(gt_folder_path, '*.nii.gz')): 
            nii_pred_paths = []
            for i in range(n_clicks+1):
                nii_pred_paths.append(os.path.join(pred_folder_path, os.path.basename(nii_gt_path).replace('.nii.gz', f'_pred{i}.nii.gz')))
            if not all(os.path.exists(p) for p in nii_pred_paths):
                print(f"Missing prediction for: {os.path.basename(nii_gt_path)}")
                continue
            nii_gt_path = plb.Path(nii_gt_path)
            dice_auc, fpv_auc, fnv_auc, dice_final, fpv_final, fnv_final = compute_interactive_metrics(nii_gt_path, nii_pred_paths, n_clicks)
            writer.writerow([
                os.path.basename(nii_gt_path),
                dice_auc,
                fpv_auc,
                fnv_auc,
                dice_final,
                fpv_final,
                fnv_final
            ])


if __name__ == "__main__":
    # example usage to call evaluation on complete test cohort, a single test case, or a single file
    # calls and interfaces are kept similar to grand-challenge.org execution environment

    # assumed directory structure (nii_gt files and nii_pred files can also be stored in the same folder):
    # /path/to/gt/labels
    #   |-- test_case_1
    #       |-- nii_gt_path_file.nii.gz
    #   |-- test_case_2
    #       |-- nii_gt_path_file.nii.gz
    #   |-- ...
    #
    # /path/to/pred/labels
    #   |-- test_case_1
    #       |-- nii_gt_path_file_pred0.nii.gz
    #       |-- nii_gt_path_file_pred1.nii.gz
    #       |-- ...
    #   |-- test_case_2
    #       |-- nii_gt_path_file_pred0.nii.gz
    #       |-- nii_gt_path_file_pred1.nii.gz
    #       |-- ...
    #   |-- ...
    
    # complete test cohort: 
    #   - non-interactive:  python evaluation_metrics.py -c -gt_path /path/to/gt/labels -prediction_path /path/to/pred/labels
    #   - interactive:      python evaluation_metrics.py -c -gt_path /path/to/gt/labels -prediction_path /path/to/pred/labels -n_clicks 11
    # single test case:     
    #   - non-interactive:  python evaluation_metrics.py -gt_path /path/to/gt/labels/test_case_1 -prediction_path /path/to/pred/labels/test_case_1
    #   - interactive:      python evaluation_metrics.py -gt_path /path/to/gt/labels/test_case_1.nii.gz -prediction_path /path/to/pred/labels/test_case_1.nii.gz -n_clicks 11
    # single file:

    gt_folder_path = args.gt_path
    pred_folder_path = args.prediction_path
    

    if os.path.isdir(gt_folder_path) and os.path.isdir(pred_folder_path):
        if args.c:
            # complete test cohort
            if not args.n_clicks:
                # non-interactive (similar to autoPET I - III)

                # prepare complete test cohort csv file
                csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
                df_cohort = pd.DataFrame(columns=csv_header)

                # loop over all test cases
                for test_case in glob.glob(gt_folder_path):
                    gt_folder = os.path.join(gt_folder_path, test_case)
                    pred_folder = os.path.join(pred_folder_path, test_case)

                    # compute metrics
                    get_metrics(gt_folder, pred_folder)
                    # aggregate metrics
                    df = pd.read_csv('metrics.csv')
                    df['gt_name'] = test_case
                    df_cohort = pd.concat([df_cohort, df], ignore_index=True)

                # save cohort metrics
                df_cohort.to_csv('metrics_cohort.csv', index=False)
                # remove temporary file
                os.remove('metrics.csv')
                # aggregate metrics
                numeric_cols = ['dice_sc', 'false_pos_vol', 'false_neg_vol']
                metrics = df_cohort[numeric_cols]
                print(metrics.mean())
            else:
                # interactive (autoPET IV)
                n_clicks = int(args.n_clicks)

                # prepare complete test cohort csv file
                csv_header = ['gt_name', 'dice_auc', 'fpv_auc', 'fnv_auc', 'dice_final', 'fpv_final', 'fnv_final']
                df_cohort = pd.DataFrame(columns=csv_header)

                # loop over all test cases
                for test_case in glob.glob(gt_folder_path):
                    gt_folder = os.path.join(gt_folder_path, test_case)
                    pred_folder = os.path.join(pred_folder_path, test_case)

                    # compute metrics
                    get_interactive_metrics(gt_folder, pred_folder, n_clicks)
                    # aggregate metrics
                    df = pd.read_csv('metrics_interactive.csv')
                    df['gt_name'] = test_case
                    df_cohort = pd.concat([df_cohort, df], ignore_index=True)
                
                # save cohort metrics
                df_cohort.to_csv('metrics_cohort_interactive.csv', index=False)
                # remove temporary file
                os.remove('metrics_interactive.csv')
                # aggregate metrics
                numeric_cols = ['dice_auc', 'fpv_auc', 'fnv_auc', 'dice_final', 'fpv_final', 'fnv_final']
                metrics = df_cohort[numeric_cols]
                print(metrics.mean())
        
        else:
            # single test case
            if not args.n_clicks:
                # non-interactive (similar to autoPET I - III)
                get_metrics(gt_folder_path, pred_folder_path)

                # aggregate metrics
                df = pd.read_csv('metrics.csv')
                numeric_cols = ['dice_sc', 'false_pos_vol', 'false_neg_vol']
                metrics = df[numeric_cols]
                print(metrics.mean())
            else:
                # interactive (autoPET IV)
                n_clicks = int(args.n_clicks)  # assuming n_clicks iterative segmentations were performed and n_clicks prediction files were generated (including no click=0th iteration)
                get_interactive_metrics(gt_folder_path, pred_folder_path, n_clicks)

                # aggregate metrics
                df = pd.read_csv('metrics_interactive.csv')
                numeric_cols = ['dice_auc', 'fpv_auc', 'fnv_auc', 'dice_final', 'fpv_final', 'fnv_final']
                metrics = df[numeric_cols]
                print(metrics.mean())

    else:
        # single test file
        if not args.n_clicks:
            # non-interactive (similar to autoPET I - III)
            dice_sc, false_pos_vol, false_neg_vol = compute_metrics(gt_folder_path, pred_folder_path)

            # print metrics
            print(f"Dice score: {dice_sc}")
            print(f"False positive volume: {false_pos_vol}")
            print(f"False negative volume: {false_neg_vol}")
        else:
            # interactive (autoPET IV)
            n_clicks = int(args.n_clicks)
            nii_gt_paths = [gt_folder_path for _ in range(n_clicks)]
            nii_pred_paths = []
            for i in range(n_clicks+1):
                nii_pred_paths.append(os.path.join(pred_folder_path, os.path.basename(gt_folder_path).replace('.nii.gz', f'_pred{i}.nii.gz')))
            dice_auc, fpv_auc, fnv_auc, dice_final, fpv_final, fnv_final = compute_interactive_metrics(nii_gt_paths, nii_pred_paths, n_clicks)
            # print metrics
            print(f"Dice score AUC: {dice_auc}")
            print(f"False positive volume AUC: {fpv_auc}")
            print(f"False negative volume AUC: {fnv_auc}")
            print(f"Dice score final: {dice_final}")
            print(f"False positive volume final: {fpv_final}")
            print(f"False negative volume final: {fnv_final}")
