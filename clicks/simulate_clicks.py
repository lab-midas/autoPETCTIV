'''
This script simulates clicks for AutoPET II (FDG PET-CT), 
AutoPET III (PSMA & FDG-PET/CT), and the Longitudinal CT dataset.

Before running the script, ensure that the labels and images are 
downloaded and converted into NIfTI files (nnunet input format).
'''


import os
import nibabel as nib

import cc3d
import numpy as np
import cupy as cp
from cucim.core.operations import morphology
import json

import argparse
import SimpleITK
from batchgenerators.utilities.file_and_folder_operations import *


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_label", required=True, help="Path to the nifti labels folder")
parser.add_argument("-d", "--dataset_name", required=True, help="autoPET dataset name (modalities)")
parser.add_argument("--json_output", required=True, help="Output path for JSON files containing all clicks")
parser.add_argument("-i_pet", "--input_pet", required=False, help="Path to the nifti pet images", default=None)
parser.add_argument("-o", "--output_heatmap", required=False, help="Output path for containing Gaussian heatmaps for each all clicks", default=None)
parser.add_argument("--center_offset", required=False, help="Random perturbation for center clicks (could be used as data augmentation for training)", default=None)
parser.add_argument("--edge_offset", required=False, help="Random perturbation for edge clicks (could be used as data augmentation for training)", default=None)

args = parser.parse_args()


SEED = 42

def identify_click_region(dataset_name):
    fg = True
    if dataset_name == "FDG-PETCT"  or dataset_name == "PSMA-FDG-PETCT":
        bg = True
        click_budget = 10
    elif dataset_name == "Longitudinal-CT":
        bg = False
        click_budget = 2
    return fg, bg, click_budget

def perturb_click(offset, click, label_im):
    import random
    random_offset = [random.randint(0, int(offset)) for _ in range(3)]
    if label_im[
        int(click[0] + random_offset[0]),
        int(click[1] + random_offset[1]),
        int(click[2] + random_offset[2])
    ]:
        return [
            int(click[0] + random_offset[0]),
            int(click[1] + random_offset[1]),
            int(click[2] + random_offset[2])
        ]
    else:
        # Fallback to original click if perturbed click is invalid
        return [int(click[0]), int(click[1]), int(click[2])]

def simulate_clicks(input_label, input_pet, fg=True, bg=True, center_offset=None, edge_offset=None, click_budget=10):
    np.random.seed(SEED)

    # check if input_label is a file or a numpy array
    if isinstance(input_label, str):
        label_im = nib.load(input_label).get_fdata()
    elif isinstance(input_label, np.ndarray):
        label_im = input_label
    else:
        raise ValueError("input_label must be a file path to a NIfTI file or a numpy array")    
    clicks = {'tumor':[], 'background': []}

    if np.sum(label_im) == 0:
        print("[WARNING] GT is empty, generating background clicks only!")
    else: 
        ##### Tumor Clicks #####
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(click_budget, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # Sample center clicks for 10 (click_budget) random components
        for label in sampled_labels:    
            labeled_mask = connected_components == label
            labeled_mask = cp.array(labeled_mask)
            try:
                # Attempt to compute EDT using GPU
                edt = morphology.distance_transform_edt(labeled_mask)
            except MemoryError:
                from scipy import ndimage

                print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
                edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
                edt = cp.array(edt)
                labeled_mask = cp.array(labeled_mask)

            center = cp.unravel_index(cp.argmax(edt), edt.shape)
            if center_offset is not None:
                center = perturb_click(center_offset, center, label_im)
            clicks['tumor'].append([int(center[0]), int(center[1]), int(center[2])])
            assert label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['tumor'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        while n_clicks < click_budget:
            for label in sampled_labels: 
                labeled_mask = connected_components == label
                labeled_mask = cp.array(labeled_mask)
                try:
                    # Attempt to compute EDT using GPU
                    edt = morphology.distance_transform_edt(labeled_mask)
                except MemoryError:
                    from scipy import ndimage

                    print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
                    edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
                    edt = cp.array(edt)
                    labeled_mask = cp.array(labeled_mask)
                edt_inverted = (cp.max(edt) - edt) * (edt > 0) 
                boundary_elements = (edt_inverted == cp.max(edt_inverted)) * (labeled_mask > 0)
                indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
                boundary_click = indices[np.random.choice(indices.shape[0])]
                if edge_offset is not None:
                    boundary_click = perturb_click(edge_offset, boundary_click, label_im)

                clicks['tumor'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == click_budget:
                    break
    if bg:
        ##### Background Clicks #####
        if input_pet is None:
            input_pet = input_label.replace('labels', 'images').replace('.nii.gz', '_0001.nii.gz') # AutoPET III specific pre-processing
            pet_img = nib.load(input_pet).get_fdata()
        else:
            # check if input_pet is a file or a numpy array
            if isinstance(input_pet, str):
                pet_img = nib.load(input_pet).get_fdata()
            elif isinstance(input_pet, np.ndarray):
                pet_img = input_pet
        non_tumor = pet_img[label_im == 0]
        th = np.percentile(non_tumor, 99.75)

        non_tumor_high_uptake = (pet_img > th) * (label_im == 0)

        connected_components = cc3d.connected_components(non_tumor_high_uptake, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(click_budget, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # Sample center clicks for 10 components (click_budget)
        for label in sampled_labels:  
            labeled_mask = connected_components == label
            labeled_mask = cp.array(labeled_mask)
            try:
                # Attempt to compute EDT using GPU
                edt = morphology.distance_transform_edt(labeled_mask)
            except MemoryError:
                from scipy import ndimage

                print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
                edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
                edt = cp.array(edt)
                labeled_mask = cp.array(labeled_mask)
            center = cp.unravel_index(cp.argmax(edt), edt.shape)
            if center_offset is not None:
                center = perturb_click(center_offset, center, ~label_im.astype(np.uint8))

            clicks['background'].append([int(center[0]), int(center[1]), int(center[2])])
            assert not label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['background'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        while n_clicks < click_budget:
            for label in sampled_labels:  # Skip background label 0
                labeled_mask = connected_components == label
                labeled_mask = cp.array(labeled_mask)
                try:
                    # Attempt to compute EDT using GPU
                    edt = morphology.distance_transform_edt(labeled_mask)
                except MemoryError:
                    from scipy import ndimage

                    print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
                    edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
                    edt = cp.array(edt)
                    labeled_mask = cp.array(labeled_mask)
                edt_inverted = (cp.max(edt) - edt) * (edt > 0)
                boundary_elements = (edt_inverted == cp.max(edt_inverted)) * (labeled_mask == 0)
                indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
                print(indices)
                if len(indices) == 0:
                    indices = cp.array(cp.nonzero(labeled_mask)).T.get()
                boundary_click = indices[np.random.choice(indices.shape[0])]
                if edge_offset is not None:
                    boundary_click = perturb_click(edge_offset, boundary_click, ~label_im.astype(np.uint8))

                clicks['background'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert not label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == click_budget:
                    break
        
    return clicks
    
def generate_gaussian_heatmap(coords, shape, sigma=2.0):
    from scipy.ndimage import gaussian_filter
    """
    Generate a 3D Gaussian heatmap for given coordinates.

    Args:
        coords (list): List of [x, y, z] coordinates.
        shape (tuple): Shape of the output volume.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 3D volume with Gaussian heatmaps at the specified coordinates.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
            heatmap[tuple(coord)] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)    
    return heatmap

def save_click_heatmaps(clicks, output, input_img_path, fg=True, bg=True):
    #input_img_path: is either path to PET (if autoPET II and III dataset are used) or CT if Longitudinal CT is used.
    img = nib.load(input_img_path)
    ref_shape = img.shape
    ref_affine = img.affine
    tumor_coords = clicks['tumor']
    non_tumor_coords = clicks['background']
    
    tumor_heatmap = generate_gaussian_heatmap(tumor_coords, ref_shape, 3)
    non_tumor_heatmap = generate_gaussian_heatmap(non_tumor_coords, ref_shape, 3)

    tumor_nifti = nib.Nifti1Image(tumor_heatmap, ref_affine)
    non_tumor_nifti = nib.Nifti1Image(non_tumor_heatmap, ref_affine)

    if bg and fg:
        nib.save(tumor_nifti, os.path.join(output, f'{input_img_path.split("/")[-1].split("_0001.nii.gz")[0]}_0002.nii.gz')) # foreground clicks
        nib.save(non_tumor_nifti, os.path.join(output, f'{input_img_path.split("/")[-1].split("_0001.nii.gz")[0]}_0003.nii.gz')) # background clicks
    else:
        nib.save(tumor_nifti, os.path.join(output, f'{input_img_path.split("/")[-1].split("_0000.nii.gz")[0]}_0003.nii.gz')) # only foreground clicks (edit this with respect to the channel name e.g _0002, _0004, etc )

def get_click_heatmaps(clicks, ref_shape, fg=True, bg=True):
    tumor_coords = clicks['tumor']
    non_tumor_coords = clicks['background']
    
    tumor_heatmap = generate_gaussian_heatmap(tumor_coords, ref_shape, 3)
    non_tumor_heatmap = generate_gaussian_heatmap(non_tumor_coords, ref_shape, 3)

    return tumor_heatmap, non_tumor_heatmap

def process_image(input_label, input_pet, dataset_name, click_format='JSON', center_offset=None, edge_offset=None, click_budget=None):
    # input_label/pet: Data can be a path to a NIfTI file or a numpy array
    # dataset_name:    is used to determine the click regions for the specific database
    # click_format:    the output click format [JSON | JSON_GC | heatmap]
    # center_offset:   random perturbation for center clicks (could be used as data augmentation for training)
    # edge_offset:     random perturbation for edge clicks (could be used as data augmentation for training)
    # click_budget:    number of clicks to be simulated
    fg, bg, click_budget_ = identify_click_region(dataset_name)
    if click_budget is None:
        click_budget = click_budget_
    clicks = simulate_clicks(input_label, input_pet, fg, bg, center_offset, edge_offset, click_budget)
    if click_format == 'JSON':
        return clicks
    elif click_format == 'JSON_GC':
        return clicks_to_gc_format(clicks)
    elif click_format == 'heatmap':
        if isinstance(input_label, str):
            # get shape of array in nifti file
            input_label_shape = nib.load(input_label).get_fdata().shape
        elif isinstance(input_label, np.ndarray):
            input_label_shape = input_label.shape
        return get_click_heatmaps(clicks, input_label_shape)

# Loop over the dataset and simulate the clicks
def process_images(input_folder, json_folder, output, save_heatmap=True, center_offset=None, edge_offset=None):
    
    for filename in os.listdir(input_folder):

        if not isfile(os.path.join(json_folder,filename.replace('.nii.gz', '_clicks.json').split('/')[-1])):
            fg, bg, click_budget = identify_click_region(args.dataset_name)
            clicks = simulate_clicks(os.path.join(input_folder, filename), None, fg, bg, center_offset, edge_offset, click_budget)

            with open(os.path.join(json_folder,filename.replace('.nii.gz', '_clicks.json').split('/')[-1]), 'w') as f:
                json.dump(clicks, f)

            if save_heatmap:
                save_click_heatmaps(clicks, output, os.path.join(input_folder, filename), fg, bg)
        else:
            continue

# Convert json to GC format:
def clicks_to_gc_format(input_clicks, gc_json_path=None):
    # check if the input is a path to a file
    if os.path.isdir(input_clicks):
        swfast_json_path = input_clicks
        assert os.path.exists(swfast_json_path)
        with open(swfast_json_path, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = input_clicks

    fg_points = json_data.get('tumor', [])
    bg_points = json_data.get('background', [])
    gc_dict = {  
        "version": {"major": 1, "minor": 0},  
        "type": "Multiple points",  
        "points": []
    }
    for fg_point in fg_points:
        gc_dict['points'].append({'point': fg_point, 'name': 'tumor'})
    for bg_point in bg_points:
        gc_dict['points'].append({'point': bg_point, 'name': 'background'})

    # Save the GC format JSON
    if gc_json_path is not None:
        with open(gc_json_path, 'w') as f_gc:
            json.dump(gc_dict, f_gc)
        print(f'Finished converting {swfast_json_path} to {gc_json_path} in the GC format!')
    else:
        return gc_dict
    

if __name__ == "__main__":
    # Example: python utils.py -i path/to/labels -d "PSMA-FDG-PET-CT" -o path/to/output --json_output path/to/json_output
    print("START")
    input_folder = args.input_label 
    json_folder = args.json_output
    output = args.output_heatmap
    save_heatmap = args.output_heatmap is not None
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(output, exist_ok=True)
    process_images(input_folder, json_folder, output, save_heatmap, args.center_offset, args.edge_offset)
    print("Done")
    print("Convert to grand-challenge json format ")
    gc_json_path = os.path.join(json_folder, 'gc')
    clicks_to_gc_format(json_folder, gc_json_path)
    