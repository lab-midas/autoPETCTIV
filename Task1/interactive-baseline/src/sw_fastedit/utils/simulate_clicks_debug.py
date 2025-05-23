import nibabel as nib
import argparse
import cc3d
import numpy as np
import cupy as cp
from cucim.core.operations import morphology
import json
import os

SEED = 42
np.random.seed(SEED)
cp.random.seed(SEED)

# For click visualization 
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

def save_click_heatmaps(clicks, ref_shape, ref_affine, debug_output):
    tumor_coords = clicks['tumor']
    non_tumor_coords = clicks['background']

    tumor_heatmap = generate_gaussian_heatmap(tumor_coords, ref_shape, 3)
    non_tumor_heatmap = generate_gaussian_heatmap(non_tumor_coords, ref_shape, 3)

    tumor_nifti = nib.Nifti1Image(tumor_heatmap, ref_affine)
    non_tumor_nifti = nib.Nifti1Image(non_tumor_heatmap, ref_affine)

    nib.save(tumor_nifti, os.path.join(debug_output, 'fg_heatmaps.nii.gz'))
    nib.save(non_tumor_nifti, os.path.join(debug_output, 'bg_heatmaps.nii.gz'))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_label", required=True, help="Path to the nifti label")
parser.add_argument("-i_pet", "--input_pet", required=False, help="Path to the nifti pet image")
parser.add_argument("--debug_output", required=False, help="Output path for click visualization for debugging")
parser.add_argument("--json_output", required=True, help="Output path for JSON containing all clicks")
args = parser.parse_args()

label_im = nib.load(args.input_label)
ref_shape = label_im.shape
ref_affine = label_im.affine
label_im = label_im.get_fdata()
clicks = {'tumor':[], 'background': []}

if np.sum(label_im) == 0:
    print("[WARNING] GT is empty, generating background clicks only!")
else: 
    ##### Tumor Clicks #####
    connected_components = cc3d.connected_components(label_im, connectivity=26)
    unique_labels = np.unique(connected_components)[1:] # Skip background label 0
    size = min(10, len(unique_labels))
    sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

    # Sample center clicks for 10 random components
    for label in sampled_labels:  
        labeled_mask = connected_components == label
        labeled_mask = cp.array(labeled_mask)
        edt = morphology.distance_transform_edt(labeled_mask)
        center = cp.unravel_index(cp.argmax(edt), edt.shape)
        clicks['tumor'].append([int(center[0]), int(center[1]), int(center[2])])
        assert label_im[int(center[0]), int(center[1]), int(center[2])]
    n_clicks = len(clicks['tumor'])

    # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
    while n_clicks < 10:
        for label in sampled_labels: 
            labeled_mask = connected_components == label
            labeled_mask = cp.array(labeled_mask)
            edt = morphology.distance_transform_edt(labeled_mask)
            edt_inverted = (cp.max(edt) - edt) * (edt > 0)
            boundary_elements = (edt_inverted == cp.max(edt_inverted))
            indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
            boundary_click = indices[np.random.choice(indices.shape[0])]
            clicks['tumor'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
            assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
            n_clicks += 1
            if n_clicks == 10:
                break

##### Background Clicks #####
in_pet = args.input_pet 
if in_pet is None:
    in_pet = args.input_label.replace('labels', 'images').replace('.nii.gz', '_0001.nii.gz') # AutoPET III specific pre-processing
pet_img = nib.load(args.input_pet).get_fdata()
non_tumor = pet_img[label_im == 0]
th = np.percentile(non_tumor, 99.75)

non_tumor_high_uptake = (pet_img > th) * (label_im == 0)

connected_components = cc3d.connected_components(non_tumor_high_uptake, connectivity=26)
unique_labels = np.unique(connected_components)[1:] # Skip background label 0
size = min(10, len(unique_labels))
sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

# Sample center clicks for 10 components
for label in sampled_labels:  
    labeled_mask = connected_components == label
    labeled_mask = cp.array(labeled_mask)
    edt = morphology.distance_transform_edt(labeled_mask)
    center = cp.unravel_index(cp.argmax(edt), edt.shape)
    clicks['background'].append([int(center[0]), int(center[1]), int(center[2])])
    assert not label_im[int(center[0]), int(center[1]), int(center[2])]
n_clicks = len(clicks['background'])

# Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
while n_clicks < 10:
    for label in sampled_labels:  # Skip background label 0
        labeled_mask = connected_components == label
        labeled_mask = cp.array(labeled_mask)
        edt = morphology.distance_transform_edt(labeled_mask)
        edt_inverted = (cp.max(edt) - edt) * (edt > 0)
        boundary_elements = (edt_inverted == cp.max(edt_inverted))
        indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
        boundary_click = indices[np.random.choice(indices.shape[0])]
        clicks['background'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
        assert not label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
        n_clicks += 1
        if n_clicks == 10:
            break

if args.debug_output is not None:
    os.path.makedirs(args.debug_output, exists_ok = True)
    save_click_heatmaps(clicks, ref_shape, ref_affine, args.debug_output)

with open(os.path.join(args.json_output, args.input_label.replace('.nii.gz', '_clicks.json').split('/')[-1]), 'w') as f:
    json.dump(clicks, f)

