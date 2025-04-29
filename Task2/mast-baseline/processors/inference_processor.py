from pathlib import Path
from typing import Dict, Iterator

import torch
import warnings
import numpy as np
import SimpleITK as sitk
from LongiSeg.longiseg.inference.predict_from_raw_data import nnUNetPredictor
from utils.utils import SegmentationMerger
from tqdm import tqdm
import uuid

def _setup_predictor(model_folder: Path, device: torch.device) -> nnUNetPredictor:
    """Initialize and setup the nnUNet predictor with given parameters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        
        print(f"Loading model from: {model_folder.name}")
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds="all",
            checkpoint_name="checkpoint_best.pth"
        )
        print("Model loaded successfully")
        return predictor

class InferenceProcessor:
    """
    Infer on the VOIs, in our case with the ULS23 baseline model, that does not consider baseline information.
    NOTE: nnUnet does not support inference batching, so we process VOIs one at a time.
    """
    
    def __init__(
        self,
        model_folder: Path,
        output_root: Path,
    ) -> None:
        """
        Initialize the InferenceProcessor.
        
        Args:
            model_folder: Path to the trained nnUNet model folder
            output_root: Path where prediction masks will be saved
            debug: If True, save intermediate VOIs and masks to a debug folder.
        """
        self.output_root = Path(output_root)

        
        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.predictor = _setup_predictor(model_folder, self.device)
        self.merger = None        
        self.current_case_key = None
    
    
    def _save_current_case_mask(self) -> None:
        """Save the current case mask and free memory."""
        # NOTE !START IMPORTANT! 
        # output_subfolder need to be set based on if the current mask is for the primary
        # or secondary scan, is currently static. Change this for your own implementation.
        # NOTE !END IMPORTANT!

        output_subfolder = 'secondary-followup-ct-tumor-lesion-seg' if self.current_case_key and 'secondary' in str(self.current_case_key) else 'primary-followup-ct-tumor-lesion-seg'

        allowed_subfolders = [
            'primary-followup-ct-tumor-lesion-seg',
            'secondary-followup-ct-tumor-lesion-seg'
        ]
        if output_subfolder not in allowed_subfolders:
            raise ValueError(f"Invalid output subfolder: {output_subfolder}. "
                             f"Must be one of {allowed_subfolders}")
        
       
        try:
            output_dir = self.output_root / output_subfolder
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{uuid.uuid4()}.mha"
            
            self.merger.write_merged_and_reset(output_file)

        except Exception as e:
            tqdm.write(f"Error saving merged mask for case {self.current_case_key}: {str(e)}")
    
    def _process_and_merge_voi(self, item: Dict) -> None:
        """Process a single VOI and merge it into the case mask."""
        try:
            voi = item['voi']
            lesion_id = int(item['lesion_id'])
            case_time_key = item['file_path']
            image_info = item['image_info']
            props = item['props']

            if self.current_case_key == None and self.merger == None:
                self.merger = SegmentationMerger(image_info, pixel_type=sitk.sitkUInt8)

            # If the case key has changed (or this is the first case), handle the transition
            if case_time_key != self.current_case_key:
                # If there was a previous case being processed, save its results
                if self.current_case_key is not None:
                    self._save_current_case_mask()

                # Initialize/reset for the new case
                self.current_case_key = case_time_key
                self.merger = SegmentationMerger(image_info, pixel_type=sitk.sitkUInt8)

            print(f'CASE KEY: {case_time_key}')
            print(f'{item["voxel_coord"]}')
            
            # Predict Segmentation
            voi_array = sitk.GetArrayFromImage(voi)
            voi_input_array = voi_array[None, ...].astype(np.float32)
            del voi_array

            # Predict using the VOI array and props from the initial read
            predicted_seg_array = self.predictor.predict_single_npy_array(voi_input_array, props, None, None, False)
            del voi_input_array

            # Convert prediction back to SimpleITK image
            seg_sitk = sitk.GetImageFromArray(predicted_seg_array.squeeze().astype(np.uint8))
            del predicted_seg_array
            
            seg_sitk.CopyInformation(voi)

            del voi
            
            # Add VOI segmentation to the merger tool
            self.merger.add_voi_segmentation(seg_sitk)
            del seg_sitk
            
            # Free memory - keep explicit cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            # Print more detailed error information
            print(f"Error processing VOI, lesion {lesion_id}: {str(e)}")
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_vois(self, vois_data: Iterator[Dict]) -> None:
        """
        Process VOIs using nnUNetPredictor.
        Each VOI is processed individually and immediately merged into its case's mask.
        Only one case's mask is kept in memory at a time.
        
        Args:
            vois_data: Generator of dictionaries containing:
                - voi: SimpleITK.Image object representing the VOI.
                - lesion_id: str or int identifier for the lesion.
                - point_idx: int, index derived from lesion_id.
                - file_path: str, path to the original image file.
                - image_info: dict containing spatial metadata (size, origin, spacing, direction, pixel_id) of the original image.
                - props: dict, properties dictionary from SimpleITKIO needed by nnUNet.
                - voxel_coord: tuple/list, voxel coordinates of the click point (for debugging).
        """
        try:
            # Process VOIs as they are generated
            for voi in vois_data:
                self._process_and_merge_voi(voi)
        finally:
            # Save the last case mask
            self._save_current_case_mask()