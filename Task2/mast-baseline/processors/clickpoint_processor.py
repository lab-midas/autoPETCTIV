import gc
import json
import numpy as np

import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
from utils.utils import *
from LongiSeg.longiseg.imageio.simpleitk_reader_writer import SimpleITKIO


def _clear_memory() -> None:
    """Helper function to clear CPU and GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

class VOIGenerator:
    """Helper class to create a generator for Volume of Interest extraction."""
    
    def __init__(self, data: List[Dict], extractor: 'VOIExtractor') -> None:
        self.data = data
        self.extractor = extractor
        self.interface = extractor.interface
        
            
    def __iter__(self) -> Iterator[Dict]:
        """Return a generator that yields VOIs."""

        for img_type in ['primary', 'secondary']:
            try:
                clickpoints = self.data.get(f'{img_type}_fu_clickpoints', {})
                img_path = self.data.get(f'{img_type}_fu_image_path') 
                if img_path is None:
                    continue
                img_array, props = SimpleITKIO().read_images([img_path])

                # Reconstruct the SimpleITK Image object from the array and properties
                # img_array has shape (C, D, H, W), SimpleITK expects (D, H, W) or (Z, Y, X)
                # Assuming single channel input C=1
                sitk_image = sitk.GetImageFromArray(img_array[0]) # Use the first channel
                sitk_image.SetSpacing(props['sitk_stuff']['spacing'])
                sitk_image.SetOrigin(props['sitk_stuff']['origin'])
                sitk_image.SetDirection(props['sitk_stuff']['direction'])

                d = self.extractor.depth
                h = self.extractor.height_width
                w = self.extractor.height_width 
                voi_dims_xyz = (w, h, d)

                extractor = VoiExtractor(sitk_image, voi_dims_xyz)
                image_info = extractor.get_image_info()          

                for lesion_id, voxel_coord in clickpoints.items():
                    try:
                        
                        voi = extractor.extract_voi(voxel_coord, coordinate_type='voxel')
                        
                        # Yield VOI data
                        yield {
                            "voi": voi,
                            "lesion_id": lesion_id,
                            "point_idx": int(lesion_id),
                            "file_path": img_path,
                            "image_info": image_info,
                            "props": props,
                            "voxel_coord": voxel_coord # NOTE only for debugging
                        }
                        
                        del voi
                        _clear_memory()
                        
                    except Exception as e:
                        tqdm.write(f"Error extracting VOI for lesion {lesion_id}: {str(e)}")
                
                # Clean up
                _clear_memory()
                    
            except Exception as e:
                tqdm.write(f"Error in VOIGenerator: {str(e)}")


class VOIExtractor:
    """Extracts Volumes of Interest (VOIs) from CT images"""
    
    def __init__(
        self,
        input_root: Path,
        depth: int = 128,
        height_width: int = 256,
    ) -> None:
        """
        Initialize the VOIExtractor.
        
        Args:
            input_root: Root path of the data (/input on GC)
            depth: Depth of VOI in voxels
            height_width: Height and width of VOI in voxels
        """
        self.input_root = Path(input_root)
        self.depth = depth
        self.height_width = height_width
        self.interface = which_interface(input_root)
    
    def _find_images(self):
        """
        Find all image files in subfolders within input_root/images/
        
        Returns:
            Dict mapping subfolder names to image paths
        """
        images_dir = self.input_root / "images"
        image_dict = {}
        
        # Walk through all subdirectories in the images directory
        for subdir in images_dir.glob("*"):
            if subdir.is_dir():
                subfolder_name = subdir.name
                # Find all image files in this subdirectory
                for ext in ["*.mha", "*.nii", "*.nii.gz"]:
                    for img_path in subdir.glob(ext):
                        image_dict[subfolder_name] = img_path
                        break 
        return image_dict

    def _parse_clickpoints_from_json(self, json_paths: List[Path]) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]]]:
        """
        Parse clickpoints (center of gravity) from a list of JSON files.
        
        Args:
            json_paths: List of paths to JSON files containing clickpoints
            can be: primary-baseline, primary-followup, secondary-baseline, secondary-followup

            For example jsons look on: https://grand-challenge.org/components/interfaces/inputs/
            
        Returns:
            Tuple of (primary_bl_points, primary_fu_points, 
                     secondary_bl_points, secondary_fu_points)
            Each element is either None or a dict mapping lesion_id to (x,y,z) coordinates
        """

        def get_clickpoints(files):
            clickpoints = {}
            for bl_file in files:
                with open(bl_file, 'r') as f:
                    data = json.load(f)
                    
                for _, point in enumerate(data.get("points", [])):
                    point_name = point.get("name", "")
                    # Extract lesion_id from point name (format: "Lesion X")
                    lesion_id = int(point_name.split()[-1])
                        
                    # Extract coordinates
                    coords = point.get("point", [])
                    if len(coords) == 3:
                        clickpoints[lesion_id] = tuple(coords)
            return clickpoints if clickpoints else None
        
        try:
            primary_bl_files = [p for p in json_paths if "primary-baseline" in p.name.lower()]
            primary_fu_files = [p for p in json_paths if "primary-followup" in p.name.lower()]
            secondary_bl_files = [p for p in json_paths if "secondary-baseline" in p.name.lower()]
            secondary_fu_files = [p for p in json_paths if "secondary-followup" in p.name.lower()]
            
            # Process all files and return tuple of results
            return (
                get_clickpoints(primary_bl_files),
                get_clickpoints(primary_fu_files),
                get_clickpoints(secondary_bl_files),
                get_clickpoints(secondary_fu_files)
            )
            
        except Exception as e:
            tqdm.write(f"Error parsing JSON files: {str(e)}")
            return None, None, None, None

    def prepare_data(self) -> List[Dict]:
        """
        Prepare the data for loading each follow-up image with its corresponding clickpoints from the csv file.
        
        Returns: a dictionary containing the image paths, and corresponding image clickpoints available in the input folder.
        based on the GC interface, this can mean some of the dict values or None if there were not provided.  

        """        
        # Find all json files and images
        json_files = [path for path in self.input_root.glob("*.json")]
        images = self._find_images()
        primary_bl_clicks, primary_fu_clicks, secondary_bl_clicks, secondary_fu_clicks = self._parse_clickpoints_from_json(json_files)
        return {
                "primary_bl_image_path": images.get('primary-baseline-ct', None),
                "primary_fu_image_path": images.get('primary-followup-ct', None),
                "primary_bl_mask_path": images.get('primary-baseline-ct-tumor-lesion-seg', None),
                "secondary_bl_image_path": images.get('secondary-baseline-ct', None),
                "secondary_fu_image_path": images.get('secondary-followup-ct', None),
                "secondary_bl_mask_path": images.get('secondary-baseline-ct-tumor-lesion-seg', None),
                "primary_bl_clickpoints": primary_bl_clicks,
                "primary_fu_clickpoints": primary_fu_clicks,
                "secondary_bl_clickpoints": secondary_bl_clicks,
                "secondary_fu_clickpoints": secondary_fu_clicks,

            }


    def extract_vois_generator(self, data: Dict) -> Iterator[Dict]:
        """
        Process a dataset containing follow-up images and clickpoints, yielding VOIs one at a time.
        
        Args:
            data: Dictionary containing:
                - primary_bl_image_path: Path to primary baseline CT image
                - primary_fu_image_path: Path to primary follow-up CT image  
                - primary_bl_mask_path: Path to primary baseline tumor segmentation
                - secondary_bl_image_path: Path to secondary baseline CT image
                - secondary_fu_image_path: Path to secondary follow-up CT image
                - secondary_bl_mask_path: Path to secondary baseline tumor segmentation
                - primary_bl_clickpoints: Dict mapping lesion IDs to coordinate tuples
                - primary_fu_clickpoints: Dict mapping lesion IDs to coordinate tuples
                - secondary_bl_clickpoints: Dict mapping lesion IDs to coordinate tuples
                - secondary_fu_clickpoints: Dict mapping lesion IDs to coordinate tuples

        If any of these is not available, value will be NONE

                
        Returns:
            Iterator yielding dictionaries containing:
                - voi: numpy array of shape (D, H, W)
                - metadata: dict with spacing, origin and direction
                - lesion_id: str/int identifier of the lesion
                - point_idx: int, index of the point
                - voi_coords: dict with coordinates in original image
                - original_shape: tuple with shape of original image
                - file_path: str with path to original image
                - voxel_coord: coordinates in voxel space (for debugging)
        """
        return VOIGenerator(data, self) 