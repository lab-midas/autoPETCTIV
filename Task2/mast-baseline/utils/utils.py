from pathlib import Path
import SimpleITK as sitk


def which_interface(path: Path):
    """
    Determine the Grand-Challenge interface that has been used. 
    Based on the interface we know which files to expect. 

    On GC your algorithm is called once for each sample, this means it does
    not loop over a dataset, instead one set of data (one sample) is provided. 
    """
    interfaces = {
                    "interface 1": [
                        "images/primary-baseline-ct",
                        "images/primary-followup-ct",
                        "images/primary-baseline-ct-tumor-lesion-seg",
                        "primary-baseline-lesion-clicks.json",
                        "primary-followup-lesion-clicks.json"
                    ],
                    
                    "interface 2": [
                        "images/primary-baseline-ct",
                        "images/secondary-baseline-ct",
                        "images/primary-followup-ct",
                        "images/secondary-followup-ct",
                        "images/primary-baseline-ct-tumor-lesion-seg",
                        "images/secondary-baseline-ct-tumor-lesion-seg",
                        "primary-baseline-lesion-clicks.json",
                        "secondary-baseline-lesion-clicks.json",
                        "primary-followup-lesion-clicks.json",
                        "secondary-followup-lesion-clicks.json"
                    ],
                    
                    "interface 3": [
                        "images/primary-baseline-ct",
                        "images/secondary-baseline-ct",
                        "images/primary-followup-ct",
                        "images/primary-baseline-ct-tumor-lesion-seg",
                        "images/secondary-baseline-ct-tumor-lesion-seg",
                        "primary-baseline-lesion-clicks.json",
                        "secondary-baseline-lesion-clicks.json",
                        "primary-followup-lesion-clicks.json"
                    ],
                    
                    "interface 4": [
                        "images/primary-baseline-ct",
                        "images/primary-followup-ct",
                        "images/secondary-followup-ct",
                        "images/primary-baseline-ct-tumor-lesion-seg",
                        "primary-baseline-lesion-clicks.json",
                        "primary-followup-lesion-clicks.json",
                        "secondary-followup-lesion-clicks.json"
                    ]
                }
    
    available_components = (
        [f"images/{p.name}" for p in (path / "images").iterdir() if p.is_dir()] +
        [p.name for p in path.glob("*.json")]
    )
    
    return next((iface for iface, comps in interfaces.items() if set(available_components) == set(comps)), None)


class VoiExtractor:
    """Extracts a Volume of Interest (VOI) from a SimpleITK image based on a physical point."""
    def __init__(self, sitk_image: sitk.Image, voi_size_xyz: tuple):
        if not isinstance(sitk_image, sitk.Image):
            raise TypeError("sitk_image must be a SimpleITK.Image")
        if not (isinstance(voi_size_xyz, (tuple, list)) and len(voi_size_xyz) == 3):
             raise ValueError("voi_size_xyz must be a tuple or list of 3 integers (x, y, z)")

        self.sitk_image = sitk_image
        self.voi_size_xyz = voi_size_xyz # SimpleITK order: (x, y, z)
        self.img_size_xyz = sitk_image.GetSize()

    def get_image_info(self) -> dict:
        """Returns a dictionary containing essential image information."""
        return {
            "size": self.sitk_image.GetSize(),
            "origin": self.sitk_image.GetOrigin(),
            "spacing": self.sitk_image.GetSpacing(),
            "direction": self.sitk_image.GetDirection(),
            "pixel_id": self.sitk_image.GetPixelIDValue() # Store the ID for potential later casting
        }


    def extract_voi(self, clickpoint) -> sitk.Image:
            """
            Extracts a VOI of exact size centered around a click point with correct spatial metadata preserved.
            Ensures the clickpoint is at the exact center of the VOI, regardless of image boundaries.

            Args:
                sitk_image: The SimpleITK image to extract VOI from
                clickpoint: A list or tuple representing the clickpoint coordinates [x, y, z]

            Returns:
                voi (sitk.Image): The extracted VOI with preserved spatial position
            """
            
            center_idx = [int(round(c)) for c in clickpoint]
            
            # Calculate the reference image for our VOI
            # This creates a blank image of the desired size, properly centered around the clickpoint
            reference_image = sitk.Image(self.voi_size_xyz, self.sitk_image.GetPixelID())
            reference_image.SetSpacing(self.sitk_image.GetSpacing())
            reference_image.SetDirection(self.sitk_image.GetDirection())
            
            # Calculate the origin so that the clickpoint is at the center of the VOI
            # Half size in physical coordinates
            half_size_physical = [(self.voi_size_xyz[i] // 2) * self.sitk_image.GetSpacing()[i] for i in range(3)]
            
            # Physical coordinates of the clickpoint
            click_physical = self.sitk_image.TransformIndexToPhysicalPoint(center_idx)
            
            # Origin is clickpoint minus half the VOI size in each dimension
            origin = [click_physical[i] - half_size_physical[i] for i in range(3)]
            reference_image.SetOrigin(origin)
            
            # Use SimpleITK's Resample to create the VOI
            # This handles edge cases and padding automatically
            default_value = float(0)
            
            # Identity transform - we're not changing the coordinate system
            transform = sitk.Transform(3, sitk.sitkIdentity)
            
            # Resample original image into the reference image space
            voi = sitk.Resample(self.sitk_image, reference_image, transform, sitk.sitkLinear, default_value)
                    
            return voi


class SegmentationMerger:
    """Merges segmentation VOIs onto a full image grid incrementally using Resample + NaryMaximum."""
    def __init__(self, reference_info: sitk.Image | dict, pixel_type: int = sitk.sitkUInt8):
        if isinstance(reference_info, sitk.Image):
            self.size = reference_info.GetSize()
            self.origin = reference_info.GetOrigin()
            self.spacing = reference_info.GetSpacing()
            self.direction = reference_info.GetDirection()

        elif isinstance(reference_info, dict):
            required_keys = {"size", "origin", "spacing", "direction"}
            if not required_keys.issubset(reference_info.keys()):
                raise ValueError(f"reference_info dict missing required keys: {required_keys - set(reference_info.keys())}")
            self.size = reference_info["size"]
            self.origin = reference_info["origin"]
            self.spacing = reference_info["spacing"]
            self.direction = reference_info["direction"]
        else:
            raise TypeError("reference_info must be a SimpleITK.Image or a dictionary")

        self.pixel_type = pixel_type
        self.default_value = 0 # Background value for resampling/initial image
        self.merged_segmentation = None # Stores the incrementally merged segmentation

    def add_voi_segmentation(self, seg_voi: sitk.Image):
        """Resamples a segmentation VOI, merges it with the current result, and updates the merged segmentation."""
        if not isinstance(seg_voi, sitk.Image):
            raise TypeError("seg_voi must be a SimpleITK.Image")

        # Ensure consistent pixel type before resampling
        seg_voi_casted = sitk.Cast(seg_voi, self.pixel_type)

        # Resample the VOI to the full reference grid
        seg_full = sitk.Resample(
            seg_voi_casted,
            self.size,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            self.origin,
            self.spacing,
            self.direction,
            self.default_value,
            self.pixel_type
        )

        # Incrementally merge the new segmentation
        if self.merged_segmentation is None:
            self.merged_segmentation = seg_full
        else:
            # Update the merged segmentation by taking the maximum with the new one
            self.merged_segmentation = sitk.NaryMaximum(self.merged_segmentation, seg_full)

    def get_merged_segmentation(self) -> sitk.Image:
        """Returns the final merged segmentation, or an empty image if no VOIs were added."""
        if self.merged_segmentation is None:
            # Create an empty image with the reference geometry if no VOIs were processed
            print("Warning: No VOIs were added to the merger. Returning an empty segmentation.")
            merged = sitk.Image(self.size, self.pixel_type)
            merged.SetSpacing(self.spacing)
            merged.SetOrigin(self.origin)
            merged.SetDirection(self.direction)
            return merged
        else:
            return self.merged_segmentation

    def reset(self):
        """Clears the merged segmentation state."""
        self.merged_segmentation = None

    def write_merged_and_reset(self, output_filename: str, use_compression: bool = True):
        """Writes the final merged segmentation to a file and resets the merger."""
        # Get the final merged image (handles the case of no VOIs)
        merged_image = self.get_merged_segmentation()

        print(f"Writing final merged segmentation...")
        sitk.WriteImage(merged_image, output_filename, use_compression)
        print(f"Saved merged segmentation to {output_filename}")
        self.reset()
