from pathlib import Path
import warnings
import os
import yaml
from box import Box

from processors.clickpoint_processor import VOIExtractor
from processors.inference_processor import InferenceProcessor

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


class MASTBaseline:
    """  
    MASTBaseline is the top level class that manages VOI (Volume of Interest) 
    extraction and inference processing.
    """
    

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the MAST baseline processor using configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        self.voi_extractor = VOIExtractor(
            input_root=Path(self.config.grand_challenge.input_path),
            depth=self.config.voi.depth,
            height_width=self.config.voi.height_width,
        )
        
        # Initialize processors as None - they'll be created on demand
        self.inference_processor = None
    
    def _init_inference_processor(self):
        """Initialize the inference processor if it hasn't been already."""
        if self.inference_processor is None:
            print("Initializing inference processor...")
            self.inference_processor = InferenceProcessor(
                model_folder=Path(self.config.paths.model_folder),
                output_root=Path(self.config.grand_challenge.output_path),
            )
        return self.inference_processor
    
    @staticmethod
    def load_config(config_path: str) -> Box:
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Box: Configuration object with dot notation access

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return Box(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def prepare_data(self) -> list:
        """
        Prepare a data object that holds all information required by the processors,
        such as file paths, clickpoints etc.

        Returns:
            list: List of dictionaries containing .
        """
        return self.voi_extractor.prepare_data()
    
    def process_cases(self, data: list) -> None:
        """
        Process the cases through VOI extraction and inference.

        Args:
            data (list): List of dictionaries containing follow-up image paths and clickpoints
        """
        # Initialize the inference processor
        inference_processor = self._init_inference_processor()
        if inference_processor is None:
            print("Inference processor initialization failed. Check your configuration.")
            return
            
        print("\n=== Starting MAST Lesion Segmentation ===\n")
        voi_generator = self.voi_extractor.extract_vois_generator(data)
        
        # Process the VOIs
        inference_processor.process_vois(voi_generator)
        print("\n=== Processing Complete ===\n")
    
    
    def run(self) -> None:
        """
        Execute the MAST baseline pipeline.
        """
        data = self.prepare_data()
        self.process_cases(data)



def main():
    try:
        processor = MASTBaseline()
        processor.run()
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error: {e}")
        print("Please ensure config.yaml exists and is properly formatted.")
        exit(1)


if __name__ == "__main__":
    main()



