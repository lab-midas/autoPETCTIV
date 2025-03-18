import json
import os
import shutil
import subprocess
from pathlib import Path
import SimpleITK
import torch
from utils import simulate_clicks, save_click_heatmaps

class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.gt_nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/labelsTs"
        )
        self.lesion_click_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self, simulate_click=False, save_click_heatmap=True):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        label_mha = os.listdir(os.path.join(self.input_path, "labels/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "labels/", label_mha),
            os.path.join(self.gt_nii_path, "TCIA_001.nii.gz"),
        )

        if simulate_click:
            simulate_clicks(os.path.join(self.gt_nii_path, "TCIA_001.nii.gz"), None,
                            self.lesion_click_path
                            )
        if not simulate_click:
            json_file = next(Path(self.input_path).rglob("*.json"), None)
            if json_file:shutil.copy(json_file,os.path.join(self.lesion_click_path, "TCIA_001_clicks.json"),
                                    )

        if save_click_heatmap: #if save_click_heatmap=False --> original nnUNet setup with two input channels (like the auptoPET III nnunet-baseline)
            click_file = os.listdir(self.lesion_click_path)[0]
            if click_file:
                with open(os.path.join(self.lesion_click_path, click_file), 'r') as f:
                    clicks = json.load(f)
                save_click_heatmaps(clicks, self.nii_path, 
                                    os.path.join(self.gt_nii_path, "TCIA_001.nii.gz"),
                                    )
        print(os.listdir(self.nii_path))

        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """
        print("nnUNet segmentation starting!")
        cproc = subprocess.run(
            f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d 221 -c 3d_fullres -f 0 --disable_tta",
            shell=True,
            check=True,
        )
        print(cproc)
        # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but
        # segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being
        # printed.
        print("Prediction finished")

   
    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
