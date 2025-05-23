# autoPET/CT IV challenge - Task 1: Single-staging whole-body PET/CT
Repository for code associated with autoPETIV machine learning challenge: <br/> 
[autopet-iv.grand-challenge.org](https://autopet-iv.grand-challenge.org/autopet-iv/) 


## Grand Challenge Algorithm Context

This codebase is designed to run as a [Grand Challenge Algorithm](https://grand-challenge.org/documentation/create-your-own-algorithm/). This means it's packaged as a Docker container that Grand Challenge executes.

Key points for running on Grand Challenge:
1.  **Input/Output:** The algorithm expects input data in `/input/` and writes results to `/output/`, following the standard Grand Challenge structure. The specific input files expected are:
    ```
    /input/
    ├── images/
    │   ├── ct/                               (MHA File)
    │   ├── pet/                              (MHA File)
    ├── lesion-clicks.json                    (JSON File)
    ```
    The algorithm container reads the necessary files from `/input` based on the specific task it's executing and writes the resulting segmentation mask (`<uuid>.mha`) to the following subdirectory within `/output/`:
    ```
    /output/images/tumor-lesion-segmentation/
    ```

2.  **Single Case Execution:** Grand Challenge runs the algorithm container *once per case*. The code in `process.py` handles a single execution based on the available inputs.

## Baseline Implementation Details

### interactive-baseline
Baseline model 1 for lesion segmentation: The SW-FastEdit model (https://github.com/Zrrr1997/SW-FastEdit) was used for training the baseline. 
The model input channels were structured as follows:
* FDG-PET (SUV) volumes from the autoPET II challenge, 
* Simulated JSON files generated for each image where each `.json` file contains 10 tumor and 10 background clicks with `(x, y, z)` coordinates (example below):
```
{"tumor": [[109, 103, 353], [99, 92, 178], [69, 98, 133], [120, 70, 360], [90, 104, 340], [71, 96, 344], [93, 70, 284], [100, 64, 312], [110, 96, 375], [93, 80, 187]], "background": [[84, 84, 264], [100, 116, 253], [80, 96, 277], [93, 87, 200], [71, 98, 263], [143, 99, 351], [68, 94, 272], [85, 90, 272], [104, 99, 388], [103, 110, 272]]}
```
    - "tumor": represent lesions (foreground) clicks
    - "background": represents background clicks
For each corresponding image, 3D Gaussian heatmaps were generated for both the foreground (tumor) and background clicks. These heatmaps were then converted to NIfTI format.
The model  was trained using three input channels.

### nnunet-baseline
Baseline model 2 for lesion segmentation: The nnUNet framework (https://github.com/MIC-DKFZ/nnUNet) 
was used for training the baseline model using the 3D fullres configuration on GPU with 32 GB VRAM. 
The model input channels were structured as follows:
* PET (SUV) volumes from autoPET III challenge (FDG & PSMA)
* Resampled CT volumes from autoPET III challenge (FDG & PSMA)
* 3D Gaussian heatmap for the foreground clicks
* 3D Gaussian heatmap for the background clicks

The model was trained using four input channels over 250k steps; the initial learning rate to 1e-4. Training was performed with 1 sample (fold 0). Inference runs without test-time augmentation because of the challenge time limit of 15 minutes.






