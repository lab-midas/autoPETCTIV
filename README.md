   ![](https://public.grand-challenge-user-content.org/b/807/autoPET-2025-banner-1_o53thn0.x20.jpeg)
   # autoPET/CT IV challenge
   Repository for code associated with autoPET/CT IV machine learning challenge: <br/> 
   [autopet-iv.grand-challenge.org](https://autopet-iv.grand-challenge.org/autopet-iv/) <br/>
   [www.autopet.org](http://www.autopet.org)
   
   ## Datasets				
   If you use the data associated to this challenge, please cite: <br/>
   
   ```
   Gatidis S, Kuestner T. A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) 
   [Dataset]. The Cancer Imaging Archive, 2022. DOI: 10.7937/gkr0-xv29

   Gatidis S., Hepp T., Früh M., La Fougère C., Nikolaou K., Pfannenberg, C., Schölkopf B., Küstner T., 
   Cyran C., Rubin D. A whole-body FDG-PET/CT Dataset with manually annotated Tumor Lesions. Sci Data 9, 601 (2022). 
   https://doi.org/10.1038/s41597-022-01718-3
   ```
   
   and
   <br/>
   
   ```
   Jeblick, K., et al. A whole-body PSMA-PET/CT dataset with manually annotated tumor lesions 
   (PSMA-PET-CT-Lesions) (Version 1) [Dataset]. The Cancer Imaging Archive, 2024.
   DOI: 10.7937/r7ep-3x37
   ```

### FDG PET/CT
DICOM: <a href="https://doi.org/10.7937/gkr0-xv29"><img src="https://img.shields.io/badge/DOI-10.7937%2Fgkr0--xv29-blue"></a>
<br/>
NiFTI: <a href="https://doi.org/10.57754/FDAT.wf9fy-txq84"><img src="https://img.shields.io/badge/DOI-10.57754%2FFDAT.wf9fy--txq84-blue"></a>
    
### PSMA PET/CT
DICOM:  <a href="https://doi.org/10.7937/r7ep-3x37 "><img src="https://img.shields.io/badge/DOI-10.7937%2Fr7ep--3x37-blue"></a>
<br/>
NiFTI: <a href="https://doi.org/10.57754/FDAT.6gjsg-zcg93"><img src="https://img.shields.io/badge/DOI-10.57754%2FFDAT.6gjsg--zcg93-blue"></a>

### Longitudinal CT
Database release soon

## Tasks
The tasks will handle lesion segmentation for whole-body hybrid imaging for a single-staging PET/CT (Task 1) or a longitudinal CT screening (Task 2). All public available data and (pre-trained) models (including foundation models) are allowed. Time points and/or imaging modality inputs can be treated individually or jointly. Participants can either develop a

- novel method and/or
- focus on data-centric approaches (using the provided baselines), i.e. development on pre- and post-processing pipelines.

![](https://public.grand-challenge-user-content.org/i/2025/03/23/53b754cd-e1d8-4b10-9bb7-f253acf14cc6.png)

### Task 1: Single-staging whole-body PET/CT
   
In Task 1, we study an interactive human-in-the-loop segmentation scenario and investigate the impact of varying degrees of label information (none to multiple clicks per lesion and background) on the segmentation performance in a multi-tracer and multi-center imaging setting - similar to the previous iterations of the autoPET challenge. The specific challenge in automated segmentation of lesions in PET/CT is to avoid false-positive segmentation of anatomical structures that have physiologically high uptake while capturing all tumor lesions. This task is particularly challenging in a multitracer setting since the physiological uptake partly differs for different tracers: e.g. brain, kidney, heart for FDG and e.g. liver, kidney, spleen, submandibular for PSMA.

We will study the behaviour over 11 interactive segmentation steps. In each step, an additional standardized and pre-simulated tumor (foreground) and background click, represented as a set of 3D coordinates, will be provided alongside the input image. This process will progress incrementally from 0 clicks (1st step) to the full allocation of 10 tumor and 10 background clicks per image (11th step). 

#### Baselines
1. [interactive-baseline](Task1/interactive-baseline/)
2. [nnunet-baseline](Task1/nnunet-baseline/)

### Task 2: Longitudinal CT screening
In Task 2, we explore the segmentation performance to detect lesions in a follow-up scan if provided with a lesion segmentation in the baseline scan together with or without clicks of the lesions in the follow-up scan. We investigate the segmentation performance in a novel provided longitudinal CT database of >300 cases - similar to autoPET I. The specific challenge lies in the correct identification of a lesion between baseline and follow-up scan under therapy. A tumor can change shape (progression or regression), disappear (complete response) or new lesions can appear (metastasis). In addition, in some cases the differentiation between malignant and benign tissue can be difficult.

We will study the behaviour over 2 interactive segmentation steps. In the first step, only the CT images of the baseline and follow-up scan are provided together with the baseline segmentation mask (for tumor localization). For the first step no tumor click is given, i.e. empty lesion click file. In the second step, the CT images (baseline and follow-up), the baseline segmentation mask and an additional pre-simulated tumor click in the follow-up CT scan are provided. We encourage the usage of registration algorithms to align the baseline and follow-up CT scan, i.e. to transform the baseline lesion (from baseline segmentation) into the follow-up CT scan and target region (click), but this is not mandatory.

#### Baselines
TBD

## Click simulation
To simulate clicks on new data and replicate the exact same conditions as our provided clicks for autoPET/CT IV, you could use the `clicks/simulate_clicks.py` script by first installing all dependencies as in 2.2 Inference with Python Script.

The script has the following arguments:

    simulate_clicks.py --input_label .. --dataset_name .. --json_output .. [--input_pet .. --output_heatmap .. --center_offset .. --edge_offset ..]

    --input_label: Path to the NIfTI label folder
    --dataset_name: Name of database [FDG-PETCT | PSMA-FDG-PETCT | Longitudinal-CT]
    --json_output: Output path for JSON files containing all clicks. The subfolder "gc" will contain the JSON files in grand-challenge format
    --input_pet: Path to the NIfTI folder containing the PET (SUV) images
    --output_heatmap: Output path for Gaussian Heatmaps of the clicks. The output will be two nifti files for foreground and background of Gaussian Heatmaps of the clicks.
    --center_offset: Maximum offset for each dimension (XYZ) to perturb each center click during simulation
    --edge_offset: Maximum offset for each dimension (XYZ) to perturb each boundary click during simulation

An example script to run for the entire autoPET FDG+PSMA database:

```bash
python clicks/simulate_clicks.py \
    --input_label test/input/autoPETIII/labelsTs/ \
    --dataset_name PSMA-FDG_PETCT \
    --json_output test/output/autoPETIII/JSON_clicks/ \
    --output_heatmap test/output/autoPETIII/Ts_clicks/ \
    --input_pet test/input/autoPETIII/imagesTs/ \
    --center_offset 2 \
    --edge_offset 2
```

or for a single image of the autoPET FDG+PSMA data (for example used as data augmentation during training):

```python
    from clicks.simulate_clicks import *
    clicks = process_image(...)
```

with the interface:
```python
 def process_image(input_label, input_pet, dataset_name, click_format='JSON', center_offset=None, edge_offset=None, click_budget=None):
    # input_label/pet: Data can be a path to a NIfTI file or a numpy array
    # dataset_name:    is used to determine the click regions for the specific database
    # click_format:    the output click format [JSON | JSON_GC | heatmap]
    # center_offset:   random perturbation for center clicks (could be used as data augmentation for training)
    # edge_offset:     random perturbation for edge clicks (could be used as data augmentation for training)
    # click_budget:    number of clicks to be simulated
```


### Task 1: Single-staging whole-body PET/CT
Pre-simulated clicks for the entire FDG+PSMA training database can be found in `simulate_clicks/FDG_PSMA_PETCT_pre-simulated-clicks.zip`.

### Task 2: Longitudinal CT
Pre-simulated clicks for the entire longitudinal CT training database can be found in `simulate_clicks/...`

## Evaluation metrics
Evaluation code for the autoPET/CT IV challenge according to [here](https://autopet-iv.grand-challenge.org/evaluation-and-ranking/) can be found in: `evaluation_metrics/evaluation_metrics.py`

## References
Challenge: TBD<br/>
   
   
   
   
   
   
   
