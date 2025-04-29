# autoPET/CT IV challenge - Task 2: Longitudinal CT
Repository for code associated with autoPETIV machine learning challenge: <br/> 
[autopet-iv.grand-challenge.org](https://autopet-iv.grand-challenge.org/autopet-iv/) 


## mast-baseline
It is based on the [ULS23 baseline](https://github.com/DIAGNijmegen/ULS23/tree/main/baseline_model) and utilizes the [LongiSeg](https://github.com/MIC-DKFZ/LongiSeg) framework (an extension of nnUNet for longitudinal data).

## Grand Challenge Algorithm Context

This codebase is designed to run as a [Grand Challenge Algorithm](https://grand-challenge.org/documentation/create-your-own-algorithm/). This means it's packaged as a Docker container that Grand Challenge executes.

Key points for running on Grand Challenge:

1.  **Input/Output:** The algorithm expects input data in `/input/` and writes results to `/output/`, following the standard Grand Challenge structure. The specific input files expected are:
    ```
    /input/
    ├── images/
    │   ├── primary-baseline-ct/                               (Image File)
    │   ├── primary-baseline-ct-tumor-lesion-seg/              (Image File)
    │   ├── primary-followup-ct/                               (Image File)
    │   ├── secondary-baseline-ct/                             (Image File)
    │   ├── secondary-baseline-ct-tumor-lesion-seg/            (Image File)
    │   └── secondary-followup-ct/                             (Image File)
    ├── primary-baseline-lesion-clicks.json                    (JSON File)
    ├── primary-followup-lesion-clicks.json                    (JSON File)
    ├── secondary-baseline-lesion-clicks.json                  (JSON File)
    └── secondary-followup-lesion-clicks.json                  (JSON File)
    ```
    The algorithm container reads the necessary files from `/input` based on the specific task it's executing and writes the resulting segmentation mask (`<uuid>.mha`) to one of the following subdirectories within `/output/`:
    ```
    /output/images/primary-followup-ct-tumor-lesion-seg/
    /output/images/secondary-followup-ct-tumor-lesion-seg/
    ```

2.  **Single Case Execution:** Grand Challenge runs the algorithm container *once per case*. The code in `main.py` handles a single execution based on the available inputs.

3.  **Configuration:** The `config/config.yaml` file defines paths (relative to the container, e.g., `/input`, `/output`, model paths) and VOI extraction parameters.

## Baseline Implementation Details

This specific baseline implementation includes:

*   **LongiSeg Framework:** Uses LongiSeg for inference.
*   **Custom Extensions:** Contains custom code (`longiseg_extensions/`) merged into the LongiSeg framework, adapted from the ULS23 baseline. This includes custom resampling and trainer logic.
*   **VOI Extraction:** Implements a Volume of Interest (VOI) extraction strategy (`processors/clickpoint_processor.py`) based on provided lesion click points (centre of gravity).
*   **Pre-trained Model:** Downloads and uses a specific pre-trained nnUNet/LongiSeg model from Zenodo during the Docker build (see `Dockerfile`). The path to this model inside the container is specified in `config/config.yaml`.

## Pipeline Overview

1.  **Initialization (`main.py:MASTBaseline`):** Loads configuration (`config/config.yaml`).
2.  **Data Preparation:** Identifies input images and clickpoints from `/input/`.
3.  **VOI Extraction (`processors/clickpoint_processor.py:VOIExtractor`):** Extracts 3D VOIs around each clickpoint from the relevant CT scans. Uses a generator for memory efficiency.
4.  **Inference (`processors/inference_processor.py:InferenceProcessor`):**
    *   Runs inference on each VOI.
    *   Merges the resulting VOI segmentations back into a single segmentation mask corresponding to the full image dimensions.
5.  **Output:** Saves the final segmentation mask to `/output/<subfoder>/<uuid>.mha`.

## Adding Your Model Weights

There are two main approaches to incorporate your own trained model:

### Option 1: Include Model Weights in the Docker Image

You can include your model weights directly in the Docker image by:

1. **Download during Docker build**: Add commands to the Dockerfile to download your model weights from a cloud storage service (e.g., Zenodo, Google Drive, S3):
   ```dockerfile
   # Example: Download model weights during Docker build
   RUN mkdir -p /opt/app/models \
       && cd /opt/app/models \
       && wget https://example.com/your-model-weights.zip -O model.zip \
       && unzip model.zip \
       && rm model.zip
   ```

2. **Copy from local files**: Store your model weights locally and add a COPY command to the Dockerfile:
   ```dockerfile
   # Copy model weights from local directory
   COPY --chown=user:user path/to/your/models /opt/app/models
   ```

Make sure to update your inference code to load model weights from the appropriate location and modify `config/config.yaml` if necessary to point to your model path.

### Option 2: Upload Model Weights Separately in Grand Challenge

For larger models, you can keep the weights separate from the Docker image and upload them through Grand Challenge:

1. Package your model as a tarball (`.tar.gz`).
2. When creating your algorithm on Grand Challenge, go to the `Models` tab and upload your weights.
3. During inference, Grand Challenge will make these weights available at `/opt/ml/model/`.

You'll need to modify your code to load weights from this location.

```python
# Example code for loading a model from the Grand Challenge models path
# NOTE !START IMPORTANT!
# If using the Grand Challenge model upload feature:
model_path = Path("/opt/ml/model/")
# Load your model with your framework of choice
# NOTE !END IMPORTANT!
```

**NOTE:** The Grand Challenge platform limits repository downloads to 1 GiB per month. For larger models or frequent usage, uploading weights separately is strongly recommended.

## Docker Setup (`Dockerfile`)

The `Dockerfile` defines the container environment:

*   Uses a PyTorch base image.
*   Sets environment variables required by nnUNet and LongiSeg.
*   Installs dependencies from `requirements.txt`.
*   Clones the LongiSeg repository and installs it.
*   Copies the baseline source code (`processors/`, `utils/`, `config/`, `main.py`, `longiseg_extensions/`) into the container.
*   **Merges** the custom `longiseg_extensions/` into the cloned LongiSeg code.
*   **Downloads** the pre-trained model weights from Zenodo.
*   Sets the container's entry point to `python main.py`, which starts the processing pipeline.

Participants can use this baseline as a starting point, potentially modifying the model, VOI extraction, or inference logic while adhering to the Grand Challenge input/output requirements.


