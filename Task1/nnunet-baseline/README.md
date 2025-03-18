# nnUNet baseline algorithm for autoPET/CT IV challenge

Source code for the nnUNet baseline algorithm container for autoPETIV challenge. Information about the 
submission can be found [here](https://autopet-iv.grand-challenge.org/submission/) and in the [grand challenge 
documentation](https://grand-challenge.org/documentation/).

## Task
Best ranked model wins! The rules are simple: Train a model which generalizes well on FDG and PSMA data. This baseline model is out of competition!

## Usage 

In order to use the baseline you first need to unzip the baseline weights via `bash unzip_model_weights.sh`. 
After that you can build the container by running `bash build.sh`. In order to upload the container, you will need to
save the image via `bash export.sh`.

## Testing

Use a python 3.10 based environment and install the requirements.txt file via `pip install -r requirements.txt`. 
Make sure model weights exist in `/nnUNet_results`. Unzip the baseline weights by running `bash unzip_model_weights.sh`. 
Then run `bash create_expected_output.sh` to create an expected_output mask. After that you can run `bash test.sh`.

