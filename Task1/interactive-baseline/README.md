# Interactive baseline algorithm for autoPETIV challenge

Source code for the interactive baseline algorithm container for autoPETIV challenge. Information about the 
submission can be found [here](https://autopet-iv.grand-challenge.org/submission/) and in the [grand challenge 
documentation](https://grand-challenge.org/documentation/).

## Task
TODO

## Usage 

In order to use the baseline you can build the container by running `bash build.sh`. In order to upload the container, you will need to save the image via `bash export.sh`.

## Testing

Use a python 3.10 based environment and install the requirements.txt file via `pip install -r requirements.txt`. 
The model weights exist in `/model`. Run `bash create_expected_output.sh` to create an expected_output mask. After that you can run `bash test.sh`.

