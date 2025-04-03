#!/bin/bash

# Capture the start time
start_time=$(date +%s)

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH

./build.sh

MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge



echo "Running evaluation"
# Do not change any of the parameters to docker run, these are fixed

docker run -it --rm \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --gpus="device=0" \
    --shm-size="2g" \
    -v $SCRIPTPATH/test/input/:/input/ \
    -v $SCRIPTPATH/test/output/:/output/ \
    autopet_baseline

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t autopet_eval .
docker run --rm -it \
    -v $SCRIPTPATH/test/output/:/output/ \
    -v $SCRIPTPATH/test/expected_output_nnUNet/:/expected_output/ \
    autopet_eval python3 -c """
import SimpleITK as sitk
import os
import numpy as np

file = os.listdir('/output/images/tumor-lesion-segmentation')[0]
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/tumor-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/psma_95b833d46f153cd2_2018-04-16.nii.gz'))

mse = sum(sum(sum((output - expected_output) ** 2)))
if mse < 10:
    print('Test passed!')
else:
    print(f'Test failed! MSE={mse}')
"""


# Capture the end time and print difference
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total runtime: $elapsed_time seconds"