#!/usr/bin/bash

# Capture the start time
start_time=$(date +%s)

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P)")"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"

echo $SCRIPTPATH

./build.sh

MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge



# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --gpus="all"  \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v $SCRIPTPATH/test/output/:/output/ \
        sw_infer 

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t sw_infer_eval .
docker run --rm -it \
        -v $SCRIPTPATH/test/output/:/output/ \
        -v $SCRIPTPATH/test/expected_output_interactive/:/expected_output/ \
        sw_infer_eval python3 -c """
import SimpleITK as sitk
import os

file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/psma_95b833d46f153cd2_2018-04-16_0001.nii.gz'))


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