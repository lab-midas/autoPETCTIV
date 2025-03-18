#!/bin/bash

# Capture the start time
start_time=$(date +%s)

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create autopet_baseline-output-$VOLUME_SUFFIX

echo "Volume created, running evaluation"
# Do not change any of the parameters to docker run, these are fixed
# --gpus="device=0" \
docker run -it --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="2g" \
        --pids-limit="256" \
        --gpus="all" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        autopet_baseline

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t autopet_eval .
docker run --rm -it \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output_nnUNet/:/expected_output/ \
        autopet_eval python3 -c """
import SimpleITK as sitk
import os

file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/psma_95b833d46f153cd2_2018-04-16.nii.gz'))


mse = sum(sum(sum((output - expected_output) ** 2)))
if mse < 10:
    print('Test passed!')
else:
    print(f'Test failed! MSE={mse}')
"""

docker volume rm autopet_baseline-output-$VOLUME_SUFFIX

# Capture the end time and print difference
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total runtime: $elapsed_time seconds"