#!/bin/bash

# Capture the start time
start_time=$(date +%s)

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
echo $SCRIPTPATHCURR

./build.sh

MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge



echo "Running evaluation"
docker run -it --rm \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --network="none" \
    --shm-size="2g" \
    --gpus="all" \
    -v $SCRIPTPATH/test/input/:/input/ \
    -v $SCRIPTPATH/test/output/:/output/ \
    mast_baseline
    
