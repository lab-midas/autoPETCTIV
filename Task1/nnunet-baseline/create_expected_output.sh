#!/bin/bash 

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
echo $SCRIPTPATH
export nnUNet_raw="${SCRIPTPATH}/test"
export nnUNet_preprocessed="${SCRIPTPATH}/test"
export nnUNet_results="${SCRIPTPATH}/nnunet-baseline/nnUNet_results"
nnUNetv2_predict -i "${SCRIPTPATH}/test/orig/images" -o "${SCRIPTPATH}/test/expected_output_nnUNet" -d 221 -c 3d_fullres -f 0 --disable_tta