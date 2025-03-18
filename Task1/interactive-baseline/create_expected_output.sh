#!/bin/bash 

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
echo $SCRIPTPATH

python src/simplified_inference.py -a -i "${SCRIPTPATH}/test/orig/" --dataset AutoPET4 \
-o "${SCRIPTPATH}/test/expected_output_interactive" --json_dir "${SCRIPTPATH}/test/orig/lesion-clicks" -e 800 \
--dont_check_output_dir  --resume_from "${SCRIPTPATH}/interactive-baseline/model/151_best_0.8534.pt" \
--eval_only --no_log --no_data --save_pred -c cache/