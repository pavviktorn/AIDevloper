#!/bin/bash
set -e  # stop the script if any command fails

#conda init
#conda activate torch260_cu124

# Shared PYTHONPATH
export PYTHONPATH=/datasets/work/vLLM/FFAA-master:$PYTHONPATH

############################################
# GLOBAL SETTINGS
############################################

# Device map variable (EDIT HERE if needed)
DEVICE_MAP="localhost:1,2,3,4,5,6,7"

vLLM_path="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_1"
mids_path="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_1/mids.pth"

echo "==== Step 1: Select Mis-classified samples ===="

run_parts () {
  local cmd="$1"
  local max="$2"
  for i in $(seq 1 $max); do
    eval "$cmd --which_part $i &"
  done
  wait
}

run_parts "python sel_hard_samples_from_folder_batch.py \
    --model_path $vLLM_path \
    --mids_path $mids_path --batch_size 140" 7

echo "All dataset parts finished."

echo "==== ALL STEPS COMPLETED SUCCESSFULLY ===="
