#!/bin/bash
set -e  # stop the script if any command fails

#conda init
#conda activate torch260_cu124

# Shared PYTHONPATH
export PYTHONPATH=/datasets/work/vLLM/FFAA-master-newfmt-faster1_transformers4372:$PYTHONPATH

############################################
# GLOBAL SETTINGS
############################################

# Device map variable (EDIT HERE if needed)
DEVICE_MAP="localhost:1,2,3"

# Base_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints/ffaa-mistral-7b"
Base_vLLM_path="/datasets/work/vLLM/FFAA-master-newfmt-faster1_transformers4372/checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_0"
Finetuned_vLLM_path="/datasets/work/vLLM/FFAA-master-newfmt-faster1_transformers4372/checkpoints_4+5fmt/effaa-llava-mistral-7b-lora"
Merged_vLLM_path="/datasets/work/vLLM/FFAA-master-newfmt-faster1_transformers4372/checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_1"
all_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json"
base_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json"
eval_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json"

# init_mids="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt_fix/mids_public.pth"
# init_mids="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt_fix/mids_orgsetup.pth"
# init_mids="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt_fix/mids_best.pth"
# init_mids="/datasets/work/vLLM/FFAA-master-newfmt-faster1_transformers4372/checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_1/mids.pth"
init_mids="/datasets/work/vLLM/temp/PAAS_ensemble_v3/runs/mids_head/mids_v1_20260624_071114/1.pth"

image_dir1="/datasets/work/vLLM/data/fmt_error_all"
output_json1="/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err1.json"

image_dir2="/datasets/work/vLLM/data/fmt_error11"
output_json2="/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err2.json"

image_dir3="/datasets/work/vLLM/data/no_delete_mids_train/miss_axonlabs_data_1_mids++_fastermodel"
output_json3="/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err3.json"

############################################
# STEP 1 — Finetune Mistral LoRA
############################################

echo "==== Step 1: Finetune Mistral LoRA ===="

# deepspeed --master_port 25642 --include $DEVICE_MAP \
#     llava/train/train_mem.py \
#     --lora_enable True --lora_r 32 --lora_alpha 48 --lora_dropout 0.05 --mm_projector_lr 1e-6 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path $Base_vLLM_path \
#     --version v1 \
#     --data_path $all_json \
#     --image_folder /datasets/newout \
#     --vision_tower ./models/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $Finetuned_vLLM_path \
#     --num_train_epochs 7 \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 24 \
#     --gradient_accumulation_steps 1 \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 3 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 16 \
#     --lazy_preprocess True \
#     --report_to "none" \
#     --eval_data_path $eval_json \
#     --evaluation_strategy steps \
#     --eval_steps 500 \
#     --metric_for_best_model eval_loss \
#     --greater_is_better False


# python3.12 merge_lora_weights.py \
# --model-path $Finetuned_vLLM_path \
# --model-base $Base_vLLM_path \
# --save-model-path $Merged_vLLM_path

############################################
# STEP 2 — Build MIDS Dataset
############################################

echo "==== Step 2: Make MIDS dataset ===="

run_parts () {
  local cmd="$1"
  local num_parts="$2"
  local raw_devices="${DEVICE_MAP#*:}"
  local devices=()
  local pids=()
  IFS=',' read -ra devices <<< "$raw_devices"
  local device_count="${#devices[@]}"
  local failed=0

  if [ "$device_count" -eq 0 ]; then
    echo "No CUDA devices configured in DEVICE_MAP=$DEVICE_MAP" >&2
    exit 1
  fi

  for i in $(seq 0 $((num_parts - 1))); do
    local gpu="${devices[$((i % device_count))]}"
    echo "Starting part $i/$((num_parts - 1)) on GPU $gpu"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      eval "$cmd --which_part $i --device 0"
    ) &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  return "$failed"
}

# # from json
# run_parts "python3.12 make_mids_dataset_from_json.py \
#    --model_path $Merged_vLLM_path \
#    --input_json $base_json --batch_size 140" 7

# # folder batch
# run_parts "python3.12 make_mids_dataset_from_folder_batch.py \
#     --model_path $Merged_vLLM_path --batch_size 140 \
#     --input_dir /datasets/work/vLLM/data/no_delete_mids_train \
#     --output_json /datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir.json \
#     --n_divided 4" 4

# # folder one-by-one (dir1)
# run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
#   --model_path $Merged_vLLM_path \
#   --input_dir $image_dir1 \
#   --output_json $output_json1 \
#   --n_divided 20 \
#   --repeat 1" 20
# # run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
# #   --model_path $Merged_vLLM_path \
# #   --input_dir $image_dir1 \
# #   --output_json /datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err1rr.json \
# #   --resume mids_dir_err1_0.json,mids_dir_err1_2.json,mids_dir_err1_4.json,mids_dir_err1_6.json,mids_dir_err1_7.json,mids_dir_err1r_0.json,mids_dir_err1r_1.json,mids_dir_err1r_2.json,mids_dir_err1r_3.json,mids_dir_err1r_4.json,mids_dir_err1r_5.json,mids_dir_err1r_6.json,mids_dir_err1r_7.json,mids_dir_err1r_8.json,mids_dir_err1r_9.json,mids_dir_err1r_10.json,mids_dir_err1r_11.json,mids_dir_err1r_12.json,mids_dir_err1r_13.json,mids_dir_err1r_14.json,mids_dir_err1r_15.json \
# #   --n_divided 16 \
# #   --repeat 1" 16

# # folder one-by-one (dir2)
# run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
#   --model_path $Merged_vLLM_path \
#   --input_dir $image_dir2 \
#   --output_json $output_json2 \
#   --n_divided 16 \
#   --repeat 10" 16

# # folder one-by-one (dir3)
# run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
#   --model_path $Merged_vLLM_path \
#   --input_dir $image_dir3 \
#   --output_json $output_json3 \
#   --n_divided 20 \
#   --repeat 10" 20

echo "All dataset parts finished."

# python3.12 merge_mids_json.py
# # python3.12 merge_check_mids_json.py

# exit

############################################
# STEP 3 — Train MIDS v1
############################################

echo "==== Step 3: Train MIDS v1 ===="

deepspeed --master_port 25636 --include $DEVICE_MAP \
    train_mids_new.py \
    --hidden_dim 768 \
    --version v1 \
    --image_model_path models/clip-vit-large-patch14-336 \
    --text_model_path models/t5-base \
    --init_model_path $init_mids \
    --data_path /datasets/newout/vqa_info_2+13+4+3_fmt/mids.json \
    --val_data_path /datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json \
    --output_dir checkpoints_4+5fmt/mids \
    --per_device_train_batch_size 24 \
    --per_device_val_batch_size 8 \
    --learning_rate 1e-5 \
    --unfreeze_vision_encoder_last_layers 2 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-5 \
    --eval_every_steps 5000 \
    --select_metric acc

echo "==== ALL STEPS COMPLETED SUCCESSFULLY ===="
