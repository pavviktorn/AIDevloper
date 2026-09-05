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

#Base_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints_phi3/llava-phi-3-mini"
Base_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora_2"
Finetuned_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora"
Merged_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora_3"
all_json="/datasets/newout/vqa_info/add_bordertest&pad&df/eFFAA_ext.json"
base_json="/datasets/newout/vqa_info/eFFAA_dataset.json"

############################################
# STEP 1 — Finetune Mistral LoRA
############################################

echo "==== Step 1: Finetune Mistral LoRA ===="

deepspeed --master_port 25642 --include $DEVICE_MAP \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 48 --lora_dropout 0.05 --mm_projector_lr 1e-6 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $Base_vLLM_path \
    --version v1 \
    --data_path $all_json \
    --image_folder /datasets/newout \
    --vision_tower ./models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $Finetuned_vLLM_path \
    --num_train_epochs 10 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to "none" \
    --eval_data_path /datasets/newout/eFFAA_val_dataset.json \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --metric_for_best_model eval_loss \
    --greater_is_better False


python merge_lora_weights.py \
--model-path $Finetuned_vLLM_path \
--model-base $Base_vLLM_path \
--save-model-path $Merged_vLLM_path

exit

############################################
# STEP 2 — Build MIDS Dataset
############################################

echo "==== Step 2: Make MIDS dataset ===="

python make_mids_dataset_from_json.py \
    --which_part 1 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
python make_mids_dataset_from_json.py \
    --which_part 2 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
python make_mids_dataset_from_json.py \
    --which_part 3 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
python make_mids_dataset_from_json.py \
    --which_part 4 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
python make_mids_dataset_from_json.py \
    --which_part 5 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
python make_mids_dataset_from_json.py \
    --which_part 6 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
python make_mids_dataset_from_json.py \
    --which_part 7 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 160 &
wait

python make_mids_dataset_from_folder_batch.py --which_part 1 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_batch.py --which_part 2 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_batch.py --which_part 3 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_batch.py --which_part 4 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_batch.py --which_part 5 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_batch.py --which_part 6 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_batch.py --which_part 7 --model_path $Merged_vLLM_path --batch_size 160 &
wait

python make_mids_dataset_from_folder_onebyone.py --which_part 1 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_onebyone.py --which_part 2 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_onebyone.py --which_part 3 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_onebyone.py --which_part 4 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_onebyone.py --which_part 5 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_onebyone.py --which_part 6 --model_path $Merged_vLLM_path --batch_size 160 &
python make_mids_dataset_from_folder_onebyone.py --which_part 7 --model_path $Merged_vLLM_path --batch_size 160 &
wait

echo "All dataset parts finished."

python merge_mids_json.py
#exit

############################################
# STEP 3 — Train MIDS v1
############################################

echo "==== Step 3: Train MIDS v1 ===="

deepspeed --master_port 25635 --include $DEVICE_MAP \
    train_mids.py \
    --hidden_dim 768 \
    --version v1 \
    --image_model_path models/clip-vit-large-patch14-336 \
    --text_model_path models/t5-base \
    --data_path /datasets/newout/vqa_info/mids.json \
    --val_data_path /datasets/newout/vqa_info/mids_eval.json \
    --output_dir checkpoints/mids \
    --per_device_train_batch_size 24 \
    --per_device_val_batch_size 8 \
    --learning_rate 1e-5 \
    --unfreeze_vision_encoder_last_layers 2 \
    --num_train_epochs 2 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-5

echo "==== ALL STEPS COMPLETED SUCCESSFULLY ===="
