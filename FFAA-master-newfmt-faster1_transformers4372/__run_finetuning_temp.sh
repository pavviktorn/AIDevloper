#!/bin/bash
set -e  # stop the script if any command fails

while pgrep -f __run_finetuning.sh > /dev/null; do
    echo "Waiting Script finished"
    sleep 10
done
echo "Script finished"

mids_dir="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt/mids"
dest_dir="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt/effaa-llava-mistral-7b-lora_0/mids.pth"

latest_dir=$(ls -d "$mids_dir"/mids_* | sort | tail -n 1)

cp "$latest_dir/final.pth" "$dest_dir"

#conda init
#conda activate torch260_cu124

# Shared PYTHONPATH
export PYTHONPATH=/datasets/work/vLLM/FFAA-master-newfmt:$PYTHONPATH

############################################
# GLOBAL SETTINGS
############################################

# Device map variable (EDIT HERE if needed)
DEVICE_MAP="localhost:1,2,3,4,5,6,7"

Base_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints/ffaa-mistral-7b"
#Base_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints/effaa-llava-mistral-7b-lora_5"
Finetuned_vLLM_path="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt/effaa-llava-mistral-7b-lora"
Merged_vLLM_path="/datasets/work/vLLM/FFAA-master-newfmt/checkpoints_4+13fmt/effaa-llava-mistral-7b-lora_0"
all_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json"
base_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json"
eval_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json"

############################################
# STEP 1 — Finetune Mistral LoRA
############################################

echo "==== Step 1: Finetune Mistral LoRA ===="

#deepspeed --master_port 25642 --include $DEVICE_MAP \
#    llava/train/train_mem.py \
#    --lora_enable True --lora_r 32 --lora_alpha 48 --lora_dropout 0.05 --mm_projector_lr 1e-6 \
#    --deepspeed ./scripts/zero3.json \
#    --model_name_or_path $Base_vLLM_path \
#    --version v1 \
#    --data_path $all_json \
#    --image_folder /datasets/newout \
#    --vision_tower ./models/clip-vit-large-patch14-336 \
#    --mm_projector_type mlp2x_gelu \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --image_aspect_ratio pad \
#    --group_by_modality_length True \
#    --bf16 True \
#    --output_dir $Finetuned_vLLM_path \
#    --num_train_epochs 3 \
#    --per_device_train_batch_size 24 \
#    --per_device_eval_batch_size 12 \
#    --gradient_accumulation_steps 1 \
#    --save_strategy "steps" \
#    --save_steps 500 \
#    --save_total_limit 3 \
#    --learning_rate 1e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to "none" \
#    --eval_data_path $eval_json \
#    --evaluation_strategy steps \
#    --eval_steps 500 \
#    --metric_for_best_model eval_loss \
#    --greater_is_better False
#
#
#python merge_lora_weights.py \
#--model-path $Finetuned_vLLM_path \
#--model-base $Base_vLLM_path \
#--save-model-path $Merged_vLLM_path

############################################
# STEP 2 — Build MIDS Dataset
############################################

echo "==== Step 2: Make MIDS dataset ===="

python make_mids_dataset_from_json.py \
    --which_part 1 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
python make_mids_dataset_from_json.py \
    --which_part 2 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
python make_mids_dataset_from_json.py \
    --which_part 3 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
python make_mids_dataset_from_json.py \
    --which_part 4 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
python make_mids_dataset_from_json.py \
    --which_part 5 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
python make_mids_dataset_from_json.py \
    --which_part 6 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
python make_mids_dataset_from_json.py \
    --which_part 7 --model_path $Merged_vLLM_path --input_json $base_json --batch_size 140 &
wait

python make_mids_dataset_from_folder_batch.py --which_part 1 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_batch.py --which_part 2 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_batch.py --which_part 3 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_batch.py --which_part 4 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_batch.py --which_part 5 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_batch.py --which_part 6 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_batch.py --which_part 7 --model_path $Merged_vLLM_path --batch_size 140 &
wait

python make_mids_dataset_from_folder_onebyone.py --which_part 1 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_onebyone.py --which_part 2 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_onebyone.py --which_part 3 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_onebyone.py --which_part 4 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_onebyone.py --which_part 5 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_onebyone.py --which_part 6 --model_path $Merged_vLLM_path --batch_size 140 &
python make_mids_dataset_from_folder_onebyone.py --which_part 7 --model_path $Merged_vLLM_path --batch_size 140 &
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
    --init_model_path $dest_dir \
    --data_path /datasets/newout/vqa_info_2+13+4+3_fmt/mids.json \
    --val_data_path /datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json \
    --output_dir checkpoints_4+13fmt/mids \
    --per_device_train_batch_size 24 \
    --per_device_val_batch_size 8 \
    --learning_rate 1e-5 \
    --unfreeze_vision_encoder_last_layers 2 \
    --num_train_epochs 2 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-5

echo "==== ALL STEPS COMPLETED SUCCESSFULLY ===="
