#!/bin/bash
#
##Note: in this file use two # symbols to comment a line!
#
## set the job name, the output files for stdout and stderr streams redirection


#SBATCH --job-name=llava_finetune_pca_hybrid_144_lora
#SBATCH --output=log_llava15_finetune_pca_hybrid_144_lora.txt
#SBATCH --error=err_llava15_finetune_pca_hybrid_144_lora.err
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4


source ~/.bashrc
source "/home/airshad/miniforge3/etc/profile.d/conda.sh"


cd /storage2/TEV/airshad/llava_finetuning/LLaVA
conda activate llava15_env
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED="True"
export HF_HOME="/storage2/TEV/airshad/huggingface"


echo "assigned gpus=$CUDA_VISIBLE_DEVICES"
export NCCL_P2P_DISABLE=1
include_var="localhost:$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED="True"


deepspeed --include="$include_var" llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /storage2/TEV/airshad/llava_finetuning/LLaVA/playground/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder /storage2/TEV/airshad/llava_finetuning/LLaVA/playground/data/LLaVA-Instruct-150K/images \
    --training_task finetuning \
    --dataloader_drop_last True \
    --sampling_method pca_hybrid \
    --indexes_json_path /storage2/TEV/airshad/Sampling/data/INDEXES/FINETUNE_PCA/pca_hybrid.jsonl \
    --num_visual_tokens 144 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/llava-v1.5-7b-sampling-pca_hybrid-144-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
