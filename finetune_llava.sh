#!/bin/bash
#
##Note: in this file use two # symbols to comment a line!
#
## set the job name, the output files for stdout and stderr streams redirection

#SBATCH --job-name=llava-1.5_finetune
#SBATCH --output=log_llava-1.5_finetune.txt
#SBATCH --error=err_llava-1.5_finetune.txt
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

source ~/.bashrc
source "/home/airshad/miniforge3/etc/profile.d/conda.sh"

cd "/storage2/TEV/airshad/llava_finetuning/LLaVA/"

conda activate llava15_env
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED="True"
export HF_HOME="/storage2/TEV/airshad/huggingface"

# Set the prompt and model versions directly in the command
srun python llava/train/train_mem.py \
    --deepspeed /storage2/TEV/airshad/llava_finetuning/LLaVA/scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /storage2/TEV/airshad/llava_finetuning/LLaVA/playground/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder /storage2/TEV/airshad/llava_finetuning/LLaVA/playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/llava-v1.5-7b-finetuned \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
