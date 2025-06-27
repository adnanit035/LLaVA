#!/bin/bash
#
##Note: in this file use two # symbols to comment a line!
#
## set the job name, the output files for stdout and stderr streams redirection

#SBATCH --job-name=llava-1.5_pretrain
#SBATCH --output=log_llava-1.5_pretrain.txt
#SBATCH --error=err_llava-1.5_pretrain.txt
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
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /storage2/TEV/airshad/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/blip_laion_cc_sbu_558k.json \
    --image_folder /storage2/TEV/airshad/llava_finetuning/LLaVA/playground/data/LLaVA-Pretrain/images \
    --vision_tower /storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/llava-v1.5-7B-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True