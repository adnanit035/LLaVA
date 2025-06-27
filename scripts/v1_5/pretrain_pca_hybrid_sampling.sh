#!/bin/bash
#
##Note: in this file use two # symbols to comment a line!
#
## set the job name, the output files for stdout and stderr streams redirection


#SBATCH --job-name=llava_pretrain_sampling_pca_hybrid_144
#SBATCH --output=log_llava15_pretrain_sampling_pca_hybrid_144.txt
#SBATCH --error=err_llava15_pretrain_sampling_pca_hybrid_144.err
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4


source ~/.bashrc
source "/home/airshad/miniforge3/etc/profile.d/conda.sh"


cd /storage2/TEV/airshad/llava_finetuning/LLaVA
conda activate llava15_env
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED="True"
export HF_HOME="/storage2/TEV/airshad/huggingface"
export CUDA_HOME="/opt/cuda/12.2"


echo "assigned gpus=$CUDA_VISIBLE_DEVICES"
export NCCL_P2P_DISABLE=1
include_var="localhost:$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED="True"


deepspeed --include="$include_var" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --version plain \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --data_path /storage2/TEV/airshad/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/blip_laion_cc_sbu_558k.json \
    --image_folder /storage2/TEV/airshad/llava_finetuning/LLaVA/playground/data/LLaVA-Pretrain/images \
    --indexes_json_path /storage2/TEV/airshad/Sampling/data/INDEXES/PRETRAIN_PCA_Optimized-2/pca_hybrid.json \
    --num_visual_tokens 144 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/llava-v1.5-7b-sampling-pca-hybrid-144-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
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
    --lazy_preprocess True \
    --report_to tensorboard
