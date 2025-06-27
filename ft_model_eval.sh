#!/bin/bash
#
##Note: in this file use two # symbols to comment a line!
#
## set the job name, the output files for stdout and stderr streams redirection

#SBATCH --job-name=llava-1.5_FT_mme_eval
#SBATCH --output=log_llava-1.5_FT_mme_eval.txt
#SBATCH --error=err_llava-1.5_FT_mme_eval.txt
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

source ~/.bashrc
source "/home/airshad/miniforge3/etc/profile.d/conda.sh"

cd "/storage2/TEV/airshad/lmms-eval"
conda activate llava_env
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED="True"
export HF_HOME="/storage2/TEV/airshad/huggingface"

srun python3 -m accelerate.commands.launch \
    --main_process_port=29524 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/storage2/TEV/airshad/llava_finetuning/LLaVA/checkpoints/llava-v1.5-7b-lora" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-1.5_FT_mme_eval \
    --output_path ./logs/

cd ..
chmod -R g+rwx .
