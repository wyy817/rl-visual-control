#!/bin/bash
#SBATCH --job-name=ppo_carracing        # Job name
#SBATCH --partition=aiaca40             # A40 partition
#SBATCH --qos=4a40                      # A40 QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4              # 降低CPU申请以减少排队等待
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out                 # 标准输出 -> <job_id>.out
#SBATCH --error=%j.err                  # 错误输出 -> <job_id>.err

# ---- 激活 conda 环境（改成你在 HPC 上的环境名）----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dts307

# ---- 进入代码目录 ----
cd path/to/ppo_carracing

# ---- 启动训练 ----
python train.py \
    --n_envs 4 \
    --n_steps 512 \
    --total_timesteps 2000000 \
    --batch_size 256 \
    --n_epochs 10 \
    --lr 3e-4 \
    --log_dir runs/ppo_v2 \
    --save_dir checkpoints
