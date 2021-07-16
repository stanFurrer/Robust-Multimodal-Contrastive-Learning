#!/usr/bin/env bash
#SBATCH -J "ViLT"
#SBATCH -o "ViLT_res.txt"
#SBATCH -e "ViLT_err.txt"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

nvidia-smi
source /itet-stor/yirxing/net_scratch/miniconda3/bin/activate contrastive_vilt

export MASTER_ADDR="localhost"
export MASTER_PORT=12356
export NODE_RANK='0'
export TORCH_DISTRIBUTED_DEBUG='DETAIL'

# python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=<BS_FITS_YOUR_GPU>

# python run.py with data_root=../Datasets/coco/ViLT num_gpus=1 num_nodes=1 task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=16
# python run.py with data_root=../Datasets/ViLT num_gpus=2 num_nodes=1 task_moco step200k per_gpu_batchsize=16 load_path="weights/vilt_200k_mlm_itm.ckpt"
python run.py with data_root=/scratch_net/tikgpu05/yirxing/ViLT num_gpus=2 num_nodes=1 task_moco step200k per_gpu_batchsize=8
