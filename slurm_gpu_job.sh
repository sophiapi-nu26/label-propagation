#!/bin/bash
#SBATCH --job-name=label_prop_gpu
#SBATCH --output=logs/label_prop_gpu_%j.out
#SBATCH --error=logs/label_prop_gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Load necessary modules (adjust for your cluster)
module load cuda/12.0
module load python/3.9
module load gcc/11.2

# Create logs directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Install/update requirements
pip install --upgrade pip
pip install -r requirements.txt

# Print GPU information
nvidia-smi
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"

# Run GPU experiment
echo "Starting GPU-accelerated label propagation experiment..."
python run_lpa_experiment_gpu.py \
    --num_params 64 \
    --n 5000 \
    --rounds 10 \
    --trials 8 \
    --output_dir results/gpu_run_$(date +%Y%m%d_%H%M%S)

# Alternative: Run CUDA C++ version
# echo "Compiling CUDA C++ version..."
# nvcc -O3 -arch=sm_80 -o label_prop_sbm_gpu label_prop_sbm_gpu.cu
# 
# echo "Running CUDA C++ experiment..."
# ./label_prop_sbm_gpu 5000 -1.0 0.0 -1.0 0.0 10 8

echo "Experiment completed!" 