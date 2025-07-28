# Makefile for Label Propagation GPU Acceleration

# Compiler and flags
NVCC = nvcc
CXX = g++
PYTHON = python3

# CUDA flags
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++17
CXX_FLAGS = -O3 -fopenmp -std=c++17

# Targets
.PHONY: all clean gpu-cuda gpu-python cpu-cpp run-gpu run-cpu help

all: gpu-cuda gpu-python

# Build CUDA C++ version
gpu-cuda: label_prop_sbm_gpu

label_prop_sbm_gpu: label_prop_sbm_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build CPU C++ version
cpu-cpp: label_prop_sbm

label_prop_sbm: label_prop_sbm.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $<

# Install Python dependencies
gpu-python:
	$(PYTHON) -m pip install -r requirements.txt

# Run GPU experiments
run-gpu: gpu-python
	$(PYTHON) run_lpa_experiment_gpu.py --num_params 32 --n 2000 --rounds 5 --trials 4

run-gpu-large: gpu-python
	$(PYTHON) run_lpa_experiment_gpu.py --num_params 64 --n 5000 --rounds 10 --trials 8

# Run CPU experiments
run-cpu: cpu-cpp
	./label_prop_sbm 2000 -1.0 0.0 -1.0 0.0 5 4

# Submit SLURM job
submit-slurm:
	sbatch slurm_gpu_job.sh

# Generate plots from existing results
plot-results:
	$(PYTHON) generate_heatmaps.py --run_folder results/latest

# Clean build artifacts
clean:
	rm -f label_prop_sbm_gpu label_prop_sbm
	rm -rf __pycache__ *.pyc
	find . -name "*.o" -delete

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all versions"
	@echo "  gpu-cuda         - Build CUDA C++ version"
	@echo "  gpu-python       - Install Python GPU dependencies"
	@echo "  cpu-cpp          - Build CPU C++ version"
	@echo "  run-gpu          - Run small GPU experiment"
	@echo "  run-gpu-large    - Run large GPU experiment"
	@echo "  run-cpu          - Run CPU experiment"
	@echo "  submit-slurm     - Submit SLURM job"
	@echo "  plot-results     - Generate plots from results"
	@echo "  clean            - Clean build artifacts"
	@echo "  help             - Show this help"

# Performance comparison
benchmark: gpu-cuda gpu-python cpu-cpp
	@echo "=== Performance Benchmark ==="
	@echo "Running CPU version..."
	@time ./label_prop_sbm 1000 -0.5 0.0 -0.5 0.0 5 2
	@echo "Running GPU Python version..."
	@time $(PYTHON) run_lpa_experiment_gpu.py --num_params 16 --n 1000 --rounds 5 --trials 2 --output_dir /tmp/benchmark
	@echo "Benchmark completed!" 