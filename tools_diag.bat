@echo off
cd /d "%~dp0"
if not exist runs mkdir runs
(
echo === nvidia-smi ===
nvidia-smi
echo === nvcc ===
nvcc --version
echo === torch ===
.venv\Scripts\python -c "import torch,sys;print(sys.version);print('torch',torch.__version__,'cuda_build',torch.version.cuda,'cuda_ok',torch.cuda.is_available())"
) > runs\_diag.txt 2>&1
echo Diagnostics written to runs\_diag.txt
