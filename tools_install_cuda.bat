@echo off
cd /d "%~dp0"
set PIP_CONFIG_FILE=nul
echo === Installing GPU (CUDA) PyTorch - about 2.5 GB, a few minutes ===
for %%I in (cu130 cu129 cu128 cu126) do (
  echo --- trying %%I ---
  .venv\Scripts\python -m pip --isolated install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/%%I && goto :done
)
echo ERROR: could not install CUDA PyTorch. Send the text above to Claude.
goto :verify
:done
echo === Installed. ===
:verify
.venv\Scripts\python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu_ok',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" > runs\_diag_cuda.txt 2>&1
type runs\_diag_cuda.txt
