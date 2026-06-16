# Paper 2 training environment — Windows / Python 3.12
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"

Write-Host "Creating .venv with Python 3.12..."
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip

# --- PyTorch: choose ONE block for your hardware ---
# NVIDIA GPU (CUDA 12.4 wheels):
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only fallback (comment the line above, uncomment below):
# & $py -m pip install torch torchvision

& $py -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Verify GPU:  $py -c `"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')`""
