Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Set-Location $HOME\lerobot-project
.\.venv\Scripts\Activate.ps1
python -c "import torch; print('CUDA:', torch.cuda.is_available())"