@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" sft_train.py --base runs/singularity_decay_b/final.pt --name Singularity --mix data/sft_mix/singularity.jsonl --out runs/singularity_sft
pause
