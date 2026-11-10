@echo off
cd /d "%~dp0"
echo === 1/4 astept sa se termine generarea locala (gen_local / gen_grounded / gen_data_wiki) ===
powershell -NoProfile -Command "while (Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'gen_local|gen_grounded|gen_data_wiki' }) { Start-Sleep -Seconds 30 }"
echo === 2/4 verific datele generate ===
python check_gen_data.py > finish_data_log.txt 2>&1
type finish_data_log.txt
echo === 3/4 construiesc mixul SFT pentru Singularity ===
python build_sft_mix.py --name Singularity --size "about 125 million" --size-ro "aproximativ 125 de milioane" >> finish_data_log.txt 2>&1
type finish_data_log.txt
echo === 4/4 gata: data\sft_mix\singularity.jsonl ===
echo Citeste data\sft_mix\singularity_preview.txt si finish_data_log.txt
echo Antrenarea pilot: python sft_train.py --pilot --base runs/singularity_decay_b/final.pt --name Singularity --mix data/sft_mix/singularity.jsonl --out runs/singularity_sft
pause
