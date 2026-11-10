@echo off
cd /d "%~dp0"
echo === 1/4 astept sa se termine generarea (nu inchide scripturile de generare, las-le sa termine) ===
powershell -NoProfile -Command "while (Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'gen_local|gen_grounded|gen_data_wiki' }) { Start-Sleep -Seconds 30 }"
echo === 2/4 verific datele ===
python check_gen_data.py > finish_data_log.txt 2>&1
echo === 3/4 construiesc mixul ===
python build_sft_mix.py --name Singularity --size "about 125 million" --size-ro "aproximativ 125 de milioane" >> finish_data_log.txt 2>&1
type finish_data_log.txt
echo === 4/4 pornesc antrenarea SFT la 11:45 daca nu ai pornit-o tu ===
powershell -NoProfile -Command "while ((Get-Date).TimeOfDay -lt [TimeSpan]'11:45:00' -and -not (Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'sft_train' })) { Start-Sleep -Seconds 30 }"
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'sft_train' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo antrenarea ruleaza deja, nu pornesc a doua
  pause
  exit /b
)
python sft_train.py --base runs/singularity_decay_b/final.pt --name Singularity --mix data/sft_mix/singularity.jsonl --out runs/singularity_sft > sft_train_log.txt 2>&1
echo ANTRENARE TERMINATA
pause
