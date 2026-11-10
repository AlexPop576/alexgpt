@echo off
rem Stops the Singularity training cleanly: it finishes the current step, saves, and exits (nothing lost).
rem To continue later: python train_singularity.py  (or ask Claude).
cd /d F:\Code\PersonalProjects\LLM
if exist runs\singularity_decay_b\log.txt (
    echo stop > runs\singularity_decay_b\STOP
)
echo stop > runs\singularity\STOP
echo Stop requested. Within ~10 seconds the log shows "checkpoint saved" and "stopped".
pause
