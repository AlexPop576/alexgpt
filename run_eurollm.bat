@echo off
rem EuroLLM-9B (better Romanian than Qwen): install/check Ollama, pull the model, test 3 conversations, then generate.
cd /d %~dp0
call setup_and_run_local_gen.bat hf.co/lmstudio-community/EuroLLM-9B-Instruct-GGUF:Q4_K_M
