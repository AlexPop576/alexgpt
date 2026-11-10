"""build_sft_mix.py with the Singularity arguments (the job runner cannot pass quoted arguments)."""
import runpy
import sys
sys.argv = ["build_sft_mix.py", "--name", "Singularity", "--size", "about 125 million", "--size-ro", "aproximativ 125 de milioane"]
runpy.run_path("build_sft_mix.py", run_name="__main__")
