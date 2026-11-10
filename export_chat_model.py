"""Makes the portable chat folder singularity_chat/ (copy it to any computer, no GPU needed):
the model in float16 (half the size), the tokenizer and the 4 alexgpt files the chat needs, then a zip of it all.
    python export_chat_model.py runs/singularity_sft4/final.pt
"""
import os
import shutil
import sys
import zipfile

import torch

source = sys.argv[1] if len(sys.argv) > 1 else "runs/singularity_sft4/final.pt"
folder = "singularity_chat"
saved = torch.load(source, map_location="cpu", weights_only=False)
weights = {name: tensor.half() for name, tensor in saved["weights"].items() if name != "out.weight"}   # out.weight = token_table (tied)
torch.save({"config": saved["config"], "weights": weights, "step": saved.get("step"), "chat": True}, os.path.join(folder, "singularity_v4.pt"))
os.makedirs(os.path.join(folder, "alexgpt"), exist_ok=True)
for name in ("__init__.py", "supernova.py", "bpe_fast.py", "sft_format.py", "tools.py"):
    shutil.copy(os.path.join("alexgpt", name), os.path.join(folder, "alexgpt", name))
shutil.copy("checkpoints/supernova_tokenizer.json", os.path.join(folder, "tokenizer.json"))
with zipfile.ZipFile("singularity_chat.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name in ("memory.json",) or "__pycache__" in root:
                continue
            path = os.path.join(root, file_name)
            archive.write(path, path)
print("model:", os.path.getsize(os.path.join(folder, "singularity_v4.pt")) // 2**20, "MB;  zip:", os.path.getsize("singularity_chat.zip") // 2**20, "MB")
