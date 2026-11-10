"""
Keeps a safe copy of Singularity as it is after pre-training (decay A, before any SFT):
    runs/singularity/final.pt  ->  checkpoints/singularity_base_decayA.pt   (+ md5 in runs/singularity/base_copy.txt)
Runs by itself in the job queue after the training. Never overwrites an existing copy.
"""
import hashlib
import os
import shutil
import sys

SOURCE = "runs/singularity/final.pt"
TARGET = "checkpoints/singularity_base_decayA.pt"


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


if not os.path.exists(SOURCE):
    print(f"{SOURCE} not found: the training is not finished.")
    sys.exit(1)
if os.path.exists(TARGET):
    print(f"{TARGET} already exists, not overwritten (md5 {md5_of(TARGET)})")
    sys.exit(0)
shutil.copyfile(SOURCE, TARGET)
source_md5, target_md5 = md5_of(SOURCE), md5_of(TARGET)
assert source_md5 == target_md5, "the copy differs from the original"
with open("runs/singularity/base_copy.txt", "w", encoding="utf-8") as file:
    file.write(f"{TARGET} md5 {target_md5}\n")
print(f"saved {TARGET} (md5 {target_md5})")
