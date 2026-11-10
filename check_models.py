import os
from alexgpt.chat import load_models
models = load_models(os.path.dirname(os.path.abspath(__file__)))
print("models:", list(models))
