"""PyTorch-only application; avoid importing Kaggle's unrelated TensorFlow/Keras stack."""
import os
os.environ.setdefault("USE_TF", "0")
