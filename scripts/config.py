from pathlib import Path
import os
import torch
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
WANDB_KEY = user_secrets.get_secret("WANDB_API_KEY")

OUT_DIR = Path("/kaggle/working/project_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_LABELS = ["anger", "fear", "joy", "sadness", "surprise"]

USE_WANDB = False
try:
    import wandb
    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
        USE_WANDB = True
        print("W&B: logged in.")
    else:
        os.environ["WANDB_MODE"] = "offline"
        USE_WANDB = True
        print("W&B running in offline mode.")
except Exception:
    print("W&B not available.")
    USE_WANDB = False
