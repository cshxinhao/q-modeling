import os
from dotenv import load_dotenv

# Load the .env
load_dotenv()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
FEATURE_DATA_DIR = os.getenv("FEATURE_DATA_DIR")
MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR")

# Google Drive
# RAW_DATA_DIR="/content/drive/MyDrive/colab_env/dataset_china_all/markte_data"
# FEATURE_DATA_DIR="/content/drive/MyDrive/colab_env/dataset_china_all/features"
# MODEL_SAVE_DIR="/content/drive/MyDrive/colab_env/model_warehouse"
