import os
from transformers import AutoConfig

MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = "services/modal-deberta/model_local/deberta-v3-aes"

print(f"Fetching config for {MODEL_NAME}...")
config = AutoConfig.from_pretrained(MODEL_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "config.json")
config.save_pretrained(OUTPUT_DIR)

print(f"✅ Config saved to {output_path}")
