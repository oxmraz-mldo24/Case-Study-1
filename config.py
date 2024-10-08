import os
import logging as log

log.basicConfig(level=log.INFO)

SERVICE_PORT = 7860
SERVER_NAME = "0.0.0.0"

IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"

LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LLM_MAX_LENGTH = 50
LLM_MAX_NEW_TOKENS = 50
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95

MUSICGEN_MODEL = "facebook/musicgen-small"
MUSICGEN_MODEL_API_URL = f"https://api-inference.huggingface.co/models/{MUSICGEN_MODEL}"
MUSICGEN_MAX_NEW_TOKENS = 256 # 5 seconds of audio

AUDIO_DIR = "Case-Study-1/data/"

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if HF_API_TOKEN:
    log.info(f"Read HF_API_TOKEN: {HF_API_TOKEN[0:4]}...")
else:
    print("HF_API_TOKEN not found in environment variables.")

# ----ATTRIBUTION-START----
# LLM: Github Copilot
# PROMPT: create an output folder for the generated audio files
# EDITS: /
def create_output_folder():
    os.makedirs(AUDIO_DIR, exist_ok=True)
# -----ATTRIBUTION-END-----

create_output_folder()
