# external imports
from transformers import pipeline

# local imports
import config


class Phi3_Mini_4k_Instruct:
    def __init__(self):
        self.local_pipeline = pipeline("text-generation", model=config.LLM_MODEL, trust_remote_code=True)
        self.local_pipeline.model.config.max_length = config.LLM_MAX_LENGTH
        self.local_pipeline.model.config.max_new_tokens = config.LLM_MAX_NEW_TOKENS

    def generate_text_local_pipeline(self, messages):
        result = self.local_pipeline(messages)
        return result
