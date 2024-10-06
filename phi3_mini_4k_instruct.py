# external imports
from transformers import pipeline
from huggingface_hub import InferenceClient
import torch
# local imports
import config
from llama_cpp import Llama


class Phi3_Mini_4k_Instruct:
    def __init__(self):
        pass

    def generate_text(self, messages, use_local_llm):
        if use_local_llm:
            return self.generate_text_llama_cpp(messages)
        else:
            return self.generate_text_api(messages)

    def generate_text_llama_cpp(self, messages):
        model = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf"
        )
        response = model.create_chat_completion(messages)
        generated_message = response['choices'][0]['message']['content']

        return generated_message

    def generate_text_local_pipeline(self, messages):
        self.local_pipeline = pipeline("text-generation", model=config.LLM_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.local_pipeline.model.config.max_length = config.LLM_MAX_LENGTH
        self.local_pipeline.model.config.max_new_tokens = config.LLM_MAX_NEW_TOKENS
        self.local_pipeline.model.config.temperature = config.LLM_TEMPERATURE
        self.local_pipeline.model.config.top_p = config.LLM_TOP_P
        result = self.local_pipeline(messages)[-1]['generated_text'][-1]['content']
        return result

    def generate_text_api(self, messages):
        client = InferenceClient(config.LLM_MODEL, token=config.HF_API_TOKEN)
        try:
            result = client.chat_completion(messages, max_tokens=config.LLM_MAX_NEW_TOKENS, temperature=config.LLM_TEMPERATURE, top_p=config.LLM_TOP_P).choices[0].message.content
        except Exception as e:
            result = f"Error: {e}"
        return result