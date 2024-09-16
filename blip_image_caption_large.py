# external imports
from transformers import pipeline
from huggingface_hub import InferenceClient

# local imports
import config

class Blip_Image_Caption_Large:
    def __init__(self):
        pass

    def caption_image(self, image_path, use_local_caption):
        if use_local_caption:
            return self.caption_image_local_pipeline(image_path)
        else:
            return self.caption_image_api(image_path)
    
    def caption_image_local_pipeline(self, image_path):
        self.local_pipeline = pipeline("image-to-text", model=config.IMAGE_CAPTION_MODEL)
        result = self.local_pipeline(image_path)[0]['generated_text']
        return result

    def caption_image_api(self, image_path):
        client = InferenceClient(config.IMAGE_CAPTION_MODEL, token=config.HF_API_TOKEN)
        result = client.image_to_text(image_path).generated_text
        return result