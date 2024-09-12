# external imports
from transformers import pipeline

# local imports
import config

class Blip_Image_Caption_Large:
    def __init__(self):
        self.local_pipeline = pipeline("image-to-text", model=config.IMAGE_CAPTION_MODEL)

    def caption_image_local_pipeline(self, image_path):
        result = self.local_pipeline(image_path)
        return result
