# external imports
from transformers import pipeline
import scipy

# local imports
import config

class Musicgen_Small:
    def __init__(self):
        self.local_pipeline = pipeline("text-to-audio", model=config.MUSICGEN_MODEL)

    def generate_music_local_pipeline(self, prompt, audio_path):
        music = self.local_pipeline(prompt, forward_params={"do_sample": True, "max_new_tokens": config.MUSICGEN_MAX_NEW_TOKENS})
        scipy.io.wavfile.write(audio_path, rate=music["sampling_rate"], data=music["audio"])
