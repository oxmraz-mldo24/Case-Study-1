# external imports
from transformers import pipeline
import scipy

# local imports
import config

class Musicgen_Small:
    def __init__(self):
        self.local_pipeline = pipeline("text-to-audio", model=config.MUSICGEN_MODEL)

    def generate_music_local_pipeline(self, prompt):
        music = self.local_pipeline(prompt, forward_params={"do_sample": True})
        scipy.io.wavfile.write("data/musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
