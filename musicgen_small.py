# external imports
from transformers import pipeline
from io import BytesIO
import requests
import scipy

# local imports
import config

class Musicgen_Small:
    def __init__(self):
        pass

    def generate_music(self, prompt, audio_path, use_local_musicgen):
        if use_local_musicgen:
            self.generate_music_local_pipeline(prompt, audio_path)
        else:
            self.generate_music_api(prompt, audio_path)
    
    def generate_music_local_pipeline(self, prompt, audio_path):
        self.local_pipeline = pipeline("text-to-audio", model=config.MUSICGEN_MODEL)
        music = self.local_pipeline(prompt, forward_params={"do_sample": True, "max_new_tokens": config.MUSICGEN_MAX_NEW_TOKENS})
        scipy.io.wavfile.write(audio_path, rate=music["sampling_rate"], data=music["audio"])

    def generate_music_api(self, prompt, audio_path):
        headers =  {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
        payload = {
            "inputs": prompt
        }

        response = requests.post(config.MUSICGEN_MODEL_API_URL, headers=headers, json=payload)

        # ----ATTRIBUTION-START----
        # LLM: ChatGPT4o
        # PROMPT: please save the audio to a .wav file
        # EDITS: changed variables to match the code

        # Convert the byte content into an audio array
        audio_buffer = BytesIO(response.content)

        # Use scipy to save the audio, assuming it's a WAV format audio stream
        # If it's raw PCM audio, you would need to decode it first.
        with open(audio_path, "wb") as f:
            f.write(audio_buffer.read())
        # -----ATTRIBUTION-END-----

