from musicgen_small import Musicgen_Small

import config
import os

# Test the local Musicgen_Small class with a 5 second music generation and assert file creation
def test_musicgen_small_local_model():
    musicgen_model = Musicgen_Small()
    prompt = "a very testy song, perfect to test the music generation model"
    audio_path = f"{config.AUDIO_DIR}/test_musicgen_small_local.wav"
    musicgen_model.generate_music(prompt, audio_path, use_local_musicgen=True)
    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 0
    os.remove(audio_path)
    assert not os.path.exists(audio_path)

# Test the Musicgen_Small API with a 30 second music generation and assert file creation
def test_musicgen_small_api():
    musicgen_model = Musicgen_Small()
    prompt = "a very testy song, perfect to test the music generation model"
    audio_path = f"{config.AUDIO_DIR}/test_musicgen_small_api.wav"
    musicgen_model.generate_music(prompt, audio_path, use_local_musicgen=False)
    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 0
    os.remove(audio_path)
    assert not os.path.exists(audio_path)