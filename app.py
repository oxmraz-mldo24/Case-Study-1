# external imports
import gc
import logging as log
import time
import uuid
import gradio as gr
import os

# local imports
from blip_image_caption_large import Blip_Image_Caption_Large
from phi3_mini_4k_instruct import Phi3_Mini_4k_Instruct
from musicgen_small import Musicgen_Small
import config

log.basicConfig(level=log.INFO)


class Image_To_Music:
    def __init__(self):
        self.image_path = None
        self.generated_caption = None
        self.generated_description = None
        self.audio_path = config.AUDIO_DIR + str(uuid.uuid4()) + ".wav"

        self.caption_generation_duration = -1
        self.description_generation_duration = -1
        self.music_generation_duration = -1
        self.create_output_folder()


    # ----ATTRIBUTION-START----
    # LLM: Github Copilot
    # PROMPT: create an output folder for the generated audio files
    # EDITS: /
    def create_output_folder(self):
        os.makedirs(config.AUDIO_DIR, exist_ok=True)
    # -----ATTRIBUTION-END-----

    def caption_image(self, image_path):
        log.info("Captioning Image...")
        caption_start_time = time.time()

        # load model
        self.image_caption_model = Blip_Image_Caption_Large()

        self.image_path = image_path
        self.generated_caption = self.image_caption_model.caption_image_local_pipeline(self.image_path)[0]["generated_text"]

        # delete model to free up ram
        del self.image_caption_model
        gc.collect()

        self.caption_generation_duration = time.time() - caption_start_time
        log.info(f"Captioning Complete in {self.caption_generation_duration:.2f} seconds: {self.generated_caption}")
        return self.generated_caption
    
    def generate_description(self):
        log.info("Generating Music Description...")
        description_start_time = time.time()

        # load model
        self.text_generation_model = Phi3_Mini_4k_Instruct()

        messages = [
            {"role": "system", "content": "You are an image caption to song description converter with a deep understanding of Music and Art. You are given the caption of an image. Your task is to generate a textual description of a musical piece that fits the caption. The description should be detailed and vivid, and should include the genre, mood, instruments, tempo, and other relevant information about the music. You should also use your knowledge of art and visual aesthetics to create a musical piece that complements the image. Only output the description of the music, without any explanation or introduction. Be concise."},
            {"role": "user", "content": self.generated_caption},
        ]
        self.generated_description = self.text_generation_model.generate_text_local_pipeline(messages)[-1]['generated_text'][-1]['content']

        # delete model to free up ram
        del self.text_generation_model
        gc.collect()

        self.description_generation_duration = time.time() - description_start_time
        log.info(f"Description Generation Complete in {self.description_generation_duration:.2f} seconds: {self.generated_description}")
        return self.generated_description
    
    def generate_music(self):
        log.info("Generating Music...")
        music_start_time = time.time()
        
        # load model
        self.music_generation_model = Musicgen_Small()

        self.music_generation_model.generate_music_local_pipeline(self.generated_description, self.audio_path)
        
        # delete model to free up ram
        del self.music_generation_model
        gc.collect()

        self.music_generation_duration = time.time() - music_start_time
        log.info(f"Music Generation Complete in {self.music_generation_duration:.2f} seconds: {self.audio_path}")
        return self.audio_path
    
    def get_durations(self):
            return f"Caption Generation Time: {self.caption_generation_duration:.2f} seconds\nDescription Generation Time: {self.description_generation_duration:.2f} seconds\nMusic Generation Time: {self.music_generation_duration:.2f} seconds\nTotal Time: {self.caption_generation_duration + self.description_generation_duration + self.music_generation_duration:.2f} seconds"

    def run_yield(self, image_path):

        self.caption_image(image_path)
        yield [self.generated_caption, None, None, None]
        self.generate_description()
        yield [self.generated_caption, self.generated_description, None, None]
        self.generate_music()
        yield [self.generated_caption, self.generated_description, self.audio_path, None]
        return [self.generated_caption, self.generated_description, self.audio_path,self.get_durations()]
    
    def run(self, image_path):
        self.caption_image(image_path)
        self.generate_description()
        self.generate_music()
        return [self.generated_caption, self.generated_description, self.audio_path, self.get_durations()]


# Gradio UI 
def gradio():
    # Define Gradio Interface, information from (https://www.gradio.app/docs/chatinterface)
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'> ⛺ Image to Music Generator 🎼</h1>")
        image_input = gr.Image(type="filepath", label="Upload Image")
        with gr.Row():
            caption_output = gr.Textbox(label="Image Caption")
            music_description_output = gr.Textbox(label="Music Description")
            durations = gr.Textbox(label="Processing Times", interactive=False, placeholder="Time statistics will appear here")

        music_output = gr.Audio(label="Generated Music")
        # Button to trigger the process
        generate_button = gr.Button("Generate Music")
        itm = Image_To_Music()
        generate_button.click(fn=itm.run, inputs=image_input, outputs=[caption_output, music_description_output, music_output, durations])
    # Launch Gradio app
    demo.launch()

gradio()
