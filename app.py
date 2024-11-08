# external imports
import gc
import logging as log
import time
import uuid
import gradio as gr
from prometheus_client import start_http_server, Counter, Histogram
import os


start_http_server(8000)

# local imports
from blip_image_caption_large import Blip_Image_Caption_Large
from phi3_mini_4k_instruct import Phi3_Mini_4k_Instruct
from musicgen_small import Musicgen_Small
import config

log.basicConfig(level=log.INFO)

# Prometheus metrics
REQUEST_COUNTER = Counter('app_requests_total', 'Total number of requests')
SUCCESSFUL_REQUESTS = Counter('app_successful_requests_total', 'Total number of successful requests')
FAILED_REQUESTS = Counter('app_failed_requests_total', 'Total number of failed requests')
TOTAL_REQUEST_DURATION = Histogram('app_total_request_duration_seconds', 'Time spent processing request', buckets=[5, 50, 100, 150, 500, 1000])
IMAGE_CAPTION_REQUEST_DURATION = Histogram('app_image_caption_request_duration_seconds', 'Time spent processing image caption request', buckets=[1, 2, 5, 10, 50, 100, 500])
MUSIC_DESCRIPTION_REQUEST_DURATION = Histogram('app_music_description_request_duration_seconds', 'Time spent processing music description request', buckets=[1, 2, 5, 10, 50, 100, 500])
MUSIC_GENERATION_REQUEST_DURATION = Histogram('app_music_generation_request_duration_seconds', 'Time spent processing music generation request', buckets=[1, 2, 5, 10, 50, 100, 500])

class Image_To_Music:
    def __init__(self, use_local_caption=False, use_local_llm=False, use_local_musicgen=False):

        self.use_local_llm = use_local_llm
        self.use_local_caption = use_local_caption
        self.use_local_musicgen = use_local_musicgen

        self.image_path = None
        self.generated_caption = None
        self.generated_description = None
        self.audio_path = config.AUDIO_DIR + str(uuid.uuid4()) + ".wav"

        self.caption_generation_duration = -1
        self.description_generation_duration = -1
        self.music_generation_duration = -1

    def caption_image(self, image_path):
        log.info("Captioning Image...")
        caption_start_time = time.time()

        # load model
        self.image_caption_model = Blip_Image_Caption_Large()

        self.image_path = image_path
        self.generated_caption = self.image_caption_model.caption_image(self.image_path, self.use_local_caption)

        # delete model to free up ram
        del self.image_caption_model
        gc.collect()

        self.caption_generation_duration = time.time() - caption_start_time
        log.info(f"Captioning Complete in {self.caption_generation_duration:.2f} seconds: {self.generated_caption} - used local model: {self.use_local_caption}")
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
        self.generated_description = self.text_generation_model.generate_text(messages, self.use_local_llm)

        # delete model to free up ram
        del self.text_generation_model
        gc.collect()

        self.description_generation_duration = time.time() - description_start_time
        log.info(f"Description Generation Complete in {self.description_generation_duration:.2f} seconds: {self.generated_description} - used local model: {self.use_local_llm}")
        return self.generated_description
    
    def generate_music(self):
        log.info("Generating Music...")
        music_start_time = time.time()
        
        # load model
        self.music_generation_model = Musicgen_Small()

        self.music_generation_model.generate_music(self.generated_description, self.audio_path, self.use_local_musicgen)
        
        # delete model to free up ram
        del self.music_generation_model
        gc.collect()

        self.music_generation_duration = time.time() - music_start_time
        log.info(f"Music Generation Complete in {self.music_generation_duration:.2f} seconds: {self.audio_path} - used local model: {self.use_local_musicgen}")
        return self.audio_path
    
    def get_durations(self):
            return f"Caption Generation Time: {self.caption_generation_duration:.2f} seconds\nDescription Generation Time: {self.description_generation_duration:.2f} seconds\nMusic Generation Time: {self.music_generation_duration:.2f} seconds\nTotal Time: {self.caption_generation_duration + self.description_generation_duration + self.music_generation_duration:.2f} seconds"


    def run(self, image_path):
        REQUEST_COUNTER.inc()
        with TOTAL_REQUEST_DURATION.time():
            with IMAGE_CAPTION_REQUEST_DURATION.time():
                self.caption_image(image_path)
            with MUSIC_DESCRIPTION_REQUEST_DURATION.time():
                self.generate_description()
            with MUSIC_GENERATION_REQUEST_DURATION.time():
                self.generate_music()   
        
        if os.path.isfile(self.audio_path) and os.path.getsize(self.audio_path) > 0:
            SUCCESSFUL_REQUESTS.inc()
        else:
            FAILED_REQUESTS.inc()
        return [self.generated_caption, self.generated_description, self.audio_path, self.get_durations()]


def run_image_to_music(image_path, llm_max_new_tokens, llm_temperature, llm_top_p, musicgen_max_seconds, use_local_caption, use_local_llm, use_local_musicgen):
    config.LLM_MAX_NEW_TOKENS = llm_max_new_tokens
    config.LLM_TEMPERATURE = llm_temperature
    config.LLM_TOP_P = llm_top_p
    config.MUSICGEN_MAX_NEW_TOKENS = musicgen_max_seconds * 51
    itm = Image_To_Music(use_local_caption=use_local_caption, use_local_llm=use_local_llm, use_local_musicgen=use_local_musicgen)
    return itm.run(image_path)

# Gradio UI 
def gradio():
    # Define Gradio Interface, information from (https://www.gradio.app/docs/chatinterface)
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'> ⛺ Image to Music Generator 🎼</h1>")
        image_input = gr.Image(type="filepath", label="Upload Image")


        # ----ATTRIBUTION-START----
        # LLM: ChatGPT4o
        # PROMPT: i need 3 checkbox fields that pass booleans to the run_image_to_music function. it should be  "Use local Image Captioning" "Use local LLM" "Use local Music Generation". please make it a nice parameter selector
        # EDITS: /

        # Checkbox parameters
        with gr.Row():
            local_captioning = gr.Checkbox(label="Use local Image Captioning", value=False)
            local_llm = gr.Checkbox(label="Use local LLM", value=False)
            local_music_gen = gr.Checkbox(label="Use local Music Generation", value=False)
        # -----ATTRIBUTION-END-----

        # ----ATTRIBUTION-START----
        # LLM: ChatGPT4o
        # PROMPT: Now, I need sliders for the different models that are used in the product:\n LLM_MAX_NEW_TOKENS = 50\nLLM_TEMPERATURE = 0.7\nLLM_TOP_P = 0.95\nMUSICGEN_MAX_NEW_TOKENS = 256 # 256 =  5 seconds of audio\n they should be in a hidden menu that opens when I click on "advanced options"\nPlease label them for the end user and fit them nicely in the following UI: <code>
        # EDITS: added interactive flags
        # Advanced options with sliders
        with gr.Accordion("Advanced Options", open=False):
            gr.Markdown("<h3>LLM Settings</h3>")
            llm_max_new_tokens = gr.Slider(1, 200, value=50, step=1, label="LLM Max Tokens", interactive=True)
            llm_temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="LLM Temperature", interactive=True)
            llm_top_p = gr.Slider(0.01, 0.99, value=0.95, step=0.01, label="LLM Top P", interactive=True)

            gr.Markdown("<h3>Music Generation Settings</h3>")
            musicgen_max_seconds = gr.Slider(1, 30, value=5, step=1, label="MusicGen Duration in Seconds (local model only)", interactive=True)
        # -----ATTRIBUTION-END-----

        with gr.Row():
            caption_output = gr.Textbox(label="Image Caption")
            music_description_output = gr.Textbox(label="Music Description")
            durations = gr.Textbox(label="Processing Times", interactive=False, placeholder="Time statistics will appear here")

        music_output = gr.Audio(label="Generated Music")
        # Button to trigger the process
        generate_button = gr.Button("Generate Music")
        generate_button.click(fn=run_image_to_music, inputs=[image_input, llm_max_new_tokens, llm_temperature, llm_top_p, musicgen_max_seconds, local_captioning, local_llm, local_music_gen], outputs=[caption_output, music_description_output, music_output, durations])
    # Launch Gradio app
    demo.launch(server_port=config.SERVICE_PORT, server_name=config.SERVER_NAME)

gradio()
