# external imports
import time
import gradio as gr

# local imports
from blip_image_caption_large import Blip_Image_Caption_Large
from phi3_mini_4k_instruct import Phi3_Mini_4k_Instruct
from musicgen_small import Musicgen_Small

#image_to_music function
def image_to_music(image_path):
    # test image captioning
    image_caption_start_time = time.time()
    image_caption_model = Blip_Image_Caption_Large()
    
    test_caption = image_caption_model.caption_image_local_pipeline(image_path)
    
    print(test_caption)
    
    image_caption_end_time = time.time()

    # test text generation
    text_generation_start_time = time.time()
    text_generation_model = Phi3_Mini_4k_Instruct()
    
    #TODO: move this to a config file
    text_generation_model.local_pipeline.model.config.max_new_tokens = 200


    #TODO: move system prompt somewhere else, allow for genre override
    messages = [
    {"role": "system", "content": "You are an image caption to song description converter with a deep understanding of Music and Art. You are given the caption of an image. Your task is to generate a textual description of a musical piece that fits the caption. The description should be detailed and vivid, and should include the genre, mood, instruments, tempo, and other relevant information about the music. You should also use your knowledge of art and visual aesthetics to create a musical piece that complements the image. Only output the description of the music, without any explanation or introduction. Be concise."},
    {"role": "user", "content": test_caption[0]["generated_text"]},
    ]
    test_text = text_generation_model.generate_text_local_pipeline(messages)
    print(test_text)
    text_generation_end_time = time.time()
    

    # test audio generation
    music_generation_start_time = time.time()
    music_generation_model = Musicgen_Small()
    music_generation_model.generate_music_local_pipeline(str(test_text[-1]['generated_text'][-1]['content']))
    music_generation_end_time = time.time()


    # calculate durations
    image_caption_duration = image_caption_end_time - image_caption_start_time
    text_generation_duration = text_generation_end_time - text_generation_start_time
    music_generation_duration = music_generation_end_time - music_generation_start_time
    total_duration = music_generation_end_time - image_caption_start_time

    # output generated_text, audio and duration to gradio
    return (test_caption[0]["generated_text"], test_text[-1]['generated_text'][-1]['content'], "data/musicgen_out.wav",
            f"Image Captioning Duration: {image_caption_duration} sec",
            f"Text Generation Duration: {text_generation_duration} sec",
            f"Music Generation Duration: {music_generation_duration} sec",
            f"Total Duration: {total_duration} sec")

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

        generate_button.click(fn=image_to_music, inputs=[image_input], outputs=[caption_output, music_description_output, music_output, durations])

    # Launch Gradio app
    demo.launch()

gradio()
