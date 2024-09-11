import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

def style_response(style, response):
    """Modify response style based on the selected style."""
    if style == "Nautical Marauder":
        response = response.replace("you", "ye").replace("hello", "ahoy").replace("friend", "matey")
        response = response.replace("is", "be").replace("my", "me").replace("the", "th'").replace("am", "be")
    elif style == "Elizabethan Prose":
        response = response.replace("you", "thou").replace("are", "art").replace("is", "be").replace("my", "mine")
        response = response.replace("your", "thy").replace("the", "thee").replace("has", "hath").replace("do", "doth")
    elif style == "Cyber Elite":
        response = response.replace("e", "3").replace("a", "4").replace("t", "7").replace("o", "0").replace("i", "1")
    elif style == "Slangify":
        response = response.replace("you", "ya").replace("are", "r").replace("hello", "hey").replace("friend", "buddy")
    return response

def get_magic_css():
    """Return magic-themed CSS."""
    return """
    body {
        background-color: #2e003e;
        font-family: 'Garamond', serif;
        color: #f0e6f6;
        background-image: url('https://www.transparenttextures.com/patterns/stardust.png');
        background-size: cover;
        background-blur: 10px;
        color: #f0e6f6;
    }
    .gradio-container {
        background: rgba(50, 50, 50, 0.8);
        border: 2px solid #9b59b6;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
    }
    .gr-chat {
        font-size: 18px;
        color: #dcdbe1;
    }
    .gr-button {
        background-color: #9b59b6;
        border: none;
        color: #fff;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .gr-button:hover {
        background-color: #8e44ad;
    }
    """

def respond(message, history: list[tuple[str, str]], style="Standard Conversational"):
    global stop_inference
    stop_inference = False  # Reset cancellation flag

    # Initialize history if it's None
    if history is None:
        history = []

    # API-based inference
    messages = [{"role": "user", "content": message}]

    response = ""
    for message_chunk in client.chat_completion(
        messages,
        max_tokens=512,  # Default max tokens for response
        stream=True,
        temperature=0.7,  # Default temperature
        top_p=0.95,  # Default top-p
    ):
        if stop_inference:
            response = "Inference cancelled."
            yield history + [(message, response)]
            return
        token = message_chunk.choices[0].delta.content
        response += token
        yield history + [(message, style_response(style, response))]  # Apply selected style to the response


def cancel_inference():
    global stop_inference
    stop_inference = True

def clear_input():
    """Function to clear the user input after submission."""
    return ""

# Define the interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🔮 StyleChat 🔮</h1>")
    gr.Markdown("Please select the style you would like to talk to the AI in.")
    
    # Add style selection at the top
    with gr.Row():
        style_selection = gr.Dropdown(
            label="Response Style", 
            choices=["Standard Conversational", "Nautical Marauder", "Elizabethan Prose", "Cyber Elite", "Slangify"], 
            value="Standard Conversational"
        )

    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

    cancel_button = gr.Button("Cancel Inference", variant="danger")

    # Apply fixed magic-themed CSS
    demo.css = get_magic_css()

    # Submit handler to clear input after submission
    def handle_submit(message, history, style):
        response_generator = respond(message, history, style)
        clear_input()
        return response_generator

    user_input.submit(handle_submit, [user_input, chat_history, style_selection], chat_history)
    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces
