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
    return response

def get_css(style):
    """Return corresponding CSS based on the selected style."""
    if style == "Nautical Marauder":
        return """
        body {
            background-color: #2b2b2b;
            font-family: 'Trebuchet MS', sans-serif;
            color: #f4e9c9;
            background-image: url('https://www.transparenttextures.com/patterns/old-map.png');
        }
        .gradio-container {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #d4af37;
            box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
        }
        .gr-chat {
            font-size: 16px;
            color: #f4e9c9;
        }
        """
    elif style == "Elizabethan Prose":
        return """
        body {
            background-color: #f5f0e1;
            font-family: 'Dancing Script', cursive;
            color: #5c4033;
            background-image: url('https://www.transparenttextures.com/patterns/old-paper.png');
        }
        .gradio-container {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid #a0522d;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .gr-chat {
            font-size: 18px;
            color: #5c4033;
        }
        """
    elif style == "Cyber Elite":
        return """
        body {
            background-color: #000000;
            font-family: 'Courier New', Courier, monospace;
            color: #00ff00;
        }
        .gradio-container {
            background: #1a1a1a;
            border: 2px solid #00ff00;
            box-shadow: 0 4px 8px rgba(0, 255, 0, 0.3);
        }
        .gr-chat {
            font-size: 16px;
            color: #00ff00;
        }
        """
    else:
        # Default style
        return """
        body {
            background-color: #f0f0f0;
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        .gradio-container {
            background: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        .gr-chat {
            font-size: 16px;
            color: #333;
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

# Define the interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🌟 Fancy AI Chatbot 🌟</h1>")
    gr.Markdown("Interact with the AI chatbot using customizable settings below.")

    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

    cancel_button = gr.Button("Cancel Inference", variant="danger")

    # New feature: Style selection with more formal names
    style_selection = gr.Dropdown(
        label="Response Style", 
        choices=["Standard Conversational", "Nautical Marauder", "Elizabethan Prose", "Cyber Elite"], 
        value="Standard Conversational"
    )

    def apply_css(style):
        return get_css(style)

    # Adjusted to ensure history is maintained and passed correctly
    user_input.submit(respond, [user_input, chat_history, style_selection], chat_history)
    style_selection.change(apply_css, style_selection, gr.CSS())
    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces
