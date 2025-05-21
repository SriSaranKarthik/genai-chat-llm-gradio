## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
Building a user-friendly application that allows seamless interaction with a large language model (LLM) is challenging without requiring specialized API keys or external resources. This project addresses the need for an accessible, open-source solution to implement such applications using pre-trained models and the Gradio Blocks framework.

### DESIGN STEPS:

### **STEP 1: Import Required Libraries**
- Install and import the necessary libraries:
  - **Gradio** for the UI.
  - **Transformers** for using pre-trained models.

### **STEP 2: Load a Pre-Trained Model**
- Use a pre-trained model like **GPT-2** or **DialoGPT** from Hugging Face.
- Initialize the model using the `pipeline` API for straightforward interaction.

### **STEP 3: Define Application Workflow**
- Create a function to:
  1. Process user input.
  2. Pass it to the language model.
  3. Return the generated response.
  
- Design the user interface using Gradio Blocks, adding components for input, output, and interactivity.

### **STEP 4: Deploy the Application**
- Run the application locally with `demo.launch()`.
- Optionally deploy it to the cloud for broader accessibility.
---

### PROGRAM:
```
Name: K.SRISARAN KARTHIK
Register No: 212224230275
```

```py
# === Import Required Libraries ===
import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
import json
import random

from dotenv import load_dotenv, find_dotenv
from text_generation import Client
import gradio as gr

# === Load environment variables ===
requests.adapters.DEFAULT_TIMEOUT = 60
_ = load_dotenv(find_dotenv())  # read local .env file

# === Set up HuggingFace API keys ===
hf_api_key = os.environ['HF_API_KEY']
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)

# === Test the client with a basic prompt ===
prompt = "Thoughts about Generative AI"
client.generate(prompt, max_new_tokens=256).generated_text

# === Simple Gradio App for Prompt Completion ===
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

gr.close_all()
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"), 
        gr.Slider(label="Max new tokens", value=20, maximum=1024, minimum=1)
    ],
    outputs=[gr.Textbox(label="Completion")]
)
demo.launch()


# === Chatbot without LLM, using random pre-defined messages ===
def respond(message, chat_history):
    bot_message = random.choice([
        "That sounds really interesting! I’d love to hear more, could you dive a little deeper into it for me?",
        "I really appreciate you sharing that, it’s definitely cool! It might not be for me right now, but I respect the effort behind it", 
        "Alright, I get where you’re coming from. Let’s keep the conversation going—I'm curious to see where this leads!"
    ])
    chat_history.append((message, bot_message))
    return "", chat_history

gr.close_all()
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch()


# === Chatbot with LLM prompt formatting ===
def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
    formatted_prompt = format_chat_prompt(message, chat_history)
    bot_message = client.generate(formatted_prompt, max_new_tokens=1024, stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
    chat_history.append((message, bot_message))
    return "", chat_history

gr.close_all()
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch(share=True, server_port=int(os.environ['PORT3']))


# === Final Chatbot with streaming, temperature and system instruction ===
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt, max_new_tokens=1024, stop_sequences=["\nUser:", "<|endoftext|>"], temperature=temperature)

    acc_text = ""
    for idx, response in enumerate(stream):
        text_token = response.token.text

        if response.details:
            return

        if idx == 0 and text_token.startswith(" "):
            text_token = text_token[1:]

        acc_text += text_token
        last_turn = list(chat_history.pop(-1))
        last_turn[-1] += acc_text
        chat_history = chat_history + [last_turn]
        yield "", chat_history
        acc_text = ""

gr.close_all()
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)
    msg = gr.Textbox(label="Prompt")
    
    with gr.Accordion(label="Advanced options", open=False):
        system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system, temperature], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system, temperature], outputs=[msg, chatbot])

demo.queue().launch(share=True, server_port=int(os.environ['PORT4']))
gr.close_all()

```
### OUTPUTS:
# 1.
![Exper 8 Output 1(1)](https://github.com/user-attachments/assets/4d2ad848-d882-460d-a8c2-1856cb3d8976)
![exper 8 output1(2)](https://github.com/user-attachments/assets/d999a65c-b8c3-4402-94e0-1595ce3d1280)
# 2.
![exper 8 output 2](https://github.com/user-attachments/assets/ee876e1c-ff57-4427-bfe9-c61ac65ff9ce)
# 3.
![exper 8 output 3](https://github.com/user-attachments/assets/c8151744-4ee8-41d7-b852-8a09a62b67da)
# 4.
![exper 8 output 4](https://github.com/user-attachments/assets/0f2e10d6-a72f-41ff-a149-21d2958b4e70)

### RESULT:
The "Chat with LLM" application was successfully designed and deployed using the Gradio Blocks framework, allowing seamless user interaction with a large language model.

