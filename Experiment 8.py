import gradio as gr
from transformers import pipeline

# Load a pre-trained LLM from Hugging Face
chat_model = pipeline("text-generation", model="gpt2", max_length=200)

# Define the function to interact with the LLM
def chat_with_llm(user_input):
    try:
        # Generate a response
        response = chat_model(user_input, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Design the Gradio Blocks UI
with gr.Blocks() as demo:
    gr.Markdown("#Chat with LLM")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Enter your query:", placeholder="Type your message here...")
            submit_button = gr.Button("Submit")
        with gr.Column():
            response_output = gr.Textbox(label="LLM Response:", lines=10, interactive=False)
    
    # Add functionality
    submit_button.click(chat_with_llm, inputs=[user_input], outputs=[response_output])

# Run the application
demo.launch()
