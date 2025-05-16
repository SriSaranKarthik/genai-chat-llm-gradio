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
Name: R Guruprasad
Register No: 212222240033
```

```py
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
    gr.Markdown("Chat with LLM")
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

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/79c36ada-cb9c-4a3c-9452-95f2ce847668)


### RESULT:
The "Chat with LLM" application was successfully designed and deployed using the Gradio Blocks framework, allowing seamless user interaction with a large language model.

