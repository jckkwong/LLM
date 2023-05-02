import gradio as gr
import shutil
from ChatGPT2_SQC_with_SBERT import askQuestion, askQuestionNIST

theme = gr.themes.Default

COPY_DIR = "C:/Users/James/PycharmProjects/LLM/Data/Other_Documents/DragAndDrop"

def ask(question):
    return askQuestion(question)

def askNIST(questionNIST):
    return askQuestionNIST (questionNIST)

def copy_file(file):
    file_path = shutil.copy(file.name, COPY_DIR)
    return f"File copied successfully to your KNOWLEDGE REPOSITORY!"


with gr.Blocks(theme=theme) as demo:
    with gr.Tab("Question - Answer"):
        text_input = gr.Textbox(label='Question')
        text_button = gr.Button("Submit")
        text_output = gr.Textbox(label='Answer')
    with gr.Tab("NIST Query"):
        NIST_input = gr.Textbox(label='NIST Query')
        NIST_button = gr.Button("Submit")
        NIST_output = gr.Textbox(label='Answer')
    with gr.Tab("Knowledge Base"):
        with gr.Row():
            file_input = gr.File(label='File Store')
            file_output = gr.Textbox(label='Saved To')
        file_button = gr.Button("Save to Store in Knowledge Base")


    text_button.click (ask, inputs=text_input, outputs=text_output)
    NIST_button.click (askNIST, inputs=NIST_input, outputs=NIST_output)
    file_button.click (copy_file, inputs=file_input, outputs=file_output)




demo.launch()

