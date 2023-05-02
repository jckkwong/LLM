import gradio as gr
# from ChatGPT2_SQ_Controls import askQuestion
from ChatGPT2_SQC_with_SBERT import askQuestion
def ask(question):
    return askQuestion(question)

demo = gr.Interface(fn=ask, inputs="text", outputs="text")

demo.launch()