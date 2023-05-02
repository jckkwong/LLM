import gradio as gr
import shutil

COPY_DIR = "C:/Users/James/PycharmProjects/LLM/Data/Other_Documents/DragAndDrop"

def copy_file(file):
    file_path = shutil.copy(file.name, COPY_DIR)
    return f"File copied successfully to {COPY_DIR} with name {file_path}!"

iface = gr.Interface(
    copy_file,
    inputs="file",
    outputs="text",
    title="Drag and Drop File Copier",
    description=f"Drag and drop a file to copy it to {COPY_DIR}.",
    allow_flagging=False,
    theme="compact"
)

if __name__ == "__main__":
    iface.launch()
    