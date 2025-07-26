import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import gradio as gr

processor = AutoProcessor.from_pretrained("microsoft/git-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption_git(image, mode):
    try:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        if mode == "Greedy (Fast, Deterministic)":
            output = model.generate(**inputs, max_new_tokens=30)
        else:  # Sampling (more diverse)
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                top_p=0.9,
                temperature=1.0
            )
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=generate_caption_git,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Radio(choices=["Greedy (Fast, Deterministic)", "Sampling (Creative)"],
                 value="Greedy (Fast, Deterministic)", label="Generation Mode")
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ GIT Image Caption Generator",
    description="Upload an image and generate a descriptive caption using Microsoft's GIT-Large model."
)

if __name__ == "__main__":
    demo.launch()
