import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def generate(content, temperature=0.7):
    messages = [
        {"role": "user", "content": content},
    ]
    outputs = pipe(messages, max_new_tokens=256, temperature=temperature, do_sample=True)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response
