import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate(content, temperature=0.7):
    messages = [
        {"role": "user", "content": content},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=300,
        temperature=temperature,
        do_sample=True,
    )
    result = outputs[0]["generated_text"][-1]
    if isinstance(result, dict):
        return result.get("content", str(result))
    return str(result)
