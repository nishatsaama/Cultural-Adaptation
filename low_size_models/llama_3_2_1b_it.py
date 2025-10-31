import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
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
