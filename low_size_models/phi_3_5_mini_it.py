import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def generate(content, temperature=0.7):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": content},
    ]

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": temperature,
        "do_sample": True,
        "use_cache": False,  # Fix for DynamicCache AttributeError
    }

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']
