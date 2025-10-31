from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
model_name = "bharatgenai/Param-1-2.9B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

def generate(content, temperature=0.7):
    conversation = [
        {
            "content": "You are helpful assistant.",
            "role": "system"
        },
        {
            "content": content,
            "role": "user"
        }
    ]

    inputs = tokenizer.apply_chat_template(
        conversation=conversation,
        return_tensors="pt",
        add_generation_prompt=True
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=300,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    # Get only the generated tokens (exclude the prompt length)
    generated_tokens = output[0][inputs.shape[-1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text
