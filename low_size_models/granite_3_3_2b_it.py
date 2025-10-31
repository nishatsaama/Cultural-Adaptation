from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "ibm-granite/granite-3.3-2b-instruct"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate(content, temperature=0.7):
    conv = [{"role": "user", "content": content}]

    input_ids = tokenizer.apply_chat_template(
        conv,
        return_tensors="pt",
        thinking=True,
        return_dict=True,
        add_generation_prompt=True
    ).to(device)

    output = model.generate(
        **input_ids,
        max_new_tokens=300,
        temperature=temperature,
        do_sample=True,
    )

    prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prediction
