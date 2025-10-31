import pandas as pd
import argparse
import importlib
import os

# CLI arguments
parser = argparse.ArgumentParser(description='Cultural Adaptation - GSM8K to Indian Context')
parser.add_argument("--model_name", type=str, required=True, help="Model identifier")
parser.add_argument("--num_rows", type=int, default=10, help="Number of rows to process")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature value")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
args = parser.parse_args()

# Model mapping with directory info
MODEL_FILES = {
    # Low-size models (1B-3B)
    "llama-3.2-1b": ("low_size_models", "llama_3_2_1b_it"),
    "llama-3.2-3b": ("low_size_models", "llama_3_2_3b_it"),
    "gemma-2-2b": ("low_size_models", "gemma_2_2b_it"),
    "qwen-2.5-1.5b": ("low_size_models", "qwen_2_5_1_5b_it"),
    "qwen-2.5-3b": ("low_size_models", "qwen_2_5_3b_it"),
    "phi-3.5-mini": ("low_size_models", "phi_3_5_mini_it"),
    "granite": ("low_size_models", "granite_3_3_2b_it"),
    "param": ("low_size_models", "param_1_2_9b_it"),

    # Mid-size models (7B-9B)
    "llama-3.1-8b": ("mid_size_models", "llama_3_1_8b_it"),
    "qwen-2.5-7b": ("mid_size_models", "qwen_2_5_7b_it"),
    "mistral-7b": ("mid_size_models", "mistral_7b_it_v0_3"),
    "gemma-2-9b": ("mid_size_models", "gemma_2_9b_it"),

    # Large-size models (14B-32B)
    "phi-3-mid-14b": ("large_size_models", "phi_3_mid_14b_it"),
    "qwen-2.5-14b": ("large_size_models", "qwen_2_5_14b_it"),
    "qwen-2.5-32b": ("large_size_models", "qwen_2_5_32b_it"),
    "gemma-2-27b": ("large_size_models", "gemma_2_27b_it"),
    "sarvam-m-24b": ("large_size_models", "sarvam_m_24b_it"),
    "sarvam": ("large_size_models", "sarvam_m_24b_it"),
}

# Detect model file and directory
model_dir = None
model_file = None
for key, (directory, filename) in MODEL_FILES.items():
    if key in args.model_name.lower():
        model_dir = directory
        model_file = filename
        break

if model_file is None or model_dir is None:
    print(f"Model {args.model_name} not supported")
    exit(1)

print(f"Loading model file: {model_dir}/{model_file}")

# Import model-specific module
model_module = importlib.import_module(f"{model_dir}.{model_file}")

# Adaptation prompt
ADAPTATION_PROMPT = """Adapt this math word problem to Indian cultural context.

Instructions:
-Keep the same math.
-Output only the adapted problem text, no explanations.
-Do not solve the math problem.
-Output ONLY the rewritten problem in Indian context.

Original: {question}

Adapted problem:"""

# Load GSM8K dataset from local data folder
print("Loading GSM8K dataset...")
df = pd.read_csv("data/gsm8k_test_no_answers.csv")

print(f"Total rows in dataset: {len(df)}")
print(f"Processing {args.num_rows} rows...")

# Process rows
results = []
for i in range(min(args.num_rows, len(df))):
    question = df.iloc[i]['question']
    print(f"\nProcessing row {i+1}/{args.num_rows}")

    # Create prompt with question
    prompt = ADAPTATION_PROMPT.format(question=question)

    # Generate adapted question
    adapted_question = model_module.generate(prompt, temperature=args.temperature)

    # Store results
    results.append({
        "row_id": i,
        "original_question": question,
        "adapted_question": adapted_question,
        "model_used": args.model_name,
        "temperature": args.temperature
    })

    print(f"Completed row {i+1}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Save to CSV
output_filename = f"{args.output_dir}/{model_file}_gsm8k_adapted_{args.num_rows}rows_temp{args.temperature}.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_filename, index=False, encoding="utf-8")

print(f"\nResults saved to: {output_filename}")
print(f"Successfully processed {len(results)} rows")
