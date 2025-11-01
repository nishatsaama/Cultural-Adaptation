import pandas as pd
import numpy as np
import argparse
import importlib
import os
import pickle

# CLI arguments
parser = argparse.ArgumentParser(description='Diwali-Enhanced Cultural Adaptation - GSM8K to Indian Context')
parser.add_argument("--model_name", type=str, required=True, help="Model identifier")
parser.add_argument("--num_rows", type=int, default=10, help="Number of rows to process")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature value")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
parser.add_argument("--num_concepts", type=int, default=25, help="Total number of concepts to sample")
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
    "granite": ("low_size_models", "granite_3_0_2b_it"),
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

# Load Diwali cultural references
print("\nLoading Diwali cultural references...")
if not os.path.exists("data/cultural_reference.pkl"):
    print("ERROR: Cultural reference file not found!")
    print("Please run: python build_diwali_index.py")
    exit(1)

with open("data/cultural_reference.pkl", "rb") as f:
    cultural_ref = pickle.load(f)

# Select appropriate facets and sample concepts
print("Sampling cultural concepts from Diwali dataset...")

# Define facets to use and how many concepts from each
FACET_DISTRIBUTION = {
    'names': 15,      # Person names
    'places': 15,     # Indian locations
    'foods': 15,      # Food items
    'drinks': 15,     # Beverages
    'festivals': 15,  # Festivals/celebrations
    'clothing': 15,   # Traditional clothing
    'games': 15,      # Traditional games/activities
}

# Adjust if user specified different total
total_requested = args.num_concepts
total_default = sum(FACET_DISTRIBUTION.values())
if total_requested != total_default:
    # Scale proportionally
    scale = total_requested / total_default
    FACET_DISTRIBUTION = {k: max(1, int(v * scale)) for k, v in FACET_DISTRIBUTION.items()}

# Sample concepts from each facet
sampled_concepts = {}
for facet, count in FACET_DISTRIBUTION.items():
    facet_data = cultural_ref.get(facet, [])
    if len(facet_data) > 0:
        sample_size = min(count, len(facet_data))
        sampled = np.random.choice(facet_data, size=sample_size, replace=False).tolist()
        sampled_concepts[facet] = sampled
        print(f"  - Sampled {len(sampled)} from {facet} (available: {len(facet_data)})")
    else:
        sampled_concepts[facet] = []
        print(f"  - Skipped {facet} (not available)")

# Build cultural context for prompt
cultural_context_parts = []
if sampled_concepts.get('names'):
    cultural_context_parts.append(f"Names: {', '.join(sampled_concepts['names'])}")
if sampled_concepts.get('places'):
    cultural_context_parts.append(f"Places: {', '.join(sampled_concepts['places'])}")
if sampled_concepts.get('foods'):
    cultural_context_parts.append(f"Foods: {', '.join(sampled_concepts['foods'])}")
if sampled_concepts.get('drinks'):
    cultural_context_parts.append(f"Drinks: {', '.join(sampled_concepts['drinks'])}")
if sampled_concepts.get('festivals'):
    cultural_context_parts.append(f"Festivals: {', '.join(sampled_concepts['festivals'])}")
if sampled_concepts.get('clothing'):
    cultural_context_parts.append(f"Clothing: {', '.join(sampled_concepts['clothing'])}")
if sampled_concepts.get('games'):
    cultural_context_parts.append(f"Games/Activities: {', '.join(sampled_concepts['games'])}")

cultural_context_parts.append("Currency: Indian Rupees (₹)")

cultural_context = "\n".join(cultural_context_parts)

print(f"\nCultural context prepared with {sum(len(v) for v in sampled_concepts.values())} concepts")

# Adaptation prompt with cultural context
ADAPTATION_PROMPT = """Adapt this math word problem to Indian cultural context using the following cultural elements.

Available Indian Cultural Elements(use these):
{cultural_context}

Instructions:
- MUST use the cultural elements provided above where appropriate.
- Keep the same math
- Use the cultural elements provided above where appropriate
- Output only the adapted problem text, no explanations
- Do not solve the math problem
- Output ONLY the rewritten problem in Indian context

Original: {question}

Adapted problem:"""

# Load GSM8K dataset from local data folder
print("\nLoading GSM8K dataset...")
df = pd.read_csv("data/gsm8k_test_no_answers.csv")

print(f"Total rows in dataset: {len(df)}")
print(f"Processing {args.num_rows} rows...")

# Process rows
results = []
for i in range(min(args.num_rows, len(df))):
    question = df.iloc[i]['question']
    print(f"\nProcessing row {i+1}/{args.num_rows}")

    # Create prompt with cultural context and question
    prompt = ADAPTATION_PROMPT.format(
        cultural_context=cultural_context,
        question=question
    )

    # Generate adapted question
    adapted_question = model_module.generate(prompt, temperature=args.temperature)

    # Store results
    results.append({
        "row_id": i,
        "original_question": question,
        "adapted_question": adapted_question,
        "cultural_concepts_used": str(sampled_concepts),
        "model_used": args.model_name,
        "temperature": args.temperature
    })

    print(f"Completed row {i+1}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Save to CSV
output_filename = f"{args.output_dir}/{model_file}_diwali_{args.num_rows}rows_temp{args.temperature}.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_filename, index=False, encoding="utf-8")

print(f"\n{'='*80}")
print("PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {output_filename}")
print(f"Successfully processed {len(results)} rows")
print(f"Cultural concepts used: {sum(len(v) for v in sampled_concepts.values())}")
print(f"{'='*80}")
