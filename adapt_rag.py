import pandas as pd
import numpy as np
import argparse
import importlib
import os
import pickle
import re
import torch
from sentence_transformers import SentenceTransformer, util

# CLI arguments
parser = argparse.ArgumentParser(description='RAG-Enhanced Cultural Adaptation - GSM8K to Indian Context')
parser.add_argument("--model_name", type=str, required=True, help="Model identifier")
parser.add_argument("--num_rows", type=int, default=10, help="Number of rows to process")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature value")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
parser.add_argument("--top_k", type=int, default=5, help="Number of cultural items to retrieve")
parser.add_argument("--use_facet", type=str, default=None, help="Filter by facet (e.g., textiles, handicrafts)")
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

# Load Diwali index
print("\nLoading Diwali cultural knowledge index...")
if not os.path.exists("data/diwali_index.pkl"):
    print("ERROR: Diwali index not found!")
    print("Please run: python build_diwali_index.py")
    exit(1)

with open("data/diwali_index.pkl", "rb") as f:
    index_data = pickle.load(f)

diwali_embeddings = index_data['embeddings']
diwali_df = index_data['dataframe']
concepts_by_facet = index_data['concepts_by_facet']
print(f"  Loaded {len(diwali_df)} cultural entries")
print(f"  Available facets: {list(concepts_by_facet.keys())}")

# Load cultural reference
with open("data/cultural_reference.pkl", "rb") as f:
    cultural_ref = pickle.load(f)

print(f"  Cultural references loaded: {sum(len(v) for v in cultural_ref.values() if isinstance(v, list))} items")

# Load models for retrieval
print("\nLoading models for retrieval...")
print("  - Loading sentence transformer...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Move Diwali embeddings to same device as embedding model
device = embedding_model.device
print(f"  - Moving Diwali embeddings to device: {device}")
if not isinstance(diwali_embeddings, torch.Tensor):
    diwali_embeddings = torch.tensor(diwali_embeddings, dtype=torch.float32)
diwali_embeddings = diwali_embeddings.to(device)

print("  - Loading spaCy NER model...")
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_query_terms(question):
    """
    Generalized query extraction using NER and semantic understanding.
    Extracts entities and nouns from the question to build a semantic query.
    """
    # Parse question with spaCy
    doc = nlp(question)

    query_components = []

    # 1. Extract named entities (PERSON, GPE, ORG, PRODUCT, etc.)
    entities = []
    for ent in doc.ents:
        # Skip numeric entities and dates
        if ent.label_ not in ['CARDINAL', 'ORDINAL', 'MONEY', 'DATE', 'TIME', 'PERCENT', 'QUANTITY']:
            entities.append(ent.text)

    # 2. Extract nouns (objects, items, concepts)
    nouns = []
    for token in doc:
        if token.pos_ == 'NOUN' and not token.is_stop:
            nouns.append(token.text)

    # 3. Extract verbs to understand context (buying, selling, making, etc.)
    verbs = []
    for token in doc:
        if token.pos_ == 'VERB' and not token.is_stop:
            verbs.append(token.lemma_)

    # Build query from extracted components
    if entities:
        query_components.append(' '.join(entities))

    if nouns:
        # Take top nouns (most frequent or all if few)
        unique_nouns = list(set(nouns))[:5]
        query_components.append(' '.join(unique_nouns))

    if verbs:
        # Add verb context for better semantic matching
        unique_verbs = list(set(verbs))[:3]
        query_components.append(' '.join(unique_verbs))

    # Combine all components
    if query_components:
        query = ' '.join(query_components)
    else:
        # Fallback: use the question itself (semantic search will handle it)
        query = question

    return query

def retrieve_cultural_context(question, top_k=5, facet_filter=None):
    """Retrieve relevant cultural items from Diwali dataset"""

    # Extract query terms
    query = extract_query_terms(question)

    # Create query embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # Filter by facet if specified
    if facet_filter and facet_filter in concepts_by_facet:
        facet_indices = diwali_df[diwali_df['facet'] == facet_filter].index.tolist()
        filtered_embeddings = diwali_embeddings[facet_indices]
        filtered_df = diwali_df.iloc[facet_indices]
    else:
        filtered_embeddings = diwali_embeddings
        filtered_df = diwali_df

    # Compute similarities
    similarities = util.cos_sim(query_embedding, filtered_embeddings)[0]

    # Get top-k indices (convert to CPU and numpy for pandas compatibility)
    top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()

    # Retrieve items
    retrieved_items = []
    for idx in top_indices:
        item = filtered_df.iloc[int(idx)]
        retrieved_items.append({
            'concept': item['concept'],
            'state': item['state'],
            'facet': item['facet'],
            'description': item['description'],
            'similarity': similarities[idx].item()
        })

    return retrieved_items, query

def format_cultural_context(retrieved_items, cultural_ref):
    """Format retrieved cultural context for prompt using Diwali dataset concepts"""

    context_parts = []

    # Add retrieved specific items (most relevant from semantic search)
    if retrieved_items:
        concepts = [item['concept'] for item in retrieved_items]
        context_parts.append(f"Relevant cultural items: {', '.join(concepts[:3])}")

    # Sample from actual Diwali dataset concepts
    # Names from Diwali dataset
    if cultural_ref['names']:
        names_sample = np.random.choice(
            cultural_ref['names'],
            size=min(5, len(cultural_ref['names'])),
            replace=False
        )
        context_parts.append(f"Indian names: {', '.join(names_sample)}")

    # Places from Diwali dataset
    if cultural_ref['places']:
        places_sample = np.random.choice(
            cultural_ref['places'],
            size=min(3, len(cultural_ref['places'])),
            replace=False
        )
        context_parts.append(f"Indian places: {', '.join(places_sample)}")

    # Foods from Diwali dataset
    if cultural_ref['foods']:
        foods_sample = np.random.choice(
            cultural_ref['foods'],
            size=min(4, len(cultural_ref['foods'])),
            replace=False
        )
        context_parts.append(f"Indian foods: {', '.join(foods_sample)}")

    # Drinks from Diwali dataset
    if cultural_ref.get('drinks'):
        drinks_sample = np.random.choice(
            cultural_ref['drinks'],
            size=min(2, len(cultural_ref['drinks'])),
            replace=False
        )
        context_parts.append(f"Indian drinks: {', '.join(drinks_sample)}")

    # Clothing/Textiles from Diwali dataset
    if cultural_ref.get('clothing'):
        clothing_sample = np.random.choice(
            cultural_ref['clothing'],
            size=min(2, len(cultural_ref['clothing'])),
            replace=False
        )
        context_parts.append(f"Indian clothing: {', '.join(clothing_sample)}")

    # Festivals from Diwali dataset
    if cultural_ref.get('festivals'):
        festivals_sample = np.random.choice(
            cultural_ref['festivals'],
            size=min(2, len(cultural_ref['festivals'])),
            replace=False
        )
        context_parts.append(f"Indian festivals: {', '.join(festivals_sample)}")

    # Add currency
    context_parts.append("Currency: Indian Rupees (₹)")

    return "\n".join(context_parts)

# RAG-Enhanced Adaptation Prompt
RAG_ADAPTATION_PROMPT = """Adapt this math word problem to Indian cultural context using the provided cultural references.

Cultural References Available:
{cultural_context}

Instructions:
- Keep the same math and numerical relationships
- Replace names, places, items with Indian alternatives from the references above
- Convert currency to Indian Rupees (₹)
- Make the context feel authentically Indian
- Output ONLY the adapted problem text, no explanations
- Do not solve the math problem

Original Problem: {question}

Adapted Problem:"""

# Load GSM8K dataset from local data folder
print("\nLoading GSM8K dataset...")
df = pd.read_csv("data/gsm8k_test_no_answers.csv")

print(f"Total rows in dataset: {len(df)}")
print(f"Processing {args.num_rows} rows...")
print(f"Retrieving top-{args.top_k} cultural items per question")
if args.use_facet:
    print(f"Filtering by facet: {args.use_facet}")

# Process rows
results = []
for i in range(min(args.num_rows, len(df))):
    question = df.iloc[i]['question']
    print(f"\nProcessing row {i+1}/{args.num_rows}")
    print(f"  Original: {question[:80]}...")

    # Retrieve cultural context
    retrieved_items, query_used = retrieve_cultural_context(
        question,
        top_k=args.top_k,
        facet_filter=args.use_facet
    )

    print(f"  Query: {query_used}")
    print(f"  Retrieved: {', '.join([item['concept'] for item in retrieved_items[:3]])}")

    # Format cultural context
    cultural_context = format_cultural_context(retrieved_items, cultural_ref)

    # Create RAG-enhanced prompt
    prompt = RAG_ADAPTATION_PROMPT.format(
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
        "retrieved_items": " | ".join([item['concept'] for item in retrieved_items]),
        "query_used": query_used,
        "model_used": args.model_name,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "facet_filter": args.use_facet or "none"
    })

    print(f"  Adapted: {adapted_question[:80]}...")
    print(f"  Completed row {i+1}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Save to CSV
facet_suffix = f"_{args.use_facet}" if args.use_facet else ""
output_filename = f"{args.output_dir}/{model_file}_rag{facet_suffix}_k{args.top_k}_{args.num_rows}rows_temp{args.temperature}.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_filename, index=False, encoding="utf-8")

print(f"\n{'='*80}")
print("PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {output_filename}")
print(f"Successfully processed {len(results)} rows")
print(f"Average cultural items retrieved: {args.top_k} per question")
print(f"{'='*80}")
