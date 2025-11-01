import pandas as pd
import numpy as np
import argparse
import os
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# For semantic similarity
from sentence_transformers import SentenceTransformer, util

# For NER
import spacy

# CLI arguments
parser = argparse.ArgumentParser(description='Evaluate Cultural Adaptation Generations')
parser.add_argument("--input_folder", type=str, required=True, help="Folder containing CSV files")
parser.add_argument("--output_file", type=str, required=True, help="Output file for evaluation results")
args = parser.parse_args()

# Load models
print("Loading evaluation models...")
print("  - Loading sentence transformer...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("  - Loading spaCy NER model...")
nlp = spacy.load('en_core_web_sm')

# Load Diwali cultural references (no hardcoding!)
print("  - Loading Diwali cultural references...")
import pickle

if not os.path.exists("data/cultural_reference.pkl"):
    print("ERROR: Cultural reference file not found!")
    print("Please run: python build_diwali_index.py")
    exit(1)

with open("data/cultural_reference.pkl", "rb") as f:
    cultural_ref = pickle.load(f)

# Extract cultural elements from Diwali dataset
INDIAN_NAMES = [name.lower() for name in cultural_ref.get('names', [])]
INDIAN_PLACES = [place.lower() for place in cultural_ref.get('places', [])]
INDIAN_FOODS = [food.lower() for food in cultural_ref.get('foods', [])]
INDIAN_DRINKS = [drink.lower() for drink in cultural_ref.get('drinks', [])]
INDIAN_FESTIVALS = [fest.lower() for fest in cultural_ref.get('festivals', [])]
INDIAN_CLOTHING = [cloth.lower() for cloth in cultural_ref.get('clothing', [])]
INDIAN_TEXTILES = [textile.lower() for textile in cultural_ref.get('textiles', [])]
INDIAN_DANCES = [dance.lower() for dance in cultural_ref.get('dances', [])]
INDIAN_ARTS = [art.lower() for art in cultural_ref.get('arts', [])]
INDIAN_GAMES = [game.lower() for game in cultural_ref.get('games', [])]

# Combine all cultural items (foods, drinks, clothing, etc.)
INDIAN_ITEMS = (INDIAN_FOODS + INDIAN_DRINKS + INDIAN_CLOTHING +
                INDIAN_TEXTILES + INDIAN_DANCES + INDIAN_ARTS + INDIAN_GAMES +
                ['rupee', 'rupees', 'rs', '₹', 'inr'])  # Add currency terms

print(f"  - Loaded {len(INDIAN_NAMES)} Indian names from Diwali dataset")
print(f"  - Loaded {len(INDIAN_PLACES)} Indian places from Diwali dataset")
print(f"  - Loaded {len(INDIAN_ITEMS)} Indian cultural items from Diwali dataset")
print(f"  - Loaded {len(INDIAN_FESTIVALS)} Indian festivals from Diwali dataset")

def check_refusal(text):
    """Check if model refused the task"""
    refusal_patterns = [
        r"i cannot", r"i can't", r"i'm unable", r"i am unable",
        r"sorry, i", r"i apologize", r"as an ai", r"i don't have",
        r"i'm not able", r"i am not able"
    ]
    text_lower = text.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def check_completeness(text):
    """Check if generation is complete"""
    text = text.strip()
    if len(text) < 20:
        return False
    # Check if ends with proper punctuation
    if text[-1] not in '.!?':
        return False
    return True

def check_contamination(text):
    """Check if model violated instructions (solved math, added explanations)"""
    text_lower = text.lower()

    # Check for solution indicators
    solution_patterns = [
        r"the answer is", r"answer:", r"solution:", r"= \d+\s*$",
        r"step \d+:", r"first,.*second,.*third", r"therefore,",
        r"so,?\s+\d+", r"total\s*=", r"result\s*="
    ]
    for pattern in solution_patterns:
        if re.search(pattern, text_lower):
            return True

    # Check for explanatory/meta text about adaptation
    meta_patterns = [r"adapted", r"original", r"indian context", r"rewritten"]
    meta_count = sum(1 for p in meta_patterns if re.search(p, text_lower))
    if meta_count >= 2:  # Multiple meta terms suggest explanation
        return True

    return False

def extract_numbers(text):
    """Extract all numbers from text"""
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(n) for n in numbers]

def count_operations(text):
    """Count mathematical operations in text"""
    text_lower = text.lower()
    operation_words = {
        'add': ['add', 'plus', 'sum', 'total', 'altogether', 'combined'],
        'subtract': ['subtract', 'minus', 'difference', 'less', 'fewer', 'remaining', 'left'],
        'multiply': ['multiply', 'times', 'product', 'each', 'every'],
        'divide': ['divide', 'split', 'share', 'per', 'equally', 'distributed']
    }

    counts = defaultdict(int)
    for op_type, keywords in operation_words.items():
        for keyword in keywords:
            counts[op_type] += len(re.findall(r'\b' + keyword + r'\b', text_lower))

    return dict(counts)

def score_math_structure(original, adapted):
    """Score mathematical structure preservation (0-1)"""
    if not adapted or len(adapted.strip()) < 10:
        return 0.0

    # Extract numbers
    orig_numbers = extract_numbers(original)
    adapt_numbers = extract_numbers(adapted)

    # Number count similarity (not values, just count)
    if len(orig_numbers) == 0:
        num_count_score = 1.0 if len(adapt_numbers) == 0 else 0.5
    else:
        num_count_score = 1.0 if len(orig_numbers) == len(adapt_numbers) else \
                          max(0, 1 - abs(len(orig_numbers) - len(adapt_numbers)) / len(orig_numbers))

    # Operation similarity
    orig_ops = count_operations(original)
    adapt_ops = count_operations(adapted)

    # Compare operation types present
    orig_op_types = set(k for k, v in orig_ops.items() if v > 0)
    adapt_op_types = set(k for k, v in adapt_ops.items() if v > 0)

    if len(orig_op_types) == 0:
        op_score = 1.0 if len(adapt_op_types) == 0 else 0.5
    else:
        op_score = len(orig_op_types & adapt_op_types) / len(orig_op_types)

    # Sentence count similarity (problem complexity)
    orig_sentences = len([s for s in re.split(r'[.!?]+', original.strip()) if s.strip()])
    adapt_sentences = len([s for s in re.split(r'[.!?]+', adapted.strip()) if s.strip()])

    if orig_sentences == 0:
        sentence_score = 1.0
    else:
        sentence_score = 1.0 if orig_sentences == adapt_sentences else \
                         max(0, 1 - abs(orig_sentences - adapt_sentences) / orig_sentences)

    # Combined score
    math_score = (num_count_score * 0.4 + op_score * 0.4 + sentence_score * 0.2)
    return math_score

def extract_entities(text):
    """Extract named entities"""
    doc = nlp(text)
    entities = {
        'PERSON': [],
        'GPE': [],  # Geo-political entities
        'ORG': [],
        'MONEY': []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text.lower())
    return entities

def score_cultural_adaptation(original, adapted):
    """Score depth of cultural adaptation (0-1)"""
    if not adapted or len(adapted.strip()) < 10:
        return 0.0

    adapted_lower = adapted.lower()
    original_lower = original.lower()

    scores = []

    # 1. Currency change
    has_dollar_orig = bool(re.search(r'\$|dollar', original_lower))
    has_rupee_adapt = bool(re.search(r'₹|rupee|rs\.?\s|inr', adapted_lower))
    if has_dollar_orig:
        scores.append(1.0 if has_rupee_adapt else 0.0)

    # 2. Indian names
    indian_name_count = sum(1 for name in INDIAN_NAMES if name in adapted_lower)
    scores.append(min(1.0, indian_name_count / 2))  # Max score at 2+ Indian names

    # 3. Indian places
    indian_place_count = sum(1 for place in INDIAN_PLACES if place in adapted_lower)
    scores.append(min(1.0, indian_place_count / 1))  # Max score at 1+ Indian place

    # 4. Indian items/food/festivals
    indian_item_count = sum(1 for item in INDIAN_ITEMS + INDIAN_FESTIVALS if item in adapted_lower)
    scores.append(min(1.0, indian_item_count / 2))  # Max score at 2+ Indian items

    # 5. Entity replacement rate (using NER)
    orig_entities = extract_entities(original)
    adapt_entities = extract_entities(adapted)

    # Check if entities were replaced (different entities)
    entity_changes = 0
    entity_total = 0
    for ent_type in ['PERSON', 'GPE']:
        if len(orig_entities[ent_type]) > 0:
            entity_total += 1
            # Check if adapted has different entities
            if len(adapt_entities[ent_type]) > 0:
                overlap = len(set(orig_entities[ent_type]) & set(adapt_entities[ent_type]))
                if overlap < len(orig_entities[ent_type]):  # Entities changed
                    entity_changes += 1

    if entity_total > 0:
        scores.append(entity_changes / entity_total)

    # Average all cultural adaptation scores
    return np.mean(scores) if scores else 0.0

def score_semantic_quality(original, adapted):
    """Score semantic quality (0-1)"""
    if not adapted or len(adapted.strip()) < 10:
        return 0.0

    scores = []

    # 1. Semantic similarity (should be high - same problem, different culture)
    orig_embedding = semantic_model.encode(original, convert_to_tensor=True)
    adapt_embedding = semantic_model.encode(adapted, convert_to_tensor=True)
    similarity = util.cos_sim(orig_embedding, adapt_embedding).item()
    scores.append(max(0, similarity))  # Should be positive

    # 2. Length ratio (shouldn't be too different)
    length_ratio = len(adapted) / max(len(original), 1)
    length_score = 1.0 if 0.5 <= length_ratio <= 2.0 else max(0, 1 - abs(1 - length_ratio) / 2)
    scores.append(length_score)

    # 3. Basic coherence (has proper sentences)
    has_sentences = bool(re.search(r'[.!?]', adapted))
    scores.append(1.0 if has_sentences else 0.5)

    return np.mean(scores)

def evaluate_generation(original, adapted):
    """Evaluate a single generation"""
    # Generation status
    is_refused = check_refusal(adapted)
    is_complete = check_completeness(adapted)
    is_contaminated = check_contamination(adapted)

    # Generation score (binary)
    generation_valid = 1.0 if (not is_refused and is_complete and not is_contaminated) else 0.0

    # Compute scores even for invalid generations (they'll just be low)
    math_score = score_math_structure(original, adapted)
    cultural_score = score_cultural_adaptation(original, adapted)
    semantic_score = score_semantic_quality(original, adapted)

    # Composite score (weighted average)
    # If generation invalid, composite will be low due to low component scores
    composite_score = (
        math_score * 0.30 +
        cultural_score * 0.40 +
        semantic_score * 0.30
    )

    return {
        'generation_valid': generation_valid,
        'is_refused': is_refused,
        'is_complete': is_complete,
        'is_contaminated': is_contaminated,
        'math_structure_score': math_score,
        'cultural_adaptation_score': cultural_score,
        'semantic_quality_score': semantic_score,
        'composite_score': composite_score
    }

# Load all CSV files from input folder
print(f"\nLoading CSV files from {args.input_folder}...")
csv_files = [f for f in os.listdir(args.input_folder) if f.endswith('.csv')]

if not csv_files:
    print(f"No CSV files found in {args.input_folder}")
    exit(1)

print(f"Found {len(csv_files)} CSV files\n")

# Evaluate all generations
all_results = []
for csv_file in csv_files:
    file_path = os.path.join(args.input_folder, csv_file)
    print(f"Processing {csv_file}...")

    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        original = str(row['original_question'])
        adapted = str(row['adapted_question'])
        model = row['model_used']

        # Evaluate
        scores = evaluate_generation(original, adapted)

        # Add metadata
        scores['model'] = model
        scores['file'] = csv_file
        scores['row_id'] = row['row_id']

        all_results.append(scores)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows")

    print(f"  Completed {csv_file}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Aggregate by model
print("\n" + "="*80)
print("AGGREGATING RESULTS BY MODEL...")
print("="*80)

model_scores = results_df.groupby('model').agg({
    'generation_valid': ['sum', 'mean'],  # Count valid, success rate
    'is_refused': 'sum',
    'is_complete': 'sum',
    'is_contaminated': 'sum',
    'math_structure_score': 'mean',
    'cultural_adaptation_score': 'mean',
    'semantic_quality_score': 'mean',
    'composite_score': 'mean',
    'row_id': 'count'  # Total generations
}).round(4)

# Flatten column names
model_scores.columns = ['_'.join(col).strip() for col in model_scores.columns.values]
model_scores = model_scores.rename(columns={
    'generation_valid_sum': 'valid_generations',
    'generation_valid_mean': 'success_rate',
    'is_refused_sum': 'refusal_count',
    'is_complete_sum': 'complete_count',
    'is_contaminated_sum': 'contamination_count',
    'math_structure_score_mean': 'avg_math_structure',
    'cultural_adaptation_score_mean': 'avg_cultural_adaptation',
    'semantic_quality_score_mean': 'avg_semantic_quality',
    'composite_score_mean': 'avg_composite_score',
    'row_id_count': 'total_generations'
})

# Reorder columns for readability
model_scores = model_scores[[
    'total_generations',
    'valid_generations',
    'success_rate',
    'refusal_count',
    'contamination_count',
    'avg_math_structure',
    'avg_cultural_adaptation',
    'avg_semantic_quality',
    'avg_composite_score'
]]

# Scale scores to 0-100 for readability
for col in ['success_rate', 'avg_math_structure', 'avg_cultural_adaptation',
            'avg_semantic_quality', 'avg_composite_score']:
    model_scores[col] = (model_scores[col] * 100).round(2)

# Sort by composite score
model_scores = model_scores.sort_values('avg_composite_score', ascending=False)

# Save results
model_scores.to_csv(args.output_file)

# Display results
print("\n" + "="*80)
print("MODEL-WISE EVALUATION SCORES (RANKED BY COMPOSITE SCORE)")
print("="*80)
print(model_scores.to_string())
print("="*80)

print("\n" + "="*80)
print("SCORE INTERPRETATION")
print("="*80)
print("""
All scores scaled 0-100 (higher is better):

  • success_rate: % of valid generations (no refusal/contamination/incompleteness)

  • avg_math_structure: Mathematical structure preservation score
    - Number count similarity (not values, just how many numbers)
    - Operation type preservation (add/subtract/multiply/divide)
    - Problem complexity (sentence count)

  • avg_cultural_adaptation: Depth of Indian cultural adaptation
    - Currency conversion ($ → ₹)
    - Indian names (from Diwali dataset: 608 names)
    - Indian places (from Diwali dataset: 547 places)
    - Indian cultural items (from Diwali dataset: 1,419 foods, 328 drinks, etc.)
    - Indian festivals (from Diwali dataset: 746 festivals)
    - Entity replacement rate (NER-based)

  • avg_semantic_quality: Text quality and coherence
    - Semantic similarity to original (same meaning, different context)
    - Length appropriateness
    - Basic coherence checks

  • avg_composite_score: Overall weighted score
    - 30% math structure + 40% cultural adaptation + 30% semantic quality

Note: ALL generations kept and scored (including refusals and failures)
""")

print("="*80)
print(f"\nDetailed results saved to: {args.output_file}")
print(f"Total generations evaluated: {len(results_df)}")
print(f"Total models evaluated: {len(model_scores)}")
print("="*80)
