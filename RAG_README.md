# RAG-Enhanced Cultural Adaptation

This implementation uses **Retrieval-Augmented Generation (RAG)** to improve cultural adaptation quality by leveraging the Diwali cultural knowledge dataset.

## What is RAG?

**RAG (Retrieval-Augmented Generation)** enhances LLM outputs by retrieving relevant information from a knowledge base before generation.

**How it improves cultural adaptation:**
1. **Indexes** 8,818 Diwali cultural concepts (names, foods, places, festivals, etc.) as searchable embeddings
2. **Retrieves** semantically relevant cultural items based on the math problem context
3. **Injects** these authentic Indian concepts into the LLM prompt
4. **Generates** culturally richer adaptations grounded in real knowledge

**Result:** Instead of generic adaptations, you get problems using authentic names like "Satyavathi", foods like "Pulihora", and places like "Visakhapatnam" from the actual Diwali dataset.

## RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  OFFLINE (One-time): Build Index                            │
│  Diwali Dataset (8,818 entries) → Embeddings → Index        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ONLINE (Per math problem):                                 │
│                                                              │
│  GSM8K Problem: "Janet bought 3 apples for $2..."          │
│         ↓                                                    │
│  1. Extract Query using NER (Generalized):                  │
│     Entities: Janet                                         │
│     Nouns: apples, market                                   │
│     Verbs: buy                                              │
│     → Query: "Janet apples market buy"                      │
│         ↓                                                    │
│  2. Semantic Search → Retrieve Top-5 from Diwali:          │
│     - Pulihora, Hyderabadi Biryani, Pesarattu...           │
│     (Items semantically similar to "apples buy market")    │
│         ↓                                                    │
│  3. Sample Cultural Elements from Dataset:                  │
│     Names: Lakshmi, Satyavathi, Arjun... (from 608 names)  │
│     Foods: Pootharekulu, Gongura... (from 1,419 foods)     │
│     Places: Visakhapatnam, Tirupati... (from 547 places)   │
│         ↓                                                    │
│  4. Inject into Prompt + Generate                           │
│         ↓                                                    │
│  "Lakshmi bought 3 samosas for ₹20 in Visakhapatnam..."    │
└─────────────────────────────────────────────────────────────┘
```

## What Gets Indexed?

The Diwali dataset contains **8,818 cultural entries** organized by facet:

- **Names:** 608 entries (Lakshmi, Satyavathi, Srinivasarao, Arjun, Bhavani...)
- **Foods:** 1,419 entries (Pulihora, Hyderabadi Biryani, Gongura Pickle, Pesarattu...)
- **Places:** 547 entries (Visakhapatnam, Tirupati, Amaravati, Vijayawada...)
- **Festivals:** 746 entries
- **Drinks:** 328 entries
- **Clothing:** 464 entries
- **Textiles:** 265 entries
- **Dances:** 1,105 entries
- **Arts:** 288 entries
- **Games:** 412 entries

Each entry is converted to an **embedding** (vector) for semantic search.

## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### 2. Build the Diwali Index (One-time)

```bash
python build_diwali_index.py
```

This will:
- Load the Diwali dataset (8,818 entries)
- Create embeddings for semantic search using sentence-transformers
- Extract cultural concepts organized by facet (names, foods, places, etc.)
- Save index files to `data/` directory

**Output files:**
- `data/diwali_index.pkl` - Main embedding index (all 8,818 entries as vectors)
- `data/cultural_reference.pkl` - Extracted concepts from dataset (608 names, 1,419 foods, 547 places, etc.)

**What gets extracted:**
- 608 authentic Indian names (Lakshmi, Satyavathi, Srinivasarao...)
- 1,419 traditional foods (Pulihora, Hyderabadi Biryani, Gongura Pickle...)
- 547 Indian places (Visakhapatnam, Tirupati, Amaravati...)
- Plus festivals, drinks, clothing, textiles, dances, arts, games

Time: ~2-5 minutes depending on your machine

## Usage

### Option 1: Single Model

```bash
python adapt_rag.py \
    --model_name "llama-3.2-3b" \
    --num_rows 50 \
    --temperature 0.7 \
    --top_k 5 \
    --output_dir "rag_results"
```

### Option 2: All Models (Batch)

```bash
./run_rag.sh
```

This will run RAG adaptation on all models (low/mid/large size).

### Option 3: With Facet Filtering

To retrieve only from specific cultural categories:

```bash
python adapt_rag.py \
    --model_name "qwen-2.5-3b" \
    --num_rows 50 \
    --top_k 5 \
    --use_facet "textiles" \
    --output_dir "rag_results"
```

Available facets: Check output of `build_diwali_index.py`

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Model identifier (e.g., "llama-3.2-3b") | Required |
| `--num_rows` | Number of GSM8K problems to process | 10 |
| `--temperature` | Generation temperature | 0.7 |
| `--top_k` | Number of cultural items to retrieve | 5 |
| `--use_facet` | Filter retrieval by facet (optional) | None |
| `--output_dir` | Output directory for results | Required |

## Output

Each run produces a CSV file with:
- `row_id`: Problem ID
- `original_question`: GSM8K problem
- `adapted_question`: RAG-enhanced adaptation
- `retrieved_items`: Cultural items used
- `query_used`: Retrieval query
- `model_used`: Model name
- `temperature`: Temperature value
- `top_k`: K value used
- `facet_filter`: Facet filter used (if any)

**Filename format:**
```
{model_name}_rag_{facet}_{k}{num_rows}rows_temp{temperature}.csv
```

Example:
```
llama_3_2_3b_it_rag_k5_50rows_temp0.7.csv
```

## Evaluation

After generating adaptations, evaluate them:

```bash
python evaluate.py \
    --input_folder "rag_results" \
    --output_file "rag_evaluation.csv"
```

This produces model-wise scores for:
- **Math Structure Preservation** (0-100)
- **Cultural Adaptation Depth** (0-100)
- **Semantic Quality** (0-100)
- **Composite Score** (weighted average)

## Comparison with Baseline

To compare RAG vs baseline (no RAG):

1. Run baseline:
```bash
python adapt_simple.py --model_name "llama-3.2-3b" --num_rows 50 --output_dir "baseline_results"
```

2. Run RAG:
```bash
python adapt_rag.py --model_name "llama-3.2-3b" --num_rows 50 --top_k 5 --output_dir "rag_results"
```

3. Evaluate both:
```bash
python evaluate.py --input_folder "baseline_results" --output_file "baseline_eval.csv"
python evaluate.py --input_folder "rag_results" --output_file "rag_eval.csv"
```

4. Compare `avg_cultural_adaptation` scores!

## Expected Improvements

RAG should improve:
- ✅ **Cultural Adaptation Score** (+10-20 points)
  - Uses 608 authentic names instead of generic ones (Satyavathi, Srinivasarao vs Priya, Raj)
  - Uses 1,419 traditional foods instead of common items (Pulihora, Pootharekulu vs samosa, dosa)
  - Uses 547 real places instead of major cities (Visakhapatnam, Lepakshi vs Mumbai, Delhi)
  - Context-specific retrieval (food problems get food items, clothing problems get textiles)

- ✅ **Semantic Quality** (+5-10 points)
  - More coherent and diverse adaptations
  - Grounded in actual cultural knowledge from Diwali dataset
  - Reduced repetition across problems

- ⚠️ **Math Structure** (should remain similar)
  - RAG doesn't affect mathematical logic, only cultural elements

## Advanced Usage

### Query Extraction Method

The system uses **generalized NER-based extraction** (not rule-based):

**How it works:**
1. Parse problem with spaCy NER
2. Extract entities (PERSON, GPE, ORG, etc.)
3. Extract nouns (objects, items)
4. Extract verbs (actions, context)
5. Combine into semantic query

**Example:**
```
Problem: "Janet bought 3 apples for $2 at the market"
Extracted:
  - Entities: Janet
  - Nouns: apples, market
  - Verbs: buy
  → Query: "Janet apples market buy"
```

This query is semantically matched against Diwali dataset embeddings to retrieve relevant cultural items.

**Test the extraction:**
```bash
python test_query_extraction.py
```

**Customize extraction:**
Edit `extract_query_terms()` in `adapt_rag.py:95-145` to modify the NER logic.

### Modifying Extracted Concepts

The cultural references are automatically extracted from the Diwali dataset. To modify:
1. Edit the facet extraction logic in `build_diwali_index.py:81-91`
2. Add or remove facets (e.g., add `jwellery`, `rituals`, etc.)
3. Rebuild index: `python build_diwali_index.py`

### Retrieval Tuning

- Increase `--top_k` for more diverse cultural context (may slow down)
- Decrease `--top_k` for faster processing with focused context
- Use `--use_facet` to specialize (e.g., only textiles, only handicrafts)

## Troubleshooting

**Error: "Diwali index not found"**
```bash
python build_diwali_index.py
```

**Error: "Model not supported"**
Check that model name matches one in `MODEL_FILES` dict in script.

**Low cultural scores even with RAG:**
- Try increasing `--top_k` to 8-10
- Check retrieved items in output CSV to debug query quality
- Enhance `extract_query_terms()` with NER for better retrieval

**Out of memory:**
- Reduce batch size in `build_diwali_index.py`
- Process fewer rows at a time
- Use smaller models

## Files Created

```
TNLP/
├── build_diwali_index.py      # Index builder (run once)
├── adapt_rag.py               # RAG adaptation script
├── run_rag.sh                 # Batch runner
├── RAG_README.md              # This file
├── data/
│   ├── diwali.csv            # Original dataset (8,818 entries)
│   ├── diwali_index.pkl      # Embedding index for semantic search
│   └── cultural_reference.pkl # Extracted concepts (608 names, 1,419 foods, etc.)
└── rag_results/               # Output directory
    └── *.csv                  # Generated adaptations
```

## Next Steps

1. **Build Index**: `python build_diwali_index.py`
2. **Test Single Model**: `python adapt_rag.py --model_name "llama-3.2-3b" --num_rows 10 --output_dir "rag_results"`
3. **Run All Models**: `./run_rag.sh`
4. **Evaluate**: `python evaluate.py --input_folder "rag_results" --output_file "rag_eval.csv"`
5. **Compare with baseline** from previous runs

Good luck with your cultural adaptation experiments!
