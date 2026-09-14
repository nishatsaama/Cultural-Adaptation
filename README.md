# Cultural Adaptation of GSM8K using LLMs and Agentic RAG

This project explores how large language models can rewrite mathematical word problems in an Indian cultural context while preserving their mathematical meaning.

It combines cultural knowledge from the DIWALI dataset with prompting, retrieval-augmented generation, and agentic decision-making. Adaptations are evaluated for mathematical equivalence, cultural authenticity, and linguistic quality.

## Objectives

- Adapt names, places, foods, activities, and other contextual elements to an Indian setting.
- Preserve quantities, mathematical relationships, and the unknown being asked for.
- Produce natural, culturally coherent problem statements without solutions or explanations.
- Compare adaptation methods using a consistent evaluation procedure.
- Investigate whether agent-directed retrieval and revision improve adaptation quality.

## Adaptation Methods

| Method | Script | Description |
|---|---|---|
| Direct prompting | `adapt_simple.py` | Rewrites problems using a general cultural adaptation prompt. |
| Cultural reference prompting | `adapt_diwali.py` | Provides sampled DIWALI concepts as adaptation guidance. |
| Zero-shot chain-of-thought prompting | `adapt_cot_0_shot.py` | Encourages step-by-step consideration of the adaptation while requesting only the rewritten problem. |
| Retrieval-augmented generation | `adapt_rag.py` | Retrieves relevant cultural references to support generation. |
| Improved RAG | `new_rag.py` | Combines problem analysis, category-specific retrieval, coherence checking, and iterative refinement. |
| Agentic RAG | `agentic_rag.py` | Allows a controller to select retrieval, drafting, verification, and revision actions based on feedback. |

## Agentic RAG

The agentic approach gives the model control over the next action within a bounded adaptation process.

Available actions include:

- **Analyze:** identify mathematical constraints and cultural elements.
- **Retrieve:** search DIWALI for relevant concepts, optionally filtered by category and state.
- **Draft:** generate or revise an adapted problem.
- **Verify mathematics:** assess whether the adaptation preserves the original mathematical meaning.
- **Review culture:** check cultural appropriateness, contextual plausibility, and language.
- **Submit:** return the candidate after the required checks pass.

The agent can retrieve additional references or change its approach when a review identifies a problem. Every revision invalidates previous checks, so the current candidate must be reviewed before submission.

Action traces record tool requests, observations, revisions, and completion status. Cases that reach the processing limit without accepted completion are marked for review.

Internal checks guide adaptation; they are not benchmark scores.

## Repository Structure

```text
Cultural_Adaptation_GSM8k/
├── data/
│   ├── gsm8k_test_no_answers.csv
│   ├── diwali.csv
│   ├── cultural_reference.pkl
│   └── diwali_index.pkl
├── low_size_models/
├── mid_size_models/
├── large_size_models/
├── adapt_simple.py
├── adapt_diwali.py
├── adapt_cot_0_shot.py
├── adapt_rag.py
├── new_rag.py
├── agentic_rag.py
├── build_diwali_index.py
├── evaluate_llm.py
├── compare_all_methods.py
├── run_agentic_rag.sh
└── evaluation_logs/
```

Adaptation scripts write generated problems to their configured output directories. Evaluation logs and comparison tables are produced separately.

## Datasets

### GSM8K

The source questions are stored in:

```text
data/gsm8k_test_no_answers.csv
```

The file must contain a `question` column. Reference answers are not included in this file.

### DIWALI

Cultural references are stored in:

```text
data/diwali.csv
```

Required columns are:

- `concept`
- `state`
- `facet`
- `description`

The agentic script reads this CSV directly and builds semantic embeddings during initialization. It does not require the prebuilt pickle files.

Some existing adaptation methods use `cultural_reference.pkl` or `diwali_index.pkl`. These can be rebuilt with:

```bash
python build_diwali_index.py
```

Before running experiments, inspect cultural entries for missing descriptions or malformed concepts. A citation-placeholder entry was identified in the supplied cultural CSV and should be removed before retrieval.

## Setup

Clone the repository and enter its root directory:

```bash
git clone https://github.com/nishatsaama/Cultural_Adaptation_GSM8k.git
cd Cultural_Adaptation_GSM8k
```

Place the agentic script and its launcher beside `new_rag.py`. Use the updated comparison script that recognizes the `agentic_rag` method.

Install the core dependencies:

```bash
pip install torch transformers accelerate pandas numpy sentence-transformers google-generativeai
```

For the existing spaCy-based RAG implementation, also install:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Model-specific wrappers may require additional dependencies. Use a PyTorch installation appropriate for your hardware.


## Running Agentic Adaptation

Run the provided launcher from the repository root:

```bash
bash run_agentic_rag.sh
```

Inspect the launcher before execution to select the generator, checker, dataset scope, and processing limits appropriate for your experiment.

To see all available options:

```bash
python agentic_rag.py --help
```

The main arguments are:

| Argument | Purpose |
|---|---|
| `--model_name` | Select the generator through an existing model wrapper. |
| `--checker_model_name` | Select the internal review model. |
| `--num_rows` | Set how many source questions to process. |
| `--temperature` | Set the draft generation temperature. |
| `--output_dir` | Choose where generated outputs are saved. |
| `--max_steps` | Limit agent actions per problem. |
| `--refinement_passes` | Limit revisions after the initial draft. |
| `--top_k` | Set the retrieval result count. |
| `--target_state` | Optionally specify a regional target. |
| `--seed` | Set the random seed. |
| `--input_file` | Override the source question file. |
| `--cultural_file` | Override the cultural reference file. |


## Output Files

An agentic run produces:

| File | Contents |
|---|---|
| Adaptation CSV | Original questions, adapted questions, model metadata, completion status, and resource-use information. |
| Trace JSONL | Agent actions, tool inputs, observations, and verification feedback for each question. |
| Configuration JSON | Run settings and checkpoint identifiers. |

The CSV retains the columns expected by the existing evaluator:

```text
row_id
original_question
adapted_question
model_used
temperature
```

Additional metadata includes completion status, action count, model calls, and elapsed time.

### Completion Status

- **`completed`:** the agent submitted a candidate with passing internal checks.
- **`needs_review`:** the agent reached its action limit without accepted completion.



## Evaluation

Use `evaluate_llm.py` to assess final adaptations. It evaluates:

- **Mathematical equivalence:** preservation of quantities, relationships, and the requested unknown.
- **Cultural authenticity:** appropriateness and coherence of the adapted context.
- **Linguistic quality:** clarity, fluency, and readability.

Provide a Gemini API key through the `GEMINI_API_KEY` environment variable and run:

```bash
python evaluate_llm.py \
  --input_folder agentic_rag_output \
  --output_file agentic_rag_evaluation.csv \
  --api_key "$GEMINI_API_KEY" \
  --log_dir evaluation_logs/agentic_rag_output
```

The command assumes a shell that supports this environment-variable syntax.

Keep the evaluator, rubric, scoring weights, and evaluation settings identical across methods. Agentic internal feedback must remain separate from final benchmark scoring.

## Comparing Methods

After generating evaluation logs, run:

```bash
python compare_all_methods.py
```

## Acknowledgments

This project uses GSM8K for mathematical word problems, DIWALI for Indian cultural references, and the model and embedding libraries used throughout the adaptation pipeline.

