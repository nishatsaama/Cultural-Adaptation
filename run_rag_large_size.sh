#!/bin/bash

# Run RAG-enhanced cultural adaptation on GSM8K dataset for large-size models

NUM_ROWS=100
TEMP=0.7
TOP_K=5
OUTPUT_DIR="large_size_models_rag_output"

echo "Starting RAG adaptation for large-size models..."
echo "Rows per model: $NUM_ROWS"
echo "Temperature: $TEMP"
echo "Top-K retrieval: $TOP_K"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Phi-3 Mid 14B
echo "Running Phi-3 Mid 14B..."
python adapt_rag.py --model_name phi-3-mid-14b --num_rows $NUM_ROWS --temperature $TEMP --top_k $TOP_K --output_dir $OUTPUT_DIR
echo "Phi-3 Mid 14B complete"
echo ""

# Qwen 2.5 14B
echo "Running Qwen 2.5 14B..."
python adapt_rag.py --model_name qwen-2.5-14b --num_rows $NUM_ROWS --temperature $TEMP --top_k $TOP_K --output_dir $OUTPUT_DIR
echo "Qwen 2.5 14B complete"
echo ""

# Qwen 2.5 32B
echo "Running Qwen 2.5 32B..."
python adapt_rag.py --model_name qwen-2.5-32b --num_rows $NUM_ROWS --temperature $TEMP --top_k $TOP_K --output_dir $OUTPUT_DIR
echo "Qwen 2.5 32B complete"
echo ""

# Gemma 2 27B
echo "Running Gemma 2 27B..."
python adapt_rag.py --model_name gemma-2-27b --num_rows $NUM_ROWS --temperature $TEMP --top_k $TOP_K --output_dir $OUTPUT_DIR
echo "Gemma 2 27B complete"
echo ""

# Sarvam M 24B
echo "Running Sarvam M 24B..."
python adapt_rag.py --model_name sarvam --num_rows $NUM_ROWS --temperature $TEMP --top_k $TOP_K --output_dir $OUTPUT_DIR
echo "Sarvam M 24B complete"
echo ""

echo "================================"
echo "All large-size models complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python evaluate.py --input_folder $OUTPUT_DIR --output_file large_size_models_rag_eval.csv"
