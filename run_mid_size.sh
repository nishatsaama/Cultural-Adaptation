#!/bin/bash

# Run cultural adaptation on GSM8K dataset for mid-size models

NUM_ROWS=100
TEMP=0.7
OUTPUT_DIR="mid_size_models_output"

echo "Starting adaptation for mid-size models..."
echo "Rows per model: $NUM_ROWS"
echo "Temperature: $TEMP"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Llama 3.1 8B
echo "Running Llama 3.1 8B..."
python adapt_simple.py --model_name llama-3.1-8b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Llama 3.1 8B complete"
echo ""

# Qwen 2.5 7B
echo "Running Qwen 2.5 7B..."
python adapt_simple.py --model_name qwen-2.5-7b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Qwen 2.5 7B complete"
echo ""

# Mistral 7B v0.3
echo "Running Mistral 7B v0.3..."
python adapt_simple.py --model_name mistral-7b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Mistral 7B v0.3 complete"
echo ""

# Gemma 2 9B
echo "Running Gemma 2 9B..."
python adapt_simple.py --model_name gemma-2-9b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Gemma 2 9B complete"
echo ""

echo "================================"
echo "All mid-size models complete!"
