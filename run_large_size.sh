#!/bin/bash

# Run cultural adaptation on GSM8K dataset for large-size models

NUM_ROWS=10
TEMP=0.7
OUTPUT_DIR="large_size_models_output"

echo "Starting adaptation for large-size models..."
echo "Rows per model: $NUM_ROWS"
echo "Temperature: $TEMP"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Phi 3 Mid 14B
echo "Running Phi 3 Mid 14B..."
python adapt_simple.py --model_name phi-3-mid-14b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Phi 3 Mid 14B complete"
echo ""

# Qwen 2.5 14B
echo "Running Qwen 2.5 14B..."
python adapt_simple.py --model_name qwen-2.5-14b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Qwen 2.5 14B complete"
echo ""

# Qwen 2.5 32B
echo "Running Qwen 2.5 32B..."
python adapt_simple.py --model_name qwen-2.5-32b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Qwen 2.5 32B complete"
echo ""

# Gemma 2 27B
echo "Running Gemma 2 27B..."
python adapt_simple.py --model_name gemma-2-27b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Gemma 2 27B complete"
echo ""

# Sarvam-M-24B
echo "Running Sarvam-M-24B..."
python adapt_simple.py --model_name sarvam-m-24b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Sarvam-M-24B complete"
echo ""

echo "================================"
echo "All large-size models complete!"
