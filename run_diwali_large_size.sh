#!/bin/bash

# Run Diwali-enhanced cultural adaptation on GSM8K dataset for large-size models

NUM_ROWS=100
TEMP=0.7
NUM_CONCEPTS=105
OUTPUT_DIR="large_size_models_diwali_output"

echo "Starting Diwali-enhanced adaptation for large-size models..."
echo "Rows per model: $NUM_ROWS"
echo "Temperature: $TEMP"
echo "Cultural concepts: $NUM_CONCEPTS"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Phi-3 Mid 14B
echo "Running Phi-3 Mid 14B..."
python adapt_diwali.py --model_name phi-3-mid-14b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Phi-3 Mid 14B complete"
echo ""

# Qwen 2.5 14B
echo "Running Qwen 2.5 14B..."
python adapt_diwali.py --model_name qwen-2.5-14b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Qwen 2.5 14B complete"
echo ""

# Qwen 2.5 32B
echo "Running Qwen 2.5 32B..."
python adapt_diwali.py --model_name qwen-2.5-32b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Qwen 2.5 32B complete"
echo ""

# Gemma 2 27B
echo "Running Gemma 2 27B..."
python adapt_diwali.py --model_name gemma-2-27b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Gemma 2 27B complete"
echo ""

# Sarvam M 24B
echo "Running Sarvam M 24B..."
python adapt_diwali.py --model_name sarvam --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Sarvam M 24B complete"
echo ""

echo "================================"
echo "All large-size models complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python evaluate.py --input_folder $OUTPUT_DIR --output_file large_size_models_diwali_eval.csv"
