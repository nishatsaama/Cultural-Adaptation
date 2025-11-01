#!/bin/bash

# Run Diwali-enhanced cultural adaptation on GSM8K dataset for mid-size models

NUM_ROWS=100
TEMP=0.7
NUM_CONCEPTS=105
OUTPUT_DIR="mid_size_models_diwali_output"

echo "Starting Diwali-enhanced adaptation for mid-size models..."
echo "Rows per model: $NUM_ROWS"
echo "Temperature: $TEMP"
echo "Cultural concepts: $NUM_CONCEPTS"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Llama 3.1 8B
echo "Running Llama 3.1 8B..."
python adapt_diwali.py --model_name llama-3.1-8b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Llama 3.1 8B complete"
echo ""

# Qwen 2.5 7B
echo "Running Qwen 2.5 7B..."
python adapt_diwali.py --model_name qwen-2.5-7b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Qwen 2.5 7B complete"
echo ""

# Mistral 7B
echo "Running Mistral 7B..."
python adapt_diwali.py --model_name mistral-7b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Mistral 7B complete"
echo ""

# Gemma 2 9B
echo "Running Gemma 2 9B..."
python adapt_diwali.py --model_name gemma-2-9b --num_rows $NUM_ROWS --temperature $TEMP --num_concepts $NUM_CONCEPTS --output_dir $OUTPUT_DIR
echo "Gemma 2 9B complete"
echo ""

echo "================================"
echo "All mid-size models complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python evaluate.py --input_folder $OUTPUT_DIR --output_file mid_size_models_diwali_eval.csv"
