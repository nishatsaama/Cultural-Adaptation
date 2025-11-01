#!/bin/bash

# Run cultural adaptation on GSM8K dataset for low-size models

NUM_ROWS=100
TEMP=0.7
OUTPUT_DIR="low_size_models_output"

echo "Starting adaptation for low-size models..."
echo "Rows per model: $NUM_ROWS"
echo "Temperature: $TEMP"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Llama 3.2 1B
echo "Running Llama 3.2 1B..."
python adapt_simple.py --model_name llama-3.2-1b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Llama 3.2 1B complete"
echo ""

# Llama 3.2 3B
echo "Running Llama 3.2 3B..."
python adapt_simple.py --model_name llama-3.2-3b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Llama 3.2 3B complete"
echo ""

# Gemma 2 2B
echo "Running Gemma 2 2B..."
python adapt_simple.py --model_name gemma-2-2b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Gemma 2 2B complete"
echo ""

# Qwen 2.5 1.5B
echo "Running Qwen 2.5 1.5B..."
python adapt_simple.py --model_name qwen-2.5-1.5b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Qwen 2.5 1.5B complete"
echo ""

# Qwen 2.5 3B
echo "Running Qwen 2.5 3B..."
python adapt_simple.py --model_name qwen-2.5-3b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Qwen 2.5 3B complete"
echo ""

# Phi-3.5 Mini
echo "Running Phi-3.5 Mini..."
python adapt_simple.py --model_name phi-3.5-mini --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Phi-3.5 Mini complete"
echo ""

# Granite 3.0 2B
echo "Running Granite 3.3 2B..."
python adapt_simple.py --model_name granite-3.3-2b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Granite 3.0 2B complete"
echo ""

# Param 1 2.9B
echo "Running Param 1 2.9B..."
python adapt_simple.py --model_name param-1-2.9b --num_rows $NUM_ROWS --temperature $TEMP --output_dir $OUTPUT_DIR
echo "Param 1 2.9B complete"
echo ""

echo "================================"
echo "All low-size models complete!"
