#!/bin/bash

# SC2-Dynamics-50k inference
TEST_FILE="data/wm_test_horizon5.json"
OUTPUT_DIR="offline_infer/output"
CLIENT_SCRIPT="offline_infer/run_inference_client.py"

# 1 trajectory inference
# TEST_FILE="data/wm_test_horizon5_1traj.json"
# OUTPUT_DIR="offline_infer/output_1traj"

API_BASE="path/to/your/api/base"
API_KEY="path/to/your/api/key"
MODEL_NAME="Qwen3-8B"
MODEL_PATH="path/to/your/qwen3-8b/model"

echo "========================================================"
echo "Starting Experiment for Model: $MODEL_NAME at $MODEL_PATH"
echo "========================================================"

echo "Running Inference: Mode = NOTHINK"
python "$CLIENT_SCRIPT" \
    --input_file "$TEST_FILE" \
    --output_file "$OUTPUT_DIR/${MODEL_NAME}_nothink_results.jsonl" \
    --mode "nothink" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --model_id "$MODEL_PATH" \
    --max_workers 100
    
echo "All experiments completed."
