#!/bin/bash

python offline_evaluate/run_eval.py \
    --wm_file path/to/your/world_model_infer_output_file \
    --zeroshot_32b_file path/to/your/zeroshot_32b_infer_output_file \
    --zeroshot_8b_file path/to/your/zeroshot_8b_infer_output_file \
    --static_bias_file path/to/your/static_bias_file \
    --output_file path/to/your/output_file \
    --plot


# # Paper Replication
# # Paper Figure 3-5: Offlien Evolution of AWD & Case Study.
# python offline_evaluate/run_eval.py \
#     --wm_file offline_infer/starwm_output_1traj/WM-final_nothink_results.jsonl \
#     --zeroshot_32b_file offline_infer/starwm_output_1traj/Qwen3-32B_nothink_results.jsonl \
#     --zeroshot_8b_file offline_infer/starwm_output_1traj/Qwen3-8B_nothink_results.jsonl \
#     --output_dir offline_evaluate/starwm_results \
#     --plot \
#     --figure2_frames 384 365

# # Paper Table 1: Offline evaluation results.
# python offline_evaluate/run_eval.py \
#     --wm_file offline_infer/starwm_output/WM-final_nothink_results.jsonl --zeroshot_32b_file offline_infer/starwm_output/Qwen3-32B_nothink_results.jsonl \
#     --zeroshot_8b_file offline_infer/starwm_output/Qwen3-8B_nothink_results.jsonl \
#     --static_bias_file offline_infer/starwm_output/static_bias_results.jsonl \
#     --output_dir offline_evaluate/starwm_results \