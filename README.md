# World Models for Policy Refinement in StarCraft II

[📄 Paper (arXiv)](https://arxiv.org/abs/2602.14857)  
[🤗 Pretrained Model](https://huggingface.co/yxzhang2024/StarWM)  
[📊 SC2-Dynamics-50K Dataset](https://huggingface.co/datasets/yxzhang2024/SC2-Dynamics-50K)

> This repository contains the official implementation of the paper **World Models for Policy Refinement in StarCraft II**.

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Installation](#-installation)
- [Datasets & Models](#-datasets--models)
- [Quick Start](#-quick-start)
  - [World Model Training (Optional)](#optional-world-model-training)
  - [Offline Inference](#offline-inference)
  - [Offline Evaluation](#offline-evaluation)
  - [Online Testing](#online-testing)
- [Citation](#-citation)

## 📝 Introduction

We propose **StarWM**, the first action-conditioned world model for StarCraft II. 
Given the current observation and a sequence of actions, StarWM predicts structured future observations under partial observability. 
It is further integrated into a world-model-augmented decision system (**StarWM-Agent**) to enable short-horizon predictive simulation and inference-time policy refinement.

<p align="center">
  <img src="assets/figure1.png" width="800">
</p>

To address dynamics modeling and decision integration in this hybrid and partially observable environment, we:

1. Introduce a **structured textual observation representation** that factorizes SC2 dynamics into five semantic modules:
  - Info (economy & status)
  - Queue (production & research)
  - My Units
  - My Structures
  - Visible Hostiles
2. Construct **SC2-Dynamics-50K**, the first instruction-tuning dataset for SC2 dynamics prediction, and train StarWM via supervised fine-tuning on Qwen3-8B
3. Develop a **multi-dimensional offline evaluation framework** that measures world model quality across Economy, Development, Micro-Entity, Macro-Situation
4. Propose **StarWM-Agent**, a Generate–Simulate–Refine decision system for inference-time policy refinement

### 🚀 Offline Evaluation Results

<p align="center">
  <img src="assets/figure2.png" width="800">
</p>

- 🟢 **60% reduction** in minerals prediction error (SMAPE)
- 🟢 **60% improvement** in self-side macro-situation consistency (AWD)
- 🟢 Significant gains in task progress prediction and unit attributes modeling

### 🧠 StarWM-Agent: Generate–Simulate–Refine

StarWM-Agent augments an LLM policy with short-horizon predictive simulation:

1. **Generate** initial actions
2. **Simulate** predicted future using StarWM
3. **Refine** actions conditioned on predicted future

This enables:
- Preemptive macro-management (including Supply bottleneck anticipation)
- Lightweight combat feasibility checking

#### 🚀 Online Decision-Making Performance

<p align="center">
  <img src="assets/figure3.png" width="800">
</p>

- +30% / +15% / +30% win-rate gains against SC2's Hard / Harder / VeryHard built-in AI
- Significant reduction in supply block rate
- Higher resource conversion rate
- Improved kill-loss ratio

## 🛠️ Installation

First, clone this repository:
```bash
git clone https://github.com/yxzzhang/StarWM.git
cd StarWM
```

Create a new conda environment with Python 3.12 and install the required dependencies:

```bash
conda create -n StarWM python=3.12 -y
conda activate StarWM
pip install -r requirements.txt
```

## 📊 Datasets & Models

Download the `SC2-Dynamics-50K` dataset and `StarWM` from Hugging Face:

```bash
hf download --repo-type dataset yxzhang2024/SC2-Dynamics-50K --local-dir ./data/
```

The dataset contains:
- `wm_train_horizon5.json`
- `wm_val_horizon5.json`
- `wm_test_horizon5.json`

Download the StarWM model from Hugging Face:

```bash
hf download yxzhang2024/StarWM --local-dir path/to/your/local/dir
```

You can deploy this model using vLLM for inference.


## 🚀 Quick Start
### (Optional) World Model Training

If you don't download the StarWM model and want to train from scratch, we provide a tutorial here, using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training.

You need to copy the `SC2-Dynamics-50K` dataset to the LLaMA-Factory data directory.

```bash
# Create the directory
mkdir -p train/LLaMA-Factory/data/SC2-Dynamics-50K

# Copy the files
cp data/*.json train/LLaMA-Factory/data/SC2-Dynamics-50K/
```

After preparation, the directory structure should look like this:

```
StarWM/
├── data/
│   ├── wm_train_horizon5.json
│   ├── ...
├── train/
│   └── LLaMA-Factory/
│       └── data/
│           └── SC2-Dynamics-50K/
│               ├── wm_train_horizon5.json
│               ├── ...
```

#### 1. Training

Modify `train/train_starwm.sh` to set your `model_name_or_path` and `output_dir`.

Navigate to the LLaMA-Factory directory to run the training script:

```bash
cd train/LLaMA-Factory
bash ../train_starwm.sh
```

#### 2. Merge LoRA Weights

After training, you need to merge the LoRA weights.
Modify `examples/merge_lora/qwen3_lora_sft.yaml` with your paths, then run:

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

Then deploy this model using vLLM for inference.

### Offline Inference

Inference scripts are located in `offline_infer/`.

1. Deploy your merged model or the StarWM using vLLM (compatible with OpenAI API).
2. Edit `offline_infer/run_experiment_wm.sh` to set:
   - `API_BASE`: Your vLLM server address (e.g., `http://localhost:8000/v1`)
   - `API_KEY`: Your API key (if any, else `EMPTY`)
   - `MODEL_PATH`: The model name served by vLLM

Run the inference:

```bash
# Run from the project root
bash offline_infer/run_experiment_wm.sh
```

### Offline Evaluation

Evaluation scripts are in `offline_evaluate/`.

#### General Usage

Edit `offline_evaluate/run_eval.sh` (lines 3-9) to point to your inference output files, then run:

```bash
bash offline_evaluate/run_eval.sh
```

#### Paper Replication

To replicate the results from the paper (Figure 3-5, Table 1, Table 4), you can use the specific commands provided below.

**Reproduce Figure 3-5 (Offline Evolution of AWD & Case Study):**
```bash
python offline_evaluate/run_eval.py \
    --wm_file offline_infer/starwm_output_1traj/WM-final_nothink_results.jsonl \
    --zeroshot_32b_file offline_infer/starwm_output_1traj/Qwen3-32B_nothink_results.jsonl \
    --zeroshot_8b_file offline_infer/starwm_output_1traj/Qwen3-8B_nothink_results.jsonl \
    --output_dir offline_evaluate/starwm_results \
    --plot \
    --figure2_frames 384 365
```

**Reproduce Table 1,4 (Offline evaluation results):**
```bash
python offline_evaluate/run_eval.py \
    --wm_file offline_infer/starwm_output/WM-final_nothink_results.jsonl \
    --zeroshot_32b_file offline_infer/starwm_output/Qwen3-32B_nothink_results.jsonl \
    --zeroshot_8b_file offline_infer/starwm_output/Qwen3-8B_nothink_results.jsonl \
    --static_bias_file offline_infer/starwm_output/static_bias_results.jsonl \
    --output_dir offline_evaluate/starwm_results
```

### Online Testing

The online testing component is built on the SC2Arena platform. As the SC2Arena codebase has not yet been publicly released, we are currently seeking permission from the SC2Arena authors for open-sourcing the related code and will release it once approval is granted.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhang2026worldmodels,
      title={World Models for Policy Refinement in StarCraft II}, 
      author={Yixin Zhang and Ziyi Wang and Yiming Rong and Haoxi Wang and Jinling Jiang and Shuang Xu and Haoran Wu and Shiyu Zhou and Bo Xu},
      year={2026},
      eprint={2602.14857},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.14857}, 
}
```
