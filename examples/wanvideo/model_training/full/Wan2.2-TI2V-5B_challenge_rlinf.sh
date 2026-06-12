#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_DIR="${MODEL_DIR:-models/Wan-AI/Wan2.2-TI2V-5B}"
DATASET_BASE="${DATASET_BASE:-mnt/amlfs-01/home/fangqiz/Challenge-phase1-dataset-rlinf)"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/seal-water-bottle-cap}"
NUM_MACHINES="${NUM_MACHINES:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-64}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

accelerate launch \
  --num_machines "${NUM_MACHINES}" \
  --num_processes "${NUM_PROCESSES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 544 \
  --width 320 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths "[
    [
      \"${MODEL_DIR}/diffusion_pytorch_model-00001-of-00003.safetensors\",
      \"${MODEL_DIR}/diffusion_pytorch_model-00002-of-00003.safetensors\",
      \"${MODEL_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors\"
    ],
    \"${MODEL_DIR}/Wan2.2_VAE.pth\",
    \"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\"
  ]" \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05 \
  --extra_inputs "input_image,action" \
  --val_interval 100 \
  --save_epochs 200 \
  --dataset RLinfNpyDataset \
  --dataset_base_path "${DATASET_BASE}" \
  --train_split_dir "train-data" \
  --val_split_dir "val-data" \
  --log_backend wandb \
  --wandb_project wan-world-model \
  --wandb_run_name seal-water-bottle-cap
