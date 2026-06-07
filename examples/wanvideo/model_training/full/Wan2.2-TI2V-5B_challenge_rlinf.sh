#!/usr/bin/env bash
set -euo pipefail

cd /project/peilab/srk/rss_2026_ws/diffsynth-studio

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_DIR="/project/peilab/srk/rss_2026_ws/models/Wan-AI/Wan2.2-TI2V-5B"
DATASET_BASE="/project/peilab/srk/rss_2026_ws/Challenge-phase1-dataset-rlinf/tower-of-hanoi-game"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/Wan2.2-TI2V-5B_challenge_rlinf}"

accelerate launch \
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
  --val_interval 10 \
  --save_epochs 200 \
  --dataset RLinfNpyDataset \
  --dataset_base_path "${DATASET_BASE}" \
  --train_split_dir "train-data" \
  --val_split_dir "val-data"
