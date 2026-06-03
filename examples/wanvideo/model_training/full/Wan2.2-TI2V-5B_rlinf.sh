CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /project/peilab/srk/rss_2026_ws/diffsynth-studio-rlinf/examples/wanvideo/model_training/full/accelerate_config.yaml \
  /project/peilab/srk/rss_2026_ws/diffsynth-studio-rlinf/examples/wanvideo/model_training/train_rlinf.py \
  --height 544 \
  --width 320 \
  --num_frames 57 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/project/peilab/srk/rss_2026_ws/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
     "/project/peilab/srk/rss_2026_ws/models/Wan-AI/Wan2.2-TI2V-5B//diffusion_pytorch_model-00002-of-00003.safetensors",
     "/project/peilab/srk/rss_2026_ws/models/Wan-AI/Wan2.2-TI2V-5B//diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/project/peilab/srk/rss_2026_ws/models/Wan-AI/Wan2.2-TI2V-5B//Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "diffsynth-studio-rlinf/outputs/tower-of-hanoi-game" \
  --trainable_models "dit" \
  --static_video_prob 0.05 \
  --extra_inputs "input_image,action" \
  --val_interval 50 \
  --save_epochs 500 \
  --dataset RLinfDataset \
  --action_dim 14 \
  --condition_frames 9 \
  --Ta 48 \
  --To 8 \
  --retain_actions True \
  --train_dataset_base_path '[
    "/project/peilab/srk/rss_2026_ws/Challenge-phase1-dataset-rlinf/tower-of-hanoi-game/train-data"
  ]' \
  --val_dataset_base_path '[
    "/project/peilab/srk/rss_2026_ws/Challenge-phase1-dataset-rlinf/tower-of-hanoi-game/val-data"
  ]'
