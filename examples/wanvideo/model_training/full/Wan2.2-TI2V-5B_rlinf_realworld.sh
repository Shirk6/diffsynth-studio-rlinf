CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00001-of-00003.safetensors",
     "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00002-of-00003.safetensors",
     "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "outputs/real_world_franka_3tasks" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.0 \
  --extra_inputs "input_image,action" \
  --val_interval 100 \
  --save_epochs 100 \
  --dataset SimpleVLARealWorldRLinfDataset \
  --train_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/franka_3tasks/pull_drawer/base_policy_rollout",
    "/mnt/project_rlinf/jzn/dataset/franka_3tasks/pick_bread/base_policy_rollout",
    "/mnt/project_rlinf/jzn/dataset/franka_3tasks/pick_banana/base_policy_rollout"
  ]' \
  --val_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/franka_3tasks/pull_drawer/val_base_policy_rollout",
    "/mnt/project_rlinf/jzn/dataset/franka_3tasks/pick_bread/val_base_policy_rollout",
    "/mnt/project_rlinf/jzn/dataset/franka_3tasks/pick_banana/val_base_policy_rollout"
  ]' \
  --action_dim 7