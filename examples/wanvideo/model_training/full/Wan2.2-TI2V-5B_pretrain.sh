PYTHONPATH=/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00001-of-00003.safetensors",
     "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00002-of-00003.safetensors",
     "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "runs/pretrain" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.0 \
  --extra_inputs "input_image,action" \
  --val_interval 1000 \
  --save_epochs 50 \
  --dataset RLinfLeRobotObsDataset \
  --train_dataset_base_path '[
    "/mnt/project_rlinf/jlchen/datasets/robomind_franka_1rgb",
    "/mnt/project_rlinf/jlchen/datasets/droid_1.0.1",
    "/mnt/project_rlinf/jlchen/datasets/AgiBot_merge",
  ]' \
  --val_dataset_base_path '[
    "/mnt/project_rlinf/jlchen/datasets/robomind_franka_1rgb",
    "/mnt/project_rlinf/jlchen/datasets/droid_1.0.1",
    # "/mnt/project_rlinf/jlchen/datasets/AgiBot_merge",
  ]' \
  --action_dim 14