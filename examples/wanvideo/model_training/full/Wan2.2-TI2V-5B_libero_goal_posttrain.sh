PYTHONPATH=/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config.yaml \
  /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/train_rlinf.py \
  --dataset_repeat 1 \
  --model_paths '[
    "/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/runs/full_libero_goal/epoch-1099.safetensors",
    "/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/runs/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "runs/full_libero_goal_after799" \
  --trainable_models "dit" \
  --static_video_prob 0.0 \
  --extra_inputs "input_image,action" \
  --val_interval 100 \
  --save_epochs 100 \
  --dataset RLinfDataset \
  --train_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_goal/base_policy_rollout/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_goal/first_policy_rollout/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_goal/full_policy_rollout/train_data"
  ]' \
  --val_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_goal/full_policy_rollout/val_data"
  ]'