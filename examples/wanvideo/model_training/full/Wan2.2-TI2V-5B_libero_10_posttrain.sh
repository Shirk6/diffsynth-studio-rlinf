export WAN_ACTION_DIM=14
export WAN_CONDITION_FRAMES=9
export WAN_DEBUG=False

PYTHONPATH=/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config.yaml \
  /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/train_rlinf.py \
  --num_frames 57 \
  --dataset_repeat 1 \
  --model_paths '[
    "/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/runs/full_libero_10/epoch-299.safetensors",
    "/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/runs/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "runs/full_libero_10_after299" \
  --trainable_models "dit" \
  --static_video_prob 0.0 \
  --extra_inputs "input_image,action" \
  --val_interval 100 \
  --save_epochs 100 \
  --dataset RLinfDataset \
  --action_dim 14 \
  --condition_frames 9 \
  --Ta 48 \
  --To 8 \
  --train_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/base_policy_rollout/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/base_policy_rollout_added/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/full_policy_rollout/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/full_policy_first_rollout/train_data"
  ]' \
  --val_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/full_policy_rollout/val_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/full_policy_first_rollout/val_data"
  ]'
