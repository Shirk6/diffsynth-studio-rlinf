PYTHONPATH=/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --dataset_repeat 1 \
  --model_paths '[
    "/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/examples/wanvideo/model_training/full/runs/full_libero_object/epoch-599.safetensors",
    "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "runs/full_libero_object_after599" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.0 \
  --extra_inputs "input_image,action" \
  --val_interval 100 \
  --save_epochs 100 \
  --dataset RLinfDataset \
  --train_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_object/base_policy_rollout/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_object/first_policy_rollout/train_data",
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_object/full_policy_rollout/train_data"
  ]' \
  --val_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_object/full_policy_rollout/val_data"
  ]'