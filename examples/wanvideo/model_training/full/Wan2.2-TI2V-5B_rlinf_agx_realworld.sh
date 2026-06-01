CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --num_frames 57 \
  --dataset_repeat 1 \
  --model_paths '[
    "/mnt/project_rlinf/jzn/ckpt_path/RLinf-Wan-Agx/model-00001_after999_after499_after299.safetensors",
    "/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio/runs/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "outputs/real_world_agx_3tasks_after999_after499_after299" \
  --trainable_models "dit" \
  --static_video_prob 0.0 \
  --extra_inputs "input_image,action" \
  --val_interval 50 \
  --save_epochs 50 \
  --dataset SimpleVLARealWorldRLinfDataset \
  --condition_frames 9 \
  --Ta 48 \
  --To 8 \
  --train_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_base_policy_rollout/fold_towel_eef_infer_data_3task_fold_towel_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_base_policy_rollout/pick_cube_delta_EEF_infer_data_3task_pick_cube_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_base_policy_rollout/pick_tomato_eef_infer_data_3task_pick_tomato_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_val_base_policy_rollout/fold_towel_eef_infer_data_3task_fold_towel_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_val_base_policy_rollout/pick_cube_delta_EEF_infer_data_3task_pick_cube_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_val_base_policy_rollout/pick_tomato_eef_infer_data_3task_pick_tomato_clean_process"
  ]' \
  --val_dataset_base_path '[
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_val_base_policy_rollout/fold_towel_eef_infer_data_3task_fold_towel_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_val_base_policy_rollout/pick_cube_delta_EEF_infer_data_3task_pick_cube_clean_process",
    "/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_val_base_policy_rollout/pick_tomato_eef_infer_data_3task_pick_tomato_clean_process"
  ]' \
  --action_dim 14
