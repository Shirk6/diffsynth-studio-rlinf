srun --gres=gpu:1 --account=peilab --time=00:30:00 --pty \
/cm/local/apps/apptainer/current/bin/apptainer exec --nv \
--bind /project/peilab/srk/rss_2026_ws:/project/peilab/srk/rss_2026_ws \
/project/peilab/srk/.cache/enroot/rlinf-embodied-wan-openpi-shirk6.sif \
bash -lc 'cd /project/peilab/srk/rss_2026_ws/diffsynth-studio &&python examples/wanvideo/model_inference/Wan2.2-TI2V-5B_challenge_rlinf_autoregressive.py --episode failure__seed_000293_seg_000 --save_compare'