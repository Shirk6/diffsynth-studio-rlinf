srun --jobid=441005 --overlap -w dgx-26 -N1 -n1 --gres=gpu:8 \
/cm/local/apps/apptainer/current/bin/apptainer exec --nv \
--bind /project/peilab/srk/rss_2026_ws:/project/peilab/srk/rss_2026_ws \
/project/peilab/srk/.cache/enroot/rlinf-embodied-wan-openpi-shirk6.sif \
bash -lc 'cd /project/peilab/srk/rss_2026_ws/diffsynth-studio && bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_challenge_rlinf.sh'