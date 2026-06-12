# seal-water-bottle-cap Wan World Model 训练 (DiffSynth-Studio)

这是基于 DiffSynth-Studio 开发的 Wan2.2-TI2V-5B 世界模型微调代码，目标是训练 `seal-water-bottle-cap` 任务的 Wan world model。下面的命令都假设从仓库根目录的 `diffsynth-studio` 目录运行，所有路径都使用相对路径。

## 环境准备

```bash
cd diffsynth-studio
pip install -e .
pip install deepspeed wandb modelscope
```

## 下载 Wan2.2-TI2V-5B 模型

将模型下载到仓库内的 `models/Wan-AI/Wan2.2-TI2V-5B`，训练脚本默认会从这个相对路径读取模型文件。

```bash
cd diffsynth-studio
mkdir -p models/Wan-AI/Wan2.2-TI2V-5B
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Wan-AI/Wan2.2-TI2V-5B
```

下载后至少应包含这些文件：

```bash
ls models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
ls models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
ls models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
ls models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
ls models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
```

如果模型放在仓库内其他目录，运行训练时用 `MODEL_DIR` 覆盖即可，例如 `MODEL_DIR=models/local/Wan2.2-TI2V-5B`。

## 训练前必须修改 action MLP 初始化

训练 `seal-water-bottle-cap` 的 Wan world model 前，必须把 `diffsynth/models/model_manager.py` 中 action MLP 的初始化开关改为 `True`。当前对应位置是第 118 行和第 142 行：

```python
need_init1 = True
need_init2 = True
```

也就是将原来的 `need_init1 = False` 和 `need_init2 = False` 改为 `True`。这一步用于强制重新初始化 `action_mlp1` 和 `action_mlp2`，对训练很重要。

## 8 机 64 卡训练 seal-water-bottle-cap

`examples/wanvideo/model_training/full/accelerate_config_14B.yaml` 已配置为 8 机 64 卡：`num_machines: 8`、`num_processes: 64`。每台机器运行同一个脚本，并设置不同的 `MACHINE_RANK`，取值为 `0` 到 `7`。`MAIN_PROCESS_IP` 使用 rank 0 机器的地址。

rank 0 示例：

```bash
cd diffsynth-studio
DATASET_BASE=../datasets/Challenge-phase1-dataset-rlinf \
OUTPUT_PATH=outputs/seal-water-bottle-cap \
MODEL_DIR=models/Wan-AI/Wan2.2-TI2V-5B \
NUM_MACHINES=8 \
NUM_PROCESSES=64 \
MACHINE_RANK=0 \
MAIN_PROCESS_IP=127.0.0.1 \
MAIN_PROCESS_PORT=29500 \
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_challenge_rlinf.sh
```

rank 1 到 rank 7 在对应机器上运行同一命令，只修改 `MACHINE_RANK`，并保持 `MAIN_PROCESS_IP` 指向 rank 0 机器。

```bash
cd diffsynth-studio
DATASET_BASE=../datasets/Challenge-phase1-dataset-rlinf \
OUTPUT_PATH=outputs/seal-water-bottle-cap \
MODEL_DIR=models/Wan-AI/Wan2.2-TI2V-5B \
NUM_MACHINES=8 \
NUM_PROCESSES=64 \
MACHINE_RANK=1 \
MAIN_PROCESS_IP=127.0.0.1 \
MAIN_PROCESS_PORT=29500 \
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_challenge_rlinf.sh
```

`DATASET_BASE` 可以在运行时覆盖，不需要改脚本。数据目录下应包含脚本使用的 `train-data` 和 `val-data` 子目录。默认输出目录是 `outputs/seal-water-bottle-cap`，也可以通过 `OUTPUT_PATH` 覆盖。

## 单机调试

需要先在单机上检查数据和模型路径时，可以覆盖进程数：

```bash
cd diffsynth-studio
DATASET_BASE=../datasets/Challenge-phase1-dataset-rlinf \
MODEL_DIR=models/Wan-AI/Wan2.2-TI2V-5B \
NUM_MACHINES=1 \
NUM_PROCESSES=8 \
MACHINE_RANK=0 \
MAIN_PROCESS_IP=127.0.0.1 \
MAIN_PROCESS_PORT=29500 \
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_challenge_rlinf.sh
```

## 推理与评估

运行 batch size 为 1 的推理脚本：

```bash
cd diffsynth-studio
python examples/wanvideo/model_inference/Wan2.2-TI2V-5B-rlinf-bs_1.py
```
