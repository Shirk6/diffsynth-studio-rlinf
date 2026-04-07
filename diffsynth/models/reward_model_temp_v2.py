import torch
import torch.nn as nn
import sys
import os
import functools
from contextlib import contextmanager
from PIL import Image
import numpy as np
from qwen_vl_utils import process_vision_info
# ==================== 库隔离装饰器 ====================

def use_local_libs(method):
    """类方法装饰器：执行方法前切换 sys.path 并清理相关模块缓存。
    确保在 4.40.1 的全局环境中也能加载 extra_libs 里的高版本库。
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "extra_libs"))
        original_sys_path = sys.path[:]
        
        modules_to_reload = ["transformers", "tokenizers", "huggingface_hub", "peft", "qwen_vl_utils", "tqdm"]
        
        if os.path.exists(extra_path):
            if extra_path not in sys.path:
                sys.path.insert(0, extra_path)
            
        saved_modules = {}
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) for m in modules_to_reload):
                saved_modules[mod] = sys.modules.pop(mod)
        
        try:
            return method(self, *args, **kwargs)
        finally:
            sys.path = original_sys_path
            for mod in list(sys.modules.keys()):
                if any(mod.startswith(m) for m in modules_to_reload):
                    sys.modules.pop(mod)
            sys.modules.update(saved_modules)
    return wrapper

@contextmanager
def local_libs_context(extra_path):
    """临时将本地库路径插入 sys.path 的上下文管理器（用于类方法）"""
    original_sys_path = sys.path[:]
    modules_to_reload = ["transformers", "tokenizers", "huggingface_hub", "peft", "qwen_vl_utils", "tqdm"]
    
    if os.path.exists(extra_path):
        sys.path.insert(0, extra_path)
    
    saved_modules = {}
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) for m in modules_to_reload):
            saved_modules[mod] = sys.modules.pop(mod)
            
    try:
        yield
    finally:
        sys.path = original_sys_path
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) for m in modules_to_reload):
                sys.modules.pop(mod)
        sys.modules.update(saved_modules)

# ==================== 常量与配置 ====================

NUM_CLASSES = 11   # reward levels 0, 1, …, 10
PROMPT_TEMPLATE = (
    "You are a reward model for a robot arm performing the following manipulation task:\n"
    "\"{task}\"\n\n"
    "The four images are consecutive observations of the robot workspace in chronological order "
    "(earliest → latest). Based on the visual progress toward completing the task, "
    "rate the current state on a scale from 0 to 10:\n"
    "  0 = no progress at all\n"
    "  1–9 = partial progress (linearly increasing toward success)\n"
    "  10 = task fully completed\n\n"
    "Reply with a single integer from 0 to 10."
)

# ==================== 模型类 ====================

class QwenVLRewardModel(nn.Module):
    @use_local_libs
    def __init__(
        self,
        model_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.num_classes = num_classes
        import os
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["DISABLE_TRANSFORMERS_VERSION_CHECK"] = "1"
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        hidden_size = self.backbone.config.text_config.hidden_size

        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        if lora_target_modules:
            self.backbone = get_peft_model(self.backbone, lora_cfg)

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes),
        )
        for m in self.reward_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        self.loss_fn = nn.CrossEntropyLoss()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    @use_local_libs
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        labels: torch.LongTensor = None,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]            # (B, L, H)
        seq_lens = attention_mask.sum(dim=1) - 1             # (B,)
        last_hidden = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_lens,
        ].float()                                            # (B, H)

        logits = self.reward_head(last_hidden)               # (B, 11)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.long())

        return logits, loss

    @torch.no_grad()
    # @use_local_libs
    def predict(self, *args, **kwargs) -> torch.Tensor:
        logits, _ = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)
    
    @torch.no_grad()
    # @use_local_libs
    def score_4frames(self, processor, images: list, task: str, device="cuda"):
        assert len(images) == 4, "Need exactly 4 frames"
        prompt = PROMPT_TEMPLATE.format(task=task)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)

        self.eval()
        logits, _ = self(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        probs = logits.softmax(dim=-1).squeeze(0).cpu()
        pred  = int(logits.argmax(dim=-1).item())
        return pred, probs

    @torch.no_grad()
    # @use_local_libs
    def predict_rew_no_batch(self, video_tensor, task_descriptions, chunk_size):


        num_envs, T_total, C, V, H, W = video_tensor.shape
        device = video_tensor.device
        all_rewards = torch.zeros((num_envs, chunk_size), device=device)

        def to_pil(t):
            img = (t * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            return Image.fromarray((img * 255).astype(np.uint8))

        for env_idx in range(num_envs):
            task = task_descriptions[env_idx]
            for i in range(chunk_size):
                target_idx = T_total - chunk_size + i
                indices = [max(0, target_idx - 3), max(0, target_idx - 2), 
                           max(0, target_idx - 1), target_idx]
                print(f"Env {env_idx}, chunk {i}: scoring frames {indices} for task '{task}'")
                window_pils = [to_pil(video_tensor[env_idx, idx, :, 0, :, :]) for idx in indices]
                
                pred, _ = self.score_4frames(self.processor, window_pils, task, device)
                all_rewards[env_idx, i] = pred

        return all_rewards

    @torch.no_grad()
    def predict_rew(self, video_tensor, task_descriptions, chunk_size, 
                            batch_size=32):  # 新增 batch_size 参数
        """
        并行版本：按 batch 处理，减少 GPU 空闲等待
        """
        num_envs, T_total, C, V, H, W = video_tensor.shape
        device = video_tensor.device
        all_rewards = torch.zeros((num_envs, chunk_size), device=device)
        
        # 1. 预生成所有样本的 metadata，不重复计算
        all_samples = []
        for env_idx in range(num_envs):
            task = task_descriptions[env_idx]
            for i in range(chunk_size):
                target_idx = T_total - chunk_size + i
                indices = [max(0, target_idx - 3), max(0, target_idx - 2), 
                        max(0, target_idx - 1), target_idx]
                all_samples.append({
                    'env_idx': env_idx,
                    'chunk_i': i,
                    'task': task,
                    'indices': indices
                })
        
        # 2. 按 batch 处理
        def to_pil(t):
            img = (t * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            return Image.fromarray((img * 255).astype(np.uint8))
        
        self.eval()
        
        for batch_start in range(0, len(all_samples), batch_size):
            batch_samples = all_samples[batch_start:batch_start + batch_size]
            current_bs = len(batch_samples)
            
            # 2.1 构造 batch 的 messages
            batch_messages = []
            for sample in batch_samples:
                env_idx = sample['env_idx']
                indices = sample['indices']
                task = sample['task']
                
                prompt = PROMPT_TEMPLATE.format(task=task)
                # 提取4帧图像
                window_pils = [to_pil(video_tensor[env_idx, idx, :, 0, :, :]) 
                            for idx in indices]
                
                content = [{"type": "image", "image": img} for img in window_pils]
                content.append({"type": "text", "text": prompt})
                batch_messages.append([{"role": "user", "content": content}])
            
            # 2.2 Batch 预处理（processor 支持 batch）
            texts = [self.processor.apply_chat_template(msgs, tokenize=False, 
                                                        add_generation_prompt=True) 
                    for msgs in batch_messages]
            
            # 收集所有 images
            all_images = []
            for msgs in batch_messages:
                img_inputs, _ = process_vision_info(msgs)
                all_images.extend(img_inputs)
            
            # 2.3 Batch forward
            inputs = self.processor(
                text=texts,
                images=all_images,
                return_tensors="pt",
                padding=True,  # 关键：batch 需要 padding
            ).to(device)
            
            # 2.4 一次 forward 处理整个 batch
            logits, _ = self(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            
            # 2.5 分配结果
            preds = logits.argmax(dim=-1)
            for idx, sample in enumerate(batch_samples):
                env_idx = sample['env_idx']
                chunk_i = sample['chunk_i']
                all_rewards[env_idx, chunk_i] = preds[idx]
        
        return all_rewards

    # @use_local_libs
    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.backbone.save_pretrained(save_path)
        torch.save(self.reward_head.state_dict(), f"{save_path}/reward_head.pt")
        torch.save({"num_classes": self.num_classes}, f"{save_path}/reward_cfg.pt")


    # @classmethod
    # @use_local_libs
    # def load_pretrained(cls, base_model_path: str, lora_ckpt_path: str, **kwargs):
    #     extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "extra_libs"))
        
    #     with local_libs_context(extra_path):
    #         from peft import PeftModel
    #         cfg = torch.load(f"{lora_ckpt_path}/reward_cfg.pt", map_location="cpu")
    #         model = cls(base_model_path, num_classes=cfg["num_classes"], lora_target_modules=[], **kwargs)
    #         model.backbone = PeftModel.from_pretrained(
    #             model.backbone, lora_ckpt_path
    #         )
    #         rh = torch.load(f"{lora_ckpt_path}/reward_head.pt", map_location="cpu")
    #         model.reward_head.load_state_dict(rh)
    #         return model

    @classmethod
    @use_local_libs
    def load_pretrained(cls, base_model_path: str, lora_ckpt_path: str, **kwargs):
        import sys
        import importlib.util

        print("\n===== IMPORT DEBUG START =====")
        print("sys.path 前5个：")
        for p in sys.path[:5]:
            print("  ", p)

        spec = importlib.util.find_spec("peft")
        print("\npeft spec:", spec)
        if spec:
            print("peft origin:", spec.origin)

        spec_t = importlib.util.find_spec("transformers")
        print("\ntransformers spec:", spec_t)
        if spec_t:
            print("transformers origin:", spec_t.origin)

        print("===== IMPORT DEBUG END =====\n")
        import peft
        print("PEFT PATH:", peft.__file__)
        from peft import PeftModel
        cfg = torch.load(f"{lora_ckpt_path}/reward_cfg.pt", map_location="cpu")
        model = cls(base_model_path, num_classes=cfg["num_classes"], lora_target_modules=[], **kwargs)
        model.backbone = PeftModel.from_pretrained(
            model.backbone, lora_ckpt_path
        )
        rh = torch.load(f"{lora_ckpt_path}/reward_head.pt", map_location="cpu")
        model.reward_head.load_state_dict(rh)
        return model

if __name__ == "__main__":
    # Test script for QwenVLRewardModel
    import glob
    import sys
    import shutil
    
    base_model_path = "/mnt/project_rlinf/ztx/models/Qwen2.5-VL-3B-Instruct"
    lora_path = "/mnt/project_rlinf/ztx/reward_model/checkpoints/best"
    image_dir = "/mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_spatial/base_policy_rollout/val_data_processed/step_0_seed_0_traj_0/images"
    task_desc = "pick up the black bowl next to the cookie box and place it on the plate"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    # 1. Load Model
    model = QwenVLRewardModel.load_pretrained(base_model_path, lora_path).to(device).eval()
    
    # 2. Load ALL images from the directory
    image_files = sorted(glob.glob(os.path.join(image_dir, "frame_*.png")))
    if not image_files:
        print(f"No images found in {image_dir}")
        sys.exit(1)
    
    total_frames = len(image_files)
    print(f"Found {total_frames} images. Testing continuous reward prediction...")
    
    pils = [Image.open(f).convert("RGB") for f in image_files]
    frames_np = []
    for img in pils:
        img_np = np.array(img).transpose(2, 0, 1) # [C, H, W]
        img_np = (img_np / 255.0) * 2.0 - 1.0     # [-1, 1]
        frames_np.append(img_np)
    
    # shape: [T, C, H, W]
    trajectory_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).float()
    
    # 3. Simulate num_envs = 8 by repeating the trajectory
    num_envs = 8
    # video_tensor shape expected by predict_rew: [batch, T, C, V, H, W]
    # Here we treat the entire trajectory as the 'chunk' to get rewards for all frames
    # C=3, V=1 (view)
    video_tensor = trajectory_tensor.unsqueeze(0).unsqueeze(3).repeat(num_envs, 1, 1, 1, 1, 1).to(device)
    
    # 4. Predict Rewards for the whole trajectory
    # WanEnv's predict_rew calculates rewards for the LAST chunk_size frames.
    # To get rewards for all frames, we set chunk_size = total_frames
    chunk_size = total_frames
    task_descriptions = [task_desc] * num_envs
    
    print(f"Running batch predict_rew for {num_envs} envs, chunk_size={chunk_size}...")
    with torch.no_grad():
        rewards = model.predict_rew(video_tensor, task_descriptions, chunk_size=chunk_size, batch_size=16)
    
    print("\n" + "="*50)
    print(f"Task: {task_desc}")
    print(f"Number of Envs: {num_envs}")
    print(f"Trajectory Length: {total_frames}")
    
    # Print rewards for the first environment (all envs are identical here)
    env0_rewards = rewards[0].cpu().numpy()
    print(f"\nRewards for Env 0 (Scale 0-10):")
    for idx, r in enumerate(env0_rewards):
        print(f"Frame {idx:03d}: {r:.1f}")
        
    print("\nBatch rewards summary (first 5 envs, first 5 rewards each):")
    print(rewards[:5, :5].cpu().numpy())
    print("="*50)
