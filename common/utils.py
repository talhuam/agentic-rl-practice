from typing import Optional, Dict, Any

import os
from dataclasses import dataclass, field
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 模型配置
    model_name: str = "Qwen/Qwen3-0.6B"
    model_revision: Optional[str] = None

    # 训练配置
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # RL特定配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # 硬件配置
    use_fp16: bool = True
    use_bf16: bool = False
    gradient_checkpointing: bool = True

    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # 监控配置
    use_tensorboard: bool = True

    # 其他配置
    seed: int = 42
    max_length: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


def setup_training_environment(config: TrainingConfig):
    """
    初始化训练环境
    """
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    # 设置随机种子
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    except ImportError:
        pass

    random.seed(config.seed)
    np.random.seed(config.seed)

    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"✅ 训练环境设置完成")
    print(f"   - 输出目录: {config.output_dir}")
    print(f"   - 随机种子: {config.seed}")
    print(f"   - 模型: {config.model_name}")


class DetailedLoggingCallback(TrainerCallback):
    """详细日志回调
    在训练过程中输出更详细的日志信息,包括:
    - Epoch/Step进度
    - Loss
    - Learning Rate
    - Reward (GRPO)
    - KL散度 (GRPO)
    """
    def __init__(self, total_steps: int, num_epochs: int):
        super().__init__()
        self.total_steps = total_steps
        self.num_epochs = num_epochs
        self.current_epoch = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """日志回调"""
        logs = kwargs.get("logs", None)
        if logs is None:
            return

        if state.epoch is not None:
            self.current_epoch = state.epoch

        log_parts = []
        # Epoch和Step信息
        if self.num_epochs:
            log_parts.append(f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        if state.global_step and self.total_steps:
            log_parts.append(f"Step {state.global_step}/{self.total_steps}")
        elif state.global_step:
            log_parts.append(f"Step {state.global_step}")

        # Loss
        if "loss" in logs:
            log_parts.append(f"Loss: {logs['loss']:.4f}")

        # Learning Rate
        if "learning_rate" in logs:
            log_parts.append(f"LR: {logs['learning_rate']:.2e}")

        # GRPO特定指标
        if "rewards/mean" in logs:
            log_parts.append(f"Reward: {logs['rewards/mean']:.4f}")

        if "objective/kl" in logs:
            log_parts.append(f"KL: {logs['objective/kl']:.4f}")

        # 输出日志
        if log_parts:
            print(" | ".join(log_parts))

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Epoch结束回调"""
        print(f"{'=' * 80}")
        print(f"✅ Epoch {self.current_epoch + 1} 完成")
        print(f"{'=' * 80}\n")

