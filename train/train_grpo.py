from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import TrainingConfig, setup_training_environment, DetailedLoggingCallback
from common.global_config import model_name, model_root_path, grpo_output_path
from common.reward_functions import create_accuracy_reward, create_step_reward
from dataset import GSM8kDataset


def train():
    config = TrainingConfig(
        output_dir=grpo_output_path,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=2048,
        learning_rate=5e-5,
        logging_steps=1
    )

    # 准备训练环境
    setup_training_environment(config)

    # 加载模型和分词器
    base_model_path = os.path.join(model_root_path, model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, "output/sft_output")

    # 设置lora模块可训练
    for name, params in model.named_parameters():
        if "lora" in name:
            params.requires_grad_(True)

    # 统计模型可训练参数
    total_params = sum([param.numel() for param in model.parameters()])
    trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f"GRPO可训练参数: {trainable_params:,} ({trainable_params / total_params:.4%})")

    # 准备训练数据
    dataset = GSM8kDataset(split="train", format_type="rl", tokenizer=tokenizer, max_samples=2000).get_dataset()

    # 创建奖励函数
    acc_fun = create_accuracy_reward()
    step_func = create_step_reward(acc_fun) # 准确性奖励和步骤奖励组合

    # 训练设置
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        fp16=config.use_fp16,
        bf16=config.use_bf16,
        report_to="tensorboard",
        num_generations=4, # 一条样本生成多少个回答，默认为4
        remove_unused_columns=False # 保留数据集中所有的列,
    )

    # 日志回调
    total_steps = (
          len(dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    ) * config.num_train_epochs * grpo_config.num_generations
    logging_callback = DetailedLoggingCallback(total_steps, config.num_train_epochs)

    # 训练器
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=step_func,
        callbacks=[logging_callback],
        processing_class=tokenizer
    )

    print("\n🚀 开始GRPO训练...")
    print(f"{'=' * 80}\n")
    trainer.train()
    print(f"\n{'=' * 80}")
    print("✅ SFT训练完成")

    # 保存模型
    trainer.save_model()


if __name__ == '__main__':
    train()