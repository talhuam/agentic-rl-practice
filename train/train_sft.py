import os.path
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import TrainingConfig, setup_training_environment, DetailedLoggingCallback
from common.global_config import model_name, sft_output_path, model_root_path
from dataset import GSM8kDataset


def train():
    config = TrainingConfig(
        model_name=model_name,
        output_dir=sft_output_path,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_liger_kernel=False,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=2048
    )

    # 准备训练环境
    setup_training_environment(config)

    # 加载模型和分词器
    model_path = os.path.join(model_root_path, model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 训练数据(prompt-completion)
    dataset = GSM8kDataset("train", format_type="sft", tokenizer=tokenizer).get_dataset()

    # 训练设置
    peft_config = LoraConfig(
        r = config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout
    )

    sft_config = SFTConfig(
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
        gradient_checkpointing=config.gradient_checkpointing,
        max_length=config.max_length,
        report_to="tensorboard",
        use_liger_kernel=config.use_liger_kernel
    )

    # 日志回调
    total_steps = (
        len(dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    ) * config.num_train_epochs
    logging_callback = DetailedLoggingCallback(total_steps, config.num_train_epochs)

    # 训练器
    trainer = SFTTrainer(
        model = model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[logging_callback],
        peft_config=peft_config # lora训练
    )

    print("\n🚀 开始SFT训练...")
    print(f"{'=' * 80}\n")
    trainer.train()
    print(f"\n{'=' * 80}")
    print("✅ SFT训练完成")

    # 保存模型
    trainer.save_model()


if __name__ == '__main__':
    train()