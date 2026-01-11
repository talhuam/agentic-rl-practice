from datasets import load_dataset
from typing import Optional, Dict, Any


class GSM8kDataset:
    def __init__(
            self,
            split: str = "train",
            max_samples: Optional[int] = None,
            format_type: str = "sft",
            tokenizer = None
    ):
        assert format_type in ["sft", "rl"], "only support sft/rl format"
        self.split = split
        self.max_samples = max_samples
        self.format_type = format_type
        self.tokenizer = tokenizer

        print(f"📥 加载 GSM8K 数据集 (split={split})...")
        self.dataset = load_dataset("openai/gsm8k", "main", split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            print(f"   使用 {len(self.dataset)} 个样本（限制: {max_samples}）")
        else:
            print(f"   加载了 {len(self.dataset)} 个样本")

    def format_for_sft(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """格式化为SFT训练格式"""
        question = sample["question"]
        answer = sample["answer"]

        if "####" in answer:
            reasoning, final_answer = answer.split("####")
            reasoning.strip()
            final_answer.strip()
        else:
            reasoning = answer
            final_answer = ""

        prompt = f"Question: {question}\n\nLet`s solve this step by step:\n"
        completion = f"{reasoning}\n\nFinal Answer: {final_answer}"
        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion
        }

    def format_for_rl(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """格式化为GRPO格式"""
        question = sample["question"]
        answer = sample["answer"]
        # rl只提取最终答案，推理过程不需要
        if "####" in answer:
            _, final_answer = answer.split("#####")
            final_answer = final_answer.strip()
        else:
            final_answer = answer.strip()

        prompt_content = f"Question: {question}\n\nLet`s solve this step by step:"

        # 应用chat_template
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = prompt_content

        return {
            "prompt": prompt,
            "ground_truth": final_answer,
            "question": question,
            "full_answer": answer
        }

    def get_dataset(self):
        if self.format_type == "sft":
            return self.dataset.map(
                self.format_for_sft,
                remove_columns=self.dataset.column_names
            )
        else:
            return self.dataset.map(
                self.format_for_rl,
                remove_columns=self.dataset.column_names
            )

    def __len__(self):
        """获取数据集大小"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.dataset[idx]
        if self.format_type == "sft":
            return self.format_for_sft(sample)
        else:
            return self.format_for_rl(sample)


if __name__ == '__main__':
    ds = GSM8kDataset(split="train")
    question = ds[0]["prompt"]
    answer = ds[0]["completion"]
    print(f"<question start>\n{question}\n</question end>\n\n<answer start>\n{answer}\n<answer end>")