import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common.global_config import model_root_path, model_name
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, default="sft", choices=["sft", "grpo"])
    parser.add_argument("-r", "--reasoning", action="store_true")
    args = parser.parse_args()

    model_path = os.path.join(model_root_path, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype="auto", device_map="auto")

    if args.mode == "sft":
        model = PeftModel.from_pretrained(base_model, "train/output/sft_output")
    else:
        model = PeftModel.from_pretrained(base_model, "train/output/grpo_output")
    model = model.merge_and_unload() # 合并权重，提升推理速度

    text = tokenizer.apply_chat_template([{
        "role": "user",
        "content": f"Question: {args.question}"
    }], add_generation_prompt=True, tokenize=False, enable_thinking=args.reasoning)

    print(f"apply_chat_template result:\n{text}")

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096 # 控制生成的最大长度
    )
    # 剔除掉prompt部分
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)



