from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = r"E:\llm\models\qwen\Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base_model, "train/output/sft_output")
model = model.merge_and_unload() # 合并权重，提升推理速度

text = tokenizer.apply_chat_template([{
    "role": "user",
    "content": "Question: Given $\sqrt{x^2+165}-\sqrt{x^2-52}=7$ and $x$ is positive, find all possible values of $x$."
}], add_generation_prompt=True, tokenize=False, enable_thinking=True)

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



