# agentic-rl-practice
本项目实现了agentic-rl训练，从模型的SFT训练再到模型的GRPO训练，两个阶段的数据集都是：openai/gsm8k，项目的部分源码参考自datawhale社区的 [hello-agent项目](https://github.com/jjyaoao/helloagents)，训练的模型是Qwen3-1.7B
## 模型下载
```shell
# 修改全局配置 common/global_config.py
python download_model_ms.py
```
## 模型训练
### SFT
+ 命令
```shell
cd train
CUDA_VISIBLE_DEVICES=0 python train_sft.py
```
+ 训练过程曲线
![训练曲线](image/sft_train.png)


