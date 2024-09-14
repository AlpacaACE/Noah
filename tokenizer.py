import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os
import argparse

# 设置随机种子
random.seed(42)

def read_texts_from_jsonl(file_path):
    """从JSONL文件中读取文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

def train_tokenizer(data_path, tokenizer_dir, vocab_size=6400):
    """训练并保存自定义的Tokenizer"""
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<unk>", "<s>", "</s>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,  # 确保三个special token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据并训练tokenizer
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

def eval_tokenizer(tokenizer_dir):
    """评估自定义的Tokenizer"""
    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": '你好'},
        {"role": "assistant", "content": '请问有什么需要帮助的？'},
        {"role": "user", "content": '今天天气怎么样？'},
        {"role": "assistant", "content": '你好，我无法查询到天气信息，请您通过天气网站查询。'}
    ]

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取词汇表大小（不包括特殊符号）
    print('tokenizer词表大小：', tokenizer.vocab_size)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('实际词表长度：', actual_vocab_size)

    new_prompt = '菩萨引众同入里面，与玉帝礼毕，又与老君、王母相见，各坐下。便问：“蟠桃盛会如何？”玉帝道：“每年请会，喜喜欢欢，今年被妖猴作乱，甚是虚邀也。'
    print(new_prompt)
    model_inputs = tokenizer(new_prompt)

    print(model_inputs)
    print('长度：', len(model_inputs['input_ids']))

    input_ids_ = model_inputs['input_ids']

    response = tokenizer.decode(input_ids_)
    print(response, end='')

def main():
    # 使用argparse管理命令行参数
    parser = argparse.ArgumentParser(description="Tokenizer Trainer and Evaluator")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, 
                        help="运行模式: 'train' 进行训练, 'eval' 进行评估")
    parser.add_argument('--data_path', type=str, default='./dataset/tokenizer_data.json',
                        help="训练数据路径，适用于训练模式")
    parser.add_argument('--tokenizer_dir', type=str, default='./minimind_tokenizer',
                        help="Tokenizer保存路径")
    parser.add_argument('--vocab_size', type=int, default=6400,
                        help="词汇表大小，适用于训练模式")
    
    args = parser.parse_args()

    if args.mode == 'train':
        train_tokenizer(args.data_path, args.tokenizer_dir, args.vocab_size)
    elif args.mode == 'eval':
        eval_tokenizer(args.tokenizer_dir)

if __name__ == '__main__':
    main()

'''
# 训练tokenizer
python tokenizer.py --mode train --data_path ./dataset/tokenizer_data.json --tokenizer_dir ./tokenizer --vocab_size 6400

# 评估tokenizer
python tokenizer.py --mode eval --tokenizer_dir ./tokenizer

# 评估mistral tokenizer
python tokenizer.py --mode eval --tokenizer_dir mistralai/Mistral-7B-v0.1
'''