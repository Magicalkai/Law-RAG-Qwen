import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# =========================== 配置区域 ===========================
# 请修改为你的本地模型路径和数据路径
BASE_MODEL_PATH = "D:/LLM/Pretrained_models/Qwen/Qwen3-0___6B/" 
DATA_PATH = "./law_train_data.jsonl"
OUTPUT_DIR = "./final_law_lora" # 最终保存 LoRA 权重的目录
MAX_LENGTH = 384
# ===============================================================

def load_data_and_tokenize(data_path, tokenizer):
    """加载 JSONL 数据并进行 tokenize 处理"""
    print(f"--- 1. 正在加载和处理数据: {data_path} ---")
    
    # 使用 pandas/Dataset 而不是 load_dataset("json") 来增强兼容性
    try:
        df = pd.read_json(data_path, lines=True)
        dataset = Dataset.from_pandas(df)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

    def process_func(example):
        # 遵循 Qwen 的 ChatML 格式：<|im_start|>role\ncontent<|im_end|>\n
        
        # 构造 Prompt
        instruction_text = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        response_text = f"{example['output']}"
        
        instruction = tokenizer(instruction_text, add_special_tokens=False)
        response = tokenizer(response_text, add_special_tokens=False)
        
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        
        # Label mask: Instruction 部分设为 -100
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
        
        # 截断
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names, desc="Tokenizing dataset")
    print(f"✅ 数据 Tokenize 完成！共 {len(tokenized_ds)} 条数据。")
    return tokenized_ds

def train():
    print("--- 2. 正在加载 Tokenizer 和 模型 ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    # Qwen-7B/14B/72B 默认 pad_token 是 None，需手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16, # 建议使用 bf16 (RTX 30系以上)
        trust_remote_code=True
    )

    # --- LoRA 配置 ---
    print("--- 3. 配置 LoRA ---")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # Qwen 模型的常见 target modules
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # --- 显存优化：启用梯度检查点 ---
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # --- 数据处理 ---
    tokenized_ds = load_data_and_tokenize(DATA_PATH, tokenizer)
    if tokenized_ds is None:
        return

    # --- 训练参数配置 ---
    print("--- 4. 配置训练参数 ---")
    args = TrainingArguments(
        output_dir="./law_lora_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1, 
        gradient_checkpointing=True, # 使用梯度检查点来节省显存
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        bf16=True, 
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        # 更多参数可根据实际情况调整...
    )

    # --- 训练器初始化与启动 ---
    print("--- 5. 开始 LoRA 微调训练 ---")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # --- 保存 LoRA 权重 ---
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA 模型微调完毕并保存至: {OUTPUT_DIR}！")

if __name__ == "__main__":
    train()
