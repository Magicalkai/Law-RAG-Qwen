import pandas as pd
import json
import os

def create_training_data(csv_path: str, jsonl_path: str):
    """
    读取 CSV 文件，转换为 Qwen Chat 格式的 JSONL 训练数据。
    
    Args:
        csv_path: 原始 FAQ 数据 CSV 路径 (需包含 'title' 和 'reply' 列)。
        jsonl_path: 输出的 JSONL 文件路径。
    """
    print(f"--- 1. 正在读取原始数据: {csv_path} ---")
    if not os.path.exists(csv_path):
        print(f"❌ 错误：找不到文件 {csv_path}。请确保文件存在。")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取 CSV 文件时发生错误: {e}")
        return

    # 假设 CSV 有 'title' (问题) 和 'reply' (答案) 两列
    df = df.dropna(subset=['title', 'reply'])
    dataset = []
    
    # 定义 Qwen 统一的 System Prompt
    system_instruction = "你是一个专业的法律助手，请根据法律法规回答用户的问题。"
    
    for _, row in df.iterrows():
        data_entry = {
            "instruction": system_instruction,
            "input": row['title'],
            "output": row['reply']
        }
        dataset.append(data_entry)

    print(f"--- 2. 正在保存为 JSONL 文件: {jsonl_path} ---")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 数据转换完成！共计 {len(dataset)} 条有效数据。")

if __name__ == "__main__":
    CSV_FILE = "./data/law_faq.csv"  # 假设 CSV 放在 data 文件夹下
    JSONL_FILE = "./law_train_data.jsonl"
    
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    # *注意：你需要手动将 law_faq.csv 放入 data 目录中*
    
    create_training_data(CSV_FILE, JSONL_FILE)
