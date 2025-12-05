import torch
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer # 使用 sentence_transformers 库简化 BGE 加载

# =========================== 配置区域 ===========================
# 检索模型：BGE-Small (自动下载)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512 # BGE-Small 的维度
# 生成模型路径 (请确保路径正确)
BASE_MODEL_PATH = "D:/LLM/Pretrained_models/Qwen/Qwen3-0___6B/" 
LORA_PATH = "./final_law_lora"
CSV_PATH = "./data/law_faq.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")
# ===============================================================

# --- 全局变量 ---
retrieval_model = None
retrieval_tokenizer = None
knowledge_index = None
data_df = None
qwen_model = None
qwen_tokenizer = None

def get_embeddings(sentences):
    """输入文本列表，输出归一化的向量"""
    global retrieval_model, retrieval_tokenizer
    
    # 推荐对查询使用指令前缀
    is_query = len(sentences) == 1
    if is_query:
        sentences = [f"为这个句子生成表示用于检索相关文章: {sentences[0]}"]

    encoded_input = retrieval_tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        model_output = retrieval_model(**encoded_input)
        # BGE 使用 CLS token (第一个 token) 作为句向量
        sentence_embeddings = model_output.last_hidden_state[:, 0]
        # 必须进行 L2 归一化 (Normalization)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
    return sentence_embeddings.cpu().numpy()

def initialize_models_and_index():
    """初始化所有模型和 FAISS 索引"""
    global retrieval_model, retrieval_tokenizer, knowledge_index, data_df, qwen_model, qwen_tokenizer

    # --- 1. 加载检索模型 (BGE-Small) ---
    print(">>> [1/4] 正在加载 BGE 检索模型...")
    # 使用 SentenceTransformer 库来简化加载过程
    try:
        model_st = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        retrieval_model = model_st._first_module() # 获取底层的 AutoModel
        retrieval_tokenizer = model_st.tokenizer
    except Exception as e:
        print(f"❌ 加载 BGE 模型失败，请检查网络或路径: {e}")
        return False

    # --- 2. 构建 FAISS 索引 (数据库向量化) ---
    print(">>> [2/4] 正在构建向量库...")
    try:
        data_df = pd.read_csv(CSV_PATH)
        data_df = data_df.dropna(subset=["title", "reply"])
        questions = data_df["title"].to_list()
    except FileNotFoundError:
        print(f"❌ 错误：找不到 {CSV_PATH}")
        return False

    all_vectors = []
    batch_size = 64
    for i in range(0, len(questions), batch_size):
        batch_sentences = questions[i : i + batch_size]
        batch_emb = get_embeddings(batch_sentences)
        all_vectors.append(batch_emb)
        
    knowledge_vectors = np.concatenate(all_vectors, axis=0)

    knowledge_index = faiss.IndexFlatIP(EMBEDDING_DIM) 
    knowledge_index.add(knowledge_vectors)
    print(f"    ✅ 索引构建完成！库容量: {knowledge_index.ntotal}, 向量维度: {EMBEDDING_DIM}")

    # --- 3. 加载生成模型 (Qwen + LoRA) ---
    print(">>> [3/4] 正在加载 Qwen + LoRA...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        qwen_model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()
        qwen_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"❌ 加载 Qwen/LoRA 模型失败，请检查路径: {e}")
        return False

    return True

def rag_chat(user_query, k=3):
    """执行 RAG 问答流程"""
    global knowledge_index, data_df, qwen_model, qwen_tokenizer
    
    if knowledge_index is None:
        print("模型和索引未初始化！")
        return ""

    # --- Step A: 检索 ---
    q_vector = get_embeddings([user_query]) 
    scores, indexes = knowledge_index.search(q_vector, k=k)
    
    retrieved_text = []
    print(f"\n🔍 [检索详情] 查询: {user_query}")
    for i, idx in enumerate(indexes[0]):
        if idx != -1 and idx < len(data_df):
            row = data_df.iloc[idx]
            score = scores[0][i]
            print(f"    Rank {i+1}: {row['title']} (Score: {score:.4f})")
            # 构造上下文格式
            retrieved_text.append(f"【案例{i+1}】\n问题：{row['title']}\n回答：{row['reply']}")
            
    context_str = "\n\n".join(retrieved_text)

    # --- Step B: 生成 ---
    # 构造 Prompt (遵循 Qwen ChatML + RAG)
    system_prompt = f"""你是一个专业的法律智能助手。请参考下面的【已知案例库】，准确回答用户的问题。
如果【已知案例库】中包含相关信息，请优先基于案例库回答。
如果案例库不相关，请利用你的专业法律知识进行回答。

【已知案例库】：
{context_str}"""
    
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
    
    with torch.inference_mode():
        outputs = qwen_model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.6, 
            top_p=0.9,
            do_sample=True, # 启用采样
            repetition_penalty=1.1
        )
        
    response = qwen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    if initialize_models_and_index():
        test_queries = [
            "寻衅滋事罪一般怎么判？", 
            "合同法中关于违约金的规定是什么？",
            "北京今天天气怎么样？" # 测试无关问题，看模型是否能回答或拒绝
        ]
        
        for query in test_queries:
            print("\n" + "="*60)
            print(f"👤 用户问题: {query}")
            final_answer = rag_chat(query)
            print("-"*60)
            print(f"🤖 AI 回答:\n{final_answer.strip()}")
            print("="*60)
