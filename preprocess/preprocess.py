import pandas as pd
import re
import json
from datetime import datetime
import numpy as np
from collections import Counter
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================
# 配置参数
# ======================
TARGET_USER_SENDER = 'wxid_z2d3h3x279gm22'  # MCMocoder的Sender ID
OUTPUT_FILE = 'mcmocoder_training_data.jsonl'  # 输出文件名
MIN_MESSAGE_LENGTH = 1  # 最小消息长度（过滤掉空消息）
CONTEXT_TIME_GAP = 300  # 上下文时间间隔（秒）
TOP_KEYWORDS_COUNT = 25  # 提取的关键词数量

# ======================
# 1. 数据加载与预处理
# ======================
def load_and_preprocess(file_path):
    """加载CSV文件并进行初步预处理"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 筛选目标用户消息
    target_df = df[df['Sender'] == TARGET_USER_SENDER].copy()
    
    # 转换时间格式
    target_df['CreateTime'] = pd.to_datetime(target_df['StrTime'])
    
    # 按时间排序
    target_df.sort_values('CreateTime', inplace=True)
    
    print(f"找到 {len(target_df)} 条目标用户消息")
    return target_df

# ======================
# 2. 内容清洗函数
# ======================
def clean_content(content, msg_type, sub_type):
    """清洗消息内容"""
    if pd.isna(content):
        return ""
    
    # 处理非文本消息
    if not isinstance(content, str):
        return ""
    
    # 处理表情消息
    if msg_type == 47 or msg_type == 49 or sub_type == 57:
        return "[表情]"
    
    # 处理XML格式的表情消息
    if content.startswith('<msg>') and '<emoji' in content:
        return "[表情]"
    
    # 移除@提及
    content = re.sub(r'@\S+', '', content)
    
    # 移除URL链接
    content = re.sub(r'http[s]?://\S+', '', content)
    
    # 替换特殊格式
    content = content.replace('（）', '')  # 移除中文括号
    
    # 移除多余空格
    content = re.sub(r'\s+', ' ', content).strip()
    
    return content

# ======================
# 3. 特征提取函数
# ======================
def extract_features(text):
    """提取文本特征"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[。！？!?]', text)),
        'punctuation': "",
        'emotion': "neutral",
        'common_phrases': [],
        'top_keywords': []
    }
    
    if not text:
        return features
    
    # 标点分析
    if '~' in text:
        features['punctuation'] = "波浪号"
    elif '!' in text or '！' in text:
        features['punctuation'] = "感叹号"
    elif '（）' in text or '(' in text:
        features['punctuation'] = "括号"
    elif '...' in text or '。。' in text:
        features['punctuation'] = "省略号"
    
    # 情感分析（简化版）
    positive_words = ['好', '开心', '喜欢', '哈哈', '不错', '谢谢']
    negative_words = ['错', '生气', '烦', '讨厌', '问题', '难']
    
    if any(word in text for word in positive_words):
        features['emotion'] = "positive"
    elif any(word in text for word in negative_words):
        features['emotion'] = "negative"
    
    # 提取常见短语（基于实际数据）
    if "我有错" in text:
        features['common_phrases'].append("我有错")
    
    return features

# ======================
# 4. 上下文重建
# ======================
def build_context_blocks(df):
    """构建上下文对话块"""
    dialog_blocks = []
    current_block = []
    last_time = None
    
    for _, row in df.iterrows():
        # 跳过空内容
        if not row['clean_content']:
            continue
            
        # 第一次迭代
        if last_time is None:
            last_time = row['CreateTime']
            current_block.append(row['clean_content'])
            continue
        
        # 计算时间差
        time_diff = (row['CreateTime'] - last_time).total_seconds()
        
        # 如果时间间隔超过阈值，保存当前块
        if time_diff > CONTEXT_TIME_GAP and current_block:
            dialog_blocks.append(" ".join(current_block))
            current_block = []
        
        current_block.append(row['clean_content'])
        last_time = row['CreateTime']
    
    # 添加最后一个块
    if current_block:
        dialog_blocks.append(" ".join(current_block))
    
    print(f"构建了 {len(dialog_blocks)} 个对话块")
    return dialog_blocks

# ======================
# 5. 关键词提取
# ======================
def extract_keywords(texts, top_n=TOP_KEYWORDS_COUNT):
    """提取关键词"""
    # 过滤掉空文本
    texts = [t for t in texts if t.strip()]
    
    # 使用TF-IDF提取关键词
    vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=['的', '了', '在', '是'])
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # 获取特征词
    feature_names = vectorizer.get_feature_names_out()
    
    # 计算每个词的TF-IDF总分
    word_scores = {}
    for col in tfidf_matrix.nonzero()[1]:
        word = feature_names[col]
        score = tfidf_matrix[0, col]
        word_scores[word] = word_scores.get(word, 0) + score
    
    # 获取前N个关键词
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]

# ======================
# 6. 主处理流程
# ======================
def process_chat_data(file_path):
    """主处理函数"""
    # 1. 加载数据
    df = load_and_preprocess(file_path)
    
    # 2. 清洗内容
    df['clean_content'] = df.apply(
        lambda row: clean_content(row['StrContent'], row['Type'], row['SubType']), 
        axis=1
    )
    
    # 3. 过滤空消息
    df = df[df['clean_content'].str.len() >= MIN_MESSAGE_LENGTH]
    
    # 4. 提取特征
    df['features'] = df['clean_content'].apply(extract_features)
    
    # 5. 重建上下文
    dialog_blocks = build_context_blocks(df)
    
    # 6. 提取关键词
    all_texts = [text for block in dialog_blocks for text in block.split()]
    keywords = extract_keywords(dialog_blocks)
    print(f"提取的关键词: {keywords[:10]}...")
    
    # 7. 准备训练数据
    training_data = []
    for block in dialog_blocks:
        # 提取该块的元数据
        features = {
            'length': len(block),
            'word_count': len(block.split()),
            'keyword_score': sum(1 for word in keywords if word in block),
            'common_phrases': ["我有错"] if "我有错" in block else []
        }
        
        entry = {
            "text": block,
            "metadata": features,
            "user_info": {
                "nickname": "MCMocoder",
                "sender_id": TARGET_USER_SENDER,
                "keywords": keywords
            }
        }
        training_data.append(entry)
    
    # 8. 保存为JSONL格式
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"处理完成! 已保存 {len(training_data)} 条训练数据到 {OUTPUT_FILE}")
    
    return training_data

# ======================
# 执行处理
# ======================
if __name__ == "__main__":
    # TODO
    input_file = "your_chat_history.csv"  
    processed_data = process_chat_data(input_file)
    
    # 打印示例数据
    print("\n示例数据:")
    for i, entry in enumerate(processed_data[:2]):
        print(f"条目 {i+1}:")
        print(f"文本: {entry['text'][:50]}...")
        print(f"元数据: {json.dumps(entry['metadata'], indent=2, ensure_ascii=False)}")
        print("-" * 50)
