import pandas as pd
import re
import json
import logging
import numpy as np
from datetime import datetime, timedelta
import os
import hashlib
from collections import defaultdict
import os

# 设置使用hf-mirror.com镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ======================
# 配置参数
# ======================
class Config:
    TARGET_USER_ID = 'wxid_z2d3h3x279gm22'  # 目标用户ID
    SYSTEM_PROMPT = "你正在扮演MCMocoder在群聊中的角色。请用自然、真实的语气回复其他群成员。"

    # 对话处理
    CONTEXT_WINDOW = timedelta(minutes=30)  # 同一对话最大时间间隔
    MIN_TURNS = 1  # 最小对话轮次 (user-assistant对)

    # 清洗设置
    MAX_MSG_LENGTH = 100
    MIN_MSG_LENGTH = 2

    # 语义模型设置
    SEMANTIC_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    SEMANTIC_THRESHOLD = 0.4  # 语义相似度阈值
    MIN_COHERENCE_SCORE = 0.5  # 最小连贯性得分
    MIN_RELEVANCE_SCORE = 0.5  # 最小相关句子得分

    # 缓存设置
    CACHE_DIR = "model_cache"


# ======================
# 日志设置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversation_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# ======================
# 路径处理函数
# ======================
def safe_path(path):
    """处理Windows路径转义问题"""
    return os.path.normpath(path)


# ======================
# 消息清洗函数 (增强版)
# ======================
def clean_content(content):
    """严格清洗消息内容"""
    if not isinstance(content, str):
        return None

    # 移除XML/技术性内容
    if re.match(r'^\s*(<[^>]+>|{[^}]+})', content):
        return None

    # 移除特殊内容
    content = re.sub(r'(@\S+|http\S+|[a-f0-9]{8,})', '', content)
    content = re.sub(r'\[.*?\]|\s+', ' ', content).strip()

    # 长度检查
    if len(content) < Config.MIN_MSG_LENGTH:
        return None
    if len(content) > Config.MAX_MSG_LENGTH:
        content = content[:Config.MAX_MSG_LENGTH] + "..."

    return content


# ======================
# 语义分析器
# ======================
class SemanticAnalyzer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 创建缓存目录
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        logger.info(f"正在加载语义模型: {Config.SEMANTIC_MODEL}")
        try:
            self.model = SentenceTransformer(
                Config.SEMANTIC_MODEL,
                cache_folder=Config.CACHE_DIR
            )
            logger.info("语义模型加载成功")
            self.initialized = True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self.initialized = False

    def encode(self, texts):
        """编码文本为向量"""
        if not self.initialized or not texts:
            return None

        try:
            return self.model.encode(texts)
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            return None

    def calculate_coherence(self, conversation):
        """计算对话连贯性得分"""
        if not self.initialized:
            return 1.0  # 如果模型未初始化，默认通过

        # 提取所有非系统消息
        messages = [msg["content"] for msg in conversation if msg["role"] != "system"]
        if len(messages) < 2:
            return 0.0

        # 编码所有消息
        embeddings = self.encode(messages)
        if embeddings is None:
            return 0.0

        # 计算相邻消息的相似度
        similarities = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            similarities.append(sim)

        # 计算平均相似度
        return np.mean(similarities) if similarities else 0.0

    def is_coherent(self, conversation):
        """检查对话是否连贯"""
        score = self.calculate_coherence(conversation)
        logger.debug(f"对话连贯性得分: {score:.2f}")
        return score >= Config.MIN_COHERENCE_SCORE

    def check_pair_relevance(self, user_msg, assistant_msg):
        """检查用户消息和助理回复的相关性"""
        if not self.initialized:
            return True  # 如果模型未初始化，默认相关

        embeddings = self.encode([user_msg, assistant_msg])
        if embeddings is None or len(embeddings) < 2:
            return False

        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity >= Config.SEMANTIC_THRESHOLD

    def filter_relevant_sentences(self, user_content, assistant_content):
        """筛选与助理回复相关的用户消息句子"""
        if not self.initialized:
            return user_content  # 如果模型未初始化，返回原始内容

        # 分割用户消息为句子
        sentences = [s.strip() for s in re.split(r'[\n。！？!?]', user_content) if s.strip()]
        if len(sentences) <= 1:
            return user_content  # 如果只有一句话，直接返回

        # 添加助理回复作为参考
        all_texts = sentences + [assistant_content]

        # 编码所有文本
        embeddings = self.encode(all_texts)
        if embeddings is None:
            return user_content

        # 计算每个句子与助理回复的相似度
        assistant_embedding = embeddings[-1]
        similarities = []
        for i, sent in enumerate(sentences):
            sim = cosine_similarity([embeddings[i]], [assistant_embedding])[0][0]
            similarities.append(sim)

        # 筛选相关句子
        relevant_sentences = []
        for i, sent in enumerate(sentences):
            if similarities[i] >= Config.MIN_RELEVANCE_SCORE:
                relevant_sentences.append(sent)

        # 如果没有找到相关句子，返回相似度最高的句子
        if not relevant_sentences:
            max_idx = np.argmax(similarities)
            relevant_sentences = [sentences[max_idx]]

        # 返回筛选后的内容
        return "\n".join(relevant_sentences)


# ======================
# 对话构建器 (增强版)
# ======================
class EnhancedConversationBuilder:
    def __init__(self):
        self.reset()
        self.semantic_analyzer = SemanticAnalyzer.get_instance()

    def reset(self):
        self.messages = []  # 原始消息列表
        self.last_time = None
        self.participants = set()  # 记录参与用户
        self.user_msg_buffer = []  # 缓存连续用户消息

    def add_message(self, role, content, timestamp, sender):
        """添加消息并检查逻辑连贯性"""
        # 角色必须是user或assistant
        if role not in ["user", "assistant"]:
            return False

        # 处理用户消息
        if role == "user":
            self.user_msg_buffer.append(content)
            self.participants.add(sender)
            self.last_time = timestamp
            return True

        # 处理助理消息 (必须是目标用户)
        if role == "assistant" and sender != Config.TARGET_USER_ID:
            return False

        # 必须有前置用户消息
        if not self.user_msg_buffer:
            return False

        # 合并用户消息
        user_content = "\n".join(self.user_msg_buffer)

        # 筛选与助理回复相关的用户消息句子
        if self.semantic_analyzer.initialized:
            filtered_user_content = self.semantic_analyzer.filter_relevant_sentences(
                user_content, content
            )

            # 记录筛选结果
            if filtered_user_content != user_content:
                logger.debug(f"用户消息已筛选: 从 {len(user_content)} 字符减少到 {len(filtered_user_content)} 字符")
                logger.debug(f"原始用户消息: {user_content}")
                logger.debug(f"筛选后用户消息: {filtered_user_content}")

            user_content = filtered_user_content

        # 检查消息相关性
        if not self.semantic_analyzer.check_pair_relevance(user_content, content):
            logger.debug("用户消息与助理回复相关性不足，跳过")
            return False

        # 添加消息对
        self.messages.append({
            "role": "user",
            "content": user_content,
            "time": self.last_time
        })
        self.messages.append({
            "role": "assistant",
            "content": content,
            "time": timestamp
        })

        # 重置用户消息缓存
        self.user_msg_buffer = []
        self.last_time = timestamp
        return True

    def build(self):
        """构建符合所有要求的对话"""
        # 必须有完整的对话对
        if len(self.messages) < 2 or len(self.messages) % 2 != 0:
            return None

        # 构建最终格式
        formatted = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
        formatted.extend(
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages
        )

        # 检查整体连贯性
        if not self.semantic_analyzer.is_coherent(formatted):
            logger.debug("对话整体连贯性不足，跳过")
            return None

        return formatted


# ======================
# 主处理流程
# ======================
def process_csv_to_jsonl(input_csv, output_jsonl):
    """完整的处理流程"""
    try:
        # 0. 初始化语义分析器
        semantic_analyzer = SemanticAnalyzer.get_instance()

        # 1. 加载数据
        logger.info(f"加载CSV文件: {input_csv}")
        df = pd.read_csv(safe_path(input_csv))

        # 检查必要列
        required_columns = {'Sender', 'StrContent', 'StrTime'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"CSV文件缺少必要列: {missing}")

        df['CreateTime'] = pd.to_datetime(df['StrTime'])
        df.sort_values('CreateTime', inplace=True)

        # 2. 处理消息
        builder = EnhancedConversationBuilder()
        valid_conversations = []
        message_count = 0

        for _, row in df.iterrows():
            message_count += 1
            if message_count % 1000 == 0:
                logger.info(f"已处理 {message_count} 条消息，找到 {len(valid_conversations)} 个有效对话")

            # 确定角色
            sender = row['Sender']
            role = 'assistant' if sender == Config.TARGET_USER_ID else 'user'
            content = clean_content(row['StrContent'])

            if not content:
                continue

            current_time = row['CreateTime']

            # 检查时间间隔
            if (builder.last_time and
                    (current_time - builder.last_time) > Config.CONTEXT_WINDOW):
                if conv := builder.build():
                    valid_conversations.append(conv)
                builder.reset()

            # 尝试添加消息
            if not builder.add_message(role, content, current_time, sender):
                # 添加失败，可能是对话不连贯，尝试保存当前对话
                if conv := builder.build():
                    valid_conversations.append(conv)
                builder.reset()
                # 重新尝试添加当前消息
                builder.add_message(role, content, current_time, sender)

        # 处理最后一个对话
        if conv := builder.build():
            valid_conversations.append(conv)

        # 3. 写入JSONL - 只保存messages部分
        logger.info(f"写入JSONL文件: {output_jsonl}")
        with open(safe_path(output_jsonl), 'w', encoding='utf-8') as f:
            for conv in valid_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        logger.info(f"处理完成! 生成 {len(valid_conversations)} 个高质量对话")

        return True

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        return False


# ======================
# 执行入口
# ======================
if __name__ == "__main__":
    input_file = r"D:\GitHub\Cloud-Sure-Cyber_MCMocoder\Data\test.csv"
    output_file = r"D:\GitHub\Cloud-Sure-Cyber_MCMocoder\jsonl\high_quality_output_1.6.jsonl"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if process_csv_to_jsonl(input_file, output_file):
        logger.info("处理成功完成!")
        print(f"新对话已保存到: {output_file}")
    else:
        logger.error("处理过程中出现错误")