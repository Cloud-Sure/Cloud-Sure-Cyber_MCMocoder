import pandas as pd
import re
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import itertools


# ======================
# 配置参数
# ======================
class Config:
    # 目标用户信息
    TARGET_USER_ID = 'wxid_z2d3h3x279gm22'  # MCMocoder的Sender ID
    TARGET_USER_NAME = 'MCMocoder'

    # 对话处理参数
    CONTEXT_WINDOW = timedelta(minutes=10)  # 10分钟内的消息视为同一对话上下文
    MIN_CONVERSATION_TURNS = 2  # 最小对话轮数 (1对user-assistant)
    MAX_CONVERSATION_TURNS = 6  # 最大对话轮数

    # 上下文关联阈值
    SEMANTIC_SIMILARITY_THRESHOLD = 0.6  # 语义相似度阈值

    # 输出设置
    OUTPUT_FILE = 'mcmocoder_training_data.jsonl'


# ======================
# 日志设置
# ======================
logger = logging.getLogger('conversation_processor')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# ======================
# 语义分析工具
# ======================
class SemanticAnalyzer:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.initialized = True
        except ImportError:
            self.initialized = False
            logger.warning("未安装sentence-transformers，将使用基础语义分析")

    def calculate_similarity(self, text1, text2):
        if not self.initialized or not text1 or not text2:
            return 0.5  # 默认相似度

        # 计算文本嵌入
        embeddings = self.model.encode([text1, text2])

        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def is_context_related(self, prev_content, current_content):
        """检查两条消息是否上下文相关"""
        if not prev_content or not current_content:
            return False

        # 简单规则：检查关键词重叠
        common_words = set(prev_content.split()) & set(current_content.split())
        if len(common_words) >= 2:
            return True

        # 使用语义模型检查
        if self.initialized:
            similarity = self.calculate_similarity(prev_content, current_content)
            return similarity > Config.SEMANTIC_SIMILARITY_THRESHOLD

        return False


# ======================
# 数据处理函数
# ======================
def load_and_prepare_data(file_path):
    """加载并准备群聊数据"""
    try:
        logger.info(f"开始加载数据文件: {file_path}")
        df = pd.read_csv(file_path)

        # 转换时间格式
        df['CreateTime'] = pd.to_datetime(df['StrTime'])
        df.sort_values('CreateTime', inplace=True)

        # 添加用户类型列
        df['user_type'] = df['Sender'].apply(
            lambda x: 'assistant' if x == Config.TARGET_USER_ID else 'user'
        )

        logger.info(f"成功加载数据: {len(df)} 条消息")
        return df
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise


def is_valid_message(content, msg_type, sub_type):
    """检查消息是否有效"""
    if pd.isna(content):
        return False
    if not isinstance(content, str):
        return False
    if msg_type in {47, 49} or sub_type == 57:
        return False
    if content.startswith('<msg>'):
        return False
    return True


def clean_message_content(content):
    """清洗消息内容"""
    content = re.sub(r'@\S+\s?', '', content)  # 移除@提及
    content = re.sub(r'http[s]?://\S+', '', content)  # 移除URL
    content = re.sub(r'\[.*?\]', '', content)  # 移除方括号内容
    content = content.replace('（）', '').replace('()', '')  # 移除括号
    content = re.sub(r'\s+', ' ', content).strip()  # 移除多余空格
    return content


# ======================
# 智能对话重建算法
# ======================
def build_contextual_conversations(df, semantic_analyzer):
    """
    构建上下文连贯的对话
    返回: [{
        "messages": [
            {"time": datetime, "sender": str, "role": str, "content": str},
            ...
        ]
    }]
    """
    logger.info("开始构建上下文连贯的对话...")

    # 预处理有效消息
    valid_messages = []
    for _, row in df.iterrows():
        if is_valid_message(row['StrContent'], row['Type'], row['SubType']):
            content = clean_message_content(row['StrContent'])
            if content:
                valid_messages.append({
                    "time": row['CreateTime'],
                    "sender": row['Sender'],
                    "role": row['user_type'],
                    "content": content
                })

    if not valid_messages:
        logger.warning("没有有效的消息可供处理")
        return []

    logger.info(f"有效消息数量: {len(valid_messages)}")

    # 按时间排序
    valid_messages.sort(key=lambda x: x['time'])

    # 构建对话块
    conversations = []
    current_conv = []
    last_assistant_time = None

    for i, msg in enumerate(valid_messages):
        # 第一条消息
        if not current_conv:
            current_conv.append(msg)
            if msg['role'] == 'assistant':
                last_assistant_time = msg['time']
            continue

        prev_msg = current_conv[-1]
        time_diff = msg['time'] - prev_msg['time']

        # 检查是否开始新对话
        start_new_conversation = False

        # 规则1: 超过时间窗口
        if time_diff > Config.CONTEXT_WINDOW:
            start_new_conversation = True

        # 规则2: 目标用户回复后长时间没有回应
        elif last_assistant_time and (msg['time'] - last_assistant_time) > timedelta(minutes=5):
            start_new_conversation = True

        # 规则3: 连续两条assistant消息
        elif msg['role'] == 'assistant' and prev_msg['role'] == 'assistant':
            start_new_conversation = True

        # 规则4: 上下文不相关
        elif not semantic_analyzer.is_context_related(prev_msg['content'], msg['content']):
            start_new_conversation = True

        # 开始新对话
        if start_new_conversation:
            # 保存当前对话（如果有效）
            if is_valid_conversation(current_conv):
                conversations.append({
                    "messages": current_conv.copy()
                })

            # 开始新对话
            current_conv = [msg]
            last_assistant_time = msg['time'] if msg['role'] == 'assistant' else None
        else:
            # 添加到当前对话
            current_conv.append(msg)
            if msg['role'] == 'assistant':
                last_assistant_time = msg['time']

    # 添加最后一个对话
    if current_conv and is_valid_conversation(current_conv):
        conversations.append({
            "messages": current_conv
        })

    logger.info(f"构建完成! 共 {len(conversations)} 个对话")
    return conversations


def is_valid_conversation(conversation):
    """检查对话是否有效"""
    messages = conversation["messages"]
    if len(messages) < Config.MIN_CONVERSATION_TURNS * 2:
        return False

    # 检查是否包含目标用户
    has_assistant = any(msg['role'] == 'assistant' for msg in messages)
    if not has_assistant:
        return False

    # 检查是否有至少一对user-assistant交互
    user_assistant_pairs = 0
    for i in range(1, len(messages)):
        if messages[i - 1]['role'] == 'user' and messages[i]['role'] == 'assistant':
            user_assistant_pairs += 1

    return user_assistant_pairs >= Config.MIN_CONVERSATION_TURNS


# ======================
# 对话精炼与格式转换
# ======================
def refine_conversation(conversation):
    """
    精炼对话结构以满足格式要求:
    1. 添加system消息
    2. 确保user和assistant交替
    3. 确保以assistant结束
    """
    messages = conversation['messages']

    # 1. 创建上下文相关的system提示
    # 分析对话主题
    topics = detect_conversation_topics(messages)
    system_prompt = f"你正在扮演{Config.TARGET_USER_NAME}在群聊中的角色。当前对话主题：{', '.join(topics[:3])}。请用自然、真实的语气回复其他群成员。"

    system_msg = {
        "role": "system",
        "content": system_prompt
    }

    # 2. 重组消息序列
    refined_messages = [system_msg]

    # 收集连续的用户消息
    current_user_msgs = []

    for msg in messages:
        if msg['role'] == 'user':
            # 收集连续的用户消息
            current_user_msgs.append(msg['content'])
        elif msg['role'] == 'assistant' and current_user_msgs:
            # 合并连续的用户消息
            user_content = "\n".join(current_user_msgs)
            refined_messages.append({
                "role": "user",
                "content": user_content
            })
            refined_messages.append({
                "role": "assistant",
                "content": msg['content']
            })
            current_user_msgs = []

    # 3. 确保以assistant结束
    if refined_messages[-1]['role'] != 'assistant':
        # 寻找最后一个有效的assistant回复
        for i in range(len(refined_messages) - 1, -1, -1):
            if refined_messages[i]['role'] == 'assistant':
                refined_messages = refined_messages[:i + 1]
                break
        else:
            return None  # 没有有效的assistant消息

    # 4. 确保有足够的对话轮次
    assistant_count = sum(1 for msg in refined_messages if msg['role'] == 'assistant')
    if assistant_count < Config.MIN_CONVERSATION_TURNS:
        return None

    # 5. 确保对话连贯性
    if not is_conversation_coherent(refined_messages):
        return None

    return {
        "messages": refined_messages
    }


def detect_conversation_topics(messages):
    """检测对话主题"""
    # 提取所有消息内容
    contents = [msg['content'] for msg in messages if msg['content']]

    # 简单关键词提取
    keywords = []
    for content in contents:
        words = content.split()
        if len(words) > 3:  # 忽略过短的消息
            # 提取名词和动词
            keywords.extend([word for word in words if len(word) > 1 and not word.isdigit()])

    # 统计高频词
    from collections import Counter
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(5)]


def is_conversation_coherent(messages):
    """检查对话是否连贯"""
    # 跳过system消息
    conversation_content = " ".join(msg['content'] for msg in messages[1:])

    # 简单规则：至少有两个不同的主题词
    words = set(conversation_content.split())
    return len(words) >= 5  # 至少5个不同的词


# ======================
# 主处理流程
# ======================
def process_to_jsonl(input_file, output_file):
    """完整的处理流程"""
    try:
        # 初始化语义分析器
        semantic_analyzer = SemanticAnalyzer()

        # 1. 加载数据
        df = load_and_prepare_data(input_file)

        # 2. 重建对话
        conversations = build_contextual_conversations(df, semantic_analyzer)

        if not conversations:
            logger.error("没有重建出任何对话，请检查数据格式")
            return 0

        logger.info(f"初步重建对话: {len(conversations)} 个")

        # 3. 精炼对话并保存
        valid_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                refined = refine_conversation(conv)
                if refined:
                    # 检查对话轮次
                    turns = sum(1 for msg in refined['messages'] if msg['role'] == 'assistant')
                    if turns >= Config.MIN_CONVERSATION_TURNS:
                        f.write(json.dumps(refined, ensure_ascii=False) + '\n')
                        valid_count += 1

                        # 记录前3个示例
                        if valid_count <= 3:
                            logger.info(f"示例对话 {valid_count} (轮次: {turns}):")
                            for msg in refined['messages']:
                                role = msg['role']
                                content = msg['content'][:50] + ('...' if len(msg['content']) > 50 else '')
                                logger.info(f"  {role}: {content}")

        logger.info(f"处理完成! 生成 {valid_count}/{len(conversations)} 个有效对话")
        return valid_count
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return 0


# ======================
# 执行入口
# ======================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        input_file = "D:\MemoTrace\data\聊天记录\亲友群\merge.csv"
    else:
        input_file = sys.argv[1]


    output_file = Config.OUTPUT_FILE

    logger.info(f"开始处理文件: {input_file}")
    result = process_to_jsonl(input_file, output_file)

    if result > 0:
        logger.info(f"成功生成 {result} 个对话，已保存到 {output_file}")
        sys.exit(0)
    else:
        logger.error("未生成有效对话")
        sys.exit(1)