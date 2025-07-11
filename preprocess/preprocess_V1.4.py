import pandas as pd
import re
import json
import logging
from datetime import datetime, timedelta
import os


# ======================
# 配置参数
# ======================
class Config:
    TARGET_USER_ID = 'wxid_z2d3h3x279gm22'  # 目标用户ID
    SYSTEM_PROMPT = "你正在扮演MCMocoder在群聊中的角色。请用自然、真实的语气回复其他群成员。"

    # 对话处理
    CONTEXT_WINDOW = timedelta(minutes=30)  # 同一对话最大时间间隔
    MIN_TURNS = 1  # 最小对话轮次 (user-assistant对)
    MAX_TURNS = 6  # 最大对话轮次

    # 清洗设置
    MAX_MSG_LENGTH = 100
    MIN_MSG_LENGTH = 2


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
# 消息清洗函数
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
# 对话构建器
# ======================
class StrictConversationBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.messages = []  # 原始消息列表
        self.last_time = None
        self.participants = set()  # 记录参与用户

    def add_message(self, role, content, timestamp, sender):
        """严格添加消息"""
        # 第一条必须是user
        if not self.messages and role != 'user':
            return False

        # 角色必须交替
        if self.messages and self.messages[-1]['role'] == role:
            return False

        # 记录参与者
        if role == 'user':
            self.participants.add(sender)

        self.messages.append({
            "role": role,
            "content": content,
            "time": timestamp,
            "sender": sender
        })
        self.last_time = timestamp
        return True

    def build(self):
        """构建符合所有要求的对话"""
        # 检查对话轮次
        user_msgs = [m for m in self.messages if m['role'] == 'user']
        assistant_msgs = [m for m in self.messages if m['role'] == 'assistant']

        if len(user_msgs) < Config.MIN_TURNS or len(assistant_msgs) < Config.MIN_TURNS:
            return None

        # 确保以assistant结束
        if self.messages[-1]['role'] != 'assistant':
            self.messages = self.messages[:-1]
            if not self.messages:
                return None

        # 合并连续user消息
        merged_messages = []
        current_user_msgs = []

        for msg in self.messages:
            if msg['role'] == 'user':
                current_user_msgs.append(msg['content'])
            elif msg['role'] == 'assistant' and current_user_msgs:
                # 合并所有user消息为一条
                merged_messages.append({
                    "role": "user",
                    "content": "\n".join(current_user_msgs)
                })
                # 添加assistant消息
                merged_messages.append({
                    "role": "assistant",
                    "content": msg['content']
                })
                current_user_msgs = []

        # 构建最终格式
        formatted = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
        formatted.extend(merged_messages)

        # 添加元数据
        metadata = {
            "start_time": self.messages[0]['time'].isoformat(),
            "end_time": self.messages[-1]['time'].isoformat(),
            "turn_count": len(assistant_msgs),
            "participants": list(self.participants)
        }

        return {"messages": formatted, "metadata": metadata}


# ======================
# 主处理流程
# ======================
def process_csv_to_jsonl(input_csv, output_jsonl):
    """完整的处理流程"""
    try:
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
        builder = StrictConversationBuilder()
        valid_conversations = []

        for _, row in df.iterrows():
            # 确定角色
            role = 'assistant' if row['Sender'] == Config.TARGET_USER_ID else 'user'
            content = clean_content(row['StrContent'])
            sender = row['Sender']

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
                if conv := builder.build():
                    valid_conversations.append(conv)
                builder.reset()
                builder.add_message(role, content, current_time, sender)

        # 处理最后一个对话
        if conv := builder.build():
            valid_conversations.append(conv)

        # 3. 写入JSONL
        logger.info(f"写入JSONL文件: {output_jsonl}")
        with open(safe_path(output_jsonl), 'w', encoding='utf-8') as f:
            for conv in valid_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        logger.info(f"处理完成! 生成 {len(valid_conversations)} 个有效对话")
        return True

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        return False


# ======================
# 执行入口
# ======================
if __name__ == "__main__":
    input_file = r"D:\MemoTrace\data\聊天记录\亲友群\merge.csv"
    output_file = r"D:\GitHub\Cloud-Sure-Cyber_MCMocoder\preprocess\output.jsonl"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if process_csv_to_jsonl(input_file, output_file):
        logger.info("处理成功完成!")
        print(f"输出文件已保存到: {output_file}")
    else:
        logger.error("处理过程中出现错误")

