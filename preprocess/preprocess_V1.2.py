import pandas as pd
import re
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict


# ======================
# 配置参数
# ======================
class Config:
    # 目标用户信息
    TARGET_USER_ID = 'wxid_z2d3h3x279gm22'  # MCMocoder的Sender ID
    TARGET_USER_NAME = 'MCMocoder'

    # 对话处理参数
    CONTEXT_WINDOW = timedelta(minutes=5)  # 5分钟内的消息视为同一对话上下文
    MIN_CONVERSATION_TURNS = 1  # 最小对话轮数
    MAX_CONVERSATION_TURNS = 6  # 最大对话轮数

    # 输出设置
    OUTPUT_FILE = 'mcmocoder_training_data.jsonl'


# ======================
# 日志设置
# ======================
def setup_logger():
    logger = logging.getLogger('conversation_processor')
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler('conversation_processing.log')
    fh.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = setup_logger()


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

    # 过滤系统消息和非文本消息
    if not isinstance(content, str):
        return False

    # 过滤特殊消息类型
    if msg_type in {47, 49} or sub_type == 57:
        return False

    # 过滤XML格式消息
    if content.startswith('<msg>'):
        return False

    return True


def clean_message_content(content):
    """清洗消息内容"""
    content = re.sub(r'@\S+', '', content)  # 移除@提及
    content = re.sub(r'http[s]?://\S+', '', content)  # 移除URL
    content = re.sub(r'\[.*?\]', '', content)  # 移除方括号内容
    content = content.replace('（）', '').replace('()', '')  # 移除括号

    # 标准化标点
    content = re.sub(r'[~～]+', '～', content)  # 统一波浪号
    content = re.sub(r'[！!]+', '!', content)  # 统一感叹号

    # 移除多余空格
    return re.sub(r'\s+', ' ', content).strip()


# ======================
# 对话重建核心算法
# ======================
def reconstruct_conversations(df):
    """
    从群聊记录重建对话结构
    返回格式: [{
        "start_time": datetime,
        "end_time": datetime,
        "messages": [{
            "time": datetime,
            "sender": str,
            "role": "user"|"assistant"|"system",
            "content": str
        }]
    }]
    """
    logger.info("开始重建对话结构...")
    conversations = []
    current_conv = []
    last_time = None
    last_role = None

    # 先过滤无效消息
    valid_df = df[df.apply(
        lambda row: is_valid_message(row['StrContent'], row['Type'], row['SubType']),
        axis=1
    )].copy()

    # 清洗内容
    valid_df['clean_content'] = valid_df['StrContent'].apply(clean_message_content)
    valid_df = valid_df[valid_df['clean_content'].str.len() > 0]

    if valid_df.empty:
        logger.warning("没有有效的消息可供处理")
        return []

    logger.info(f"有效消息数量: {len(valid_df)}")

    for _, row in valid_df.iterrows():
        content = row['clean_content']
        current_time = row['CreateTime']
        sender = row['Sender']
        role = row['user_type']

        # 第一条消息
        if last_time is None:
            current_conv.append({
                "time": current_time,
                "sender": sender,
                "role": role,
                "content": content
            })
            last_time = current_time
            last_role = role
            continue

        # 计算时间差
        time_diff = current_time - last_time

        # 检查是否开始新对话
        start_new_conversation = False

        # 规则1: 超过时间窗口
        if time_diff > Config.CONTEXT_WINDOW:
            start_new_conversation = True

        # 规则2: 目标用户回复后长时间没有回应
        elif last_role == 'assistant' and time_diff > timedelta(minutes=1):
            start_new_conversation = True

        # 规则3: 连续两条assistant消息
        elif role == 'assistant' and last_role == 'assistant':
            start_new_conversation = True

        # 开始新对话
        if start_new_conversation:
            # 保存当前对话（如果有效）
            if is_valid_conversation(current_conv):
                conv_obj = {
                    "start_time": current_conv[0]['time'],
                    "end_time": current_conv[-1]['time'],
                    "messages": current_conv
                }
                conversations.append(conv_obj)
                logger.debug(f"保存对话: {len(current_conv)} 条消息")

            # 开始新对话
            current_conv = []

        # 添加消息到当前对话
        current_conv.append({
            "time": current_time,
            "sender": sender,
            "role": role,
            "content": content
        })

        last_time = current_time
        last_role = role

    # 添加最后一个对话
    if current_conv and is_valid_conversation(current_conv):
        conv_obj = {
            "start_time": current_conv[0]['time'],
            "end_time": current_conv[-1]['time'],
            "messages": current_conv
        }
        conversations.append(conv_obj)

    logger.info(f"重建完成! 共 {len(conversations)} 个对话")
    return conversations


def is_valid_conversation(conversation):
    """检查对话是否有效"""
    if len(conversation) < 2:
        return False

    # 检查是否包含目标用户
    has_assistant = any(msg['role'] == 'assistant' for msg in conversation)
    if not has_assistant:
        return False

    # 检查角色顺序
    first_role = conversation[0]['role']
    if first_role == 'assistant':
        return False

    # 检查角色交替
    for i in range(1, len(conversation)):
        prev_role = conversation[i - 1]['role']
        curr_role = conversation[i]['role']

        # 禁止连续相同角色
        if prev_role == curr_role and prev_role != 'system':
            return False

    return True


def refine_conversation(conversation):
    """
    精炼对话结构以满足格式要求:
    1. 添加system消息
    2. 确保user和assistant交替
    3. 确保以assistant结束
    """
    messages = conversation['messages']

    # 1. 添加system消息
    system_msg = {
        "role": "system",
        "content": f"你正在扮演{Config.TARGET_USER_NAME}在群聊中的角色。请用自然、真实的语气回复其他群成员。"
    }

    # 2. 确保第一条非system消息是user
    if messages[0]['role'] != 'user':
        # 寻找第一个user消息
        first_user_index = next((i for i, msg in enumerate(messages) if msg['role'] == 'user'), None)
        if first_user_index is not None:
            messages = messages[first_user_index:]
        else:
            return None  # 没有user消息，无效对话

    # 3. 确保角色交替
    refined_messages = [system_msg]
    last_role = 'system'

    for msg in messages:
        current_role = msg['role']

        # 跳过连续相同角色的消息
        if current_role == last_role and current_role != 'system':
            continue

        # 添加消息
        refined_messages.append({
            "role": current_role,
            "content": msg['content']
        })
        last_role = current_role

    # 4. 确保以assistant结束
    if refined_messages[-1]['role'] != 'assistant':
        # 寻找最后一个assistant消息
        last_assistant_index = next(
            (i for i in range(len(refined_messages) - 1, -1, -1)
             if refined_messages[i]['role'] == 'assistant'), None
        )

        if last_assistant_index is not None:
            refined_messages = refined_messages[:last_assistant_index + 1]
        else:
            return None  # 没有assistant消息，无效对话

    # 5. 确保对话轮数符合要求
    user_messages = [msg for msg in refined_messages if msg['role'] == 'user']
    assistant_messages = [msg for msg in refined_messages if msg['role'] == 'assistant']

    if len(user_messages) < Config.MIN_CONVERSATION_TURNS or len(assistant_messages) < Config.MIN_CONVERSATION_TURNS:
        return None

    # 6. 确保user和assistant成对出现
    # 从索引1开始（跳过system）
    final_messages = []
    for i in range(1, len(refined_messages)):
        current_msg = refined_messages[i]
        prev_msg = refined_messages[i - 1]

        # 如果当前消息与前一条消息角色相同，尝试修复
        if current_msg['role'] == prev_msg['role']:
            # 跳过重复的user消息
            if current_msg['role'] == 'user':
                continue
            # 对于assistant消息，插入一个占位user消息
            elif current_msg['role'] == 'assistant':
                final_messages.append({
                    "role": "user",
                    "content": "请继续"
                })

        final_messages.append(current_msg)

    # 如果没有修复，使用原始消息
    if not final_messages:
        final_messages = refined_messages[1:]

    # 重建完整消息列表
    final_refined = [system_msg] + final_messages

    # 再次确保以assistant结束
    if final_refined[-1]['role'] != 'assistant':
        return None

    return {
        "messages": final_refined
    }


# ======================
# 主处理流程
# ======================
def process_to_jsonl(input_file, output_file):
    """完整的处理流程"""
    try:
        # 1. 加载数据
        df = load_and_prepare_data(input_file)

        # 2. 重建对话
        conversations = reconstruct_conversations(df)

        if not conversations:
            logger.error("没有重建出任何对话，请检查数据格式")
            return 0

        # 3. 精炼对话并保存
        valid_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, conv in enumerate(conversations):
                refined = refine_conversation(conv)
                if refined:
                    f.write(json.dumps(refined, ensure_ascii=False) + '\n')
                    valid_count += 1

                    # 记录前3个示例
                    if valid_count <= 3:
                        logger.info(f"示例对话 {valid_count}:")
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