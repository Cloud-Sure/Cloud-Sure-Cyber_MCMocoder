import pandas as pd
import re
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict


# ======================
# 配置参数
# ======================
class Config:
    TARGET_USER_ID = 'wxid_z2d3h3x279gm22'  # MCMocoder的Sender ID
    TARGET_USER_NAME = 'MCMocoder'

    # 对话处理
    CONTEXT_WINDOW = timedelta(minutes=15)
    MIN_CONVERSATION_TURNS = 2
    MAX_CONVERSATION_TURNS = 6

    # 清洗设置
    MAX_MESSAGE_LENGTH = 500
    MIN_MESSAGE_LENGTH = 2

    # 输出
    OUTPUT_FILE = 'complete_training_data.jsonl'


# ======================
# 日志设置
# ======================
logger = logging.getLogger('conversation_processor')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('conversation_processing.log', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)


# ======================
# 强化消息清洗函数
# ======================
def clean_message_content(content):
    """深度清洗消息内容"""
    if not content or not isinstance(content, str):
        return None

    # 1. 完全过滤XML消息
    if content.strip().startswith('<?xml') or content.strip().startswith('<msg>'):
        return "[图片/表情]"

    # 2. 移除XML标签和属性
    content = re.sub(r'<[^>]+>', '', content)

    # 3. 移除技术性内容
    content = re.sub(r'\b[a-f0-9]{32}\b', '', content)  # MD5
    content = re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', '', content)  # UUID

    # 4. 移除特殊符号和表情代码
    content = re.sub(r'&\w+;', '', content)  # HTML实体

    # 5. 标准化文本
    content = re.sub(r'@\S+\s?', '', content)  # 移除@提及
    content = re.sub(r'http[s]?://\S+', '', content)  # 移除URL
    content = re.sub(r'\[.*?\]', '', content)  # 移除方括号内容
    content = re.sub(r'\s+', ' ', content).strip()  # 标准化空格

    # 6. 过滤无效内容
    if len(content) < Config.MIN_MESSAGE_LENGTH:
        return None
    if len(content) > Config.MAX_MESSAGE_LENGTH:
        return content[:Config.MAX_MESSAGE_LENGTH] + "..."

    return content if content else None


def is_valid_message(row):
    """强化消息有效性检查"""
    content = row['StrContent']
    if pd.isna(content) or not isinstance(content, str):
        return False

    # 过滤特殊消息类型
    if row['Type'] in {47, 49} or row['SubType'] == 57:
        return False

    # 过滤空消息和过短消息
    if len(content.strip()) < Config.MIN_MESSAGE_LENGTH:
        return False

    return True


# ======================
# 对话重建与整合
# ======================
def build_complete_conversations(df):
    """构建包含所有群友发言的完整对话"""
    logger.info("开始构建完整对话结构...")

    # 预处理有效消息
    valid_messages = []
    for _, row in df.iterrows():
        if is_valid_message(row):
            content = clean_message_content(row['StrContent'])
            if content:
                # 确定角色：目标用户为assistant，其他为user
                role = 'assistant' if row['Sender'] == Config.TARGET_USER_ID else 'user'

                # 添加昵称前缀以区分不同用户
                nickname = row.get('NickName', '未知用户') or row.get('Remark', '未知用户') or '未知用户'
                prefixed_content = f"{nickname}: {content}" if role == 'user' else content

                valid_messages.append({
                    "time": row['CreateTime'],
                    "sender": row['Sender'],
                    "nickname": nickname,
                    "role": role,
                    "content": prefixed_content
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
    last_time = None

    for msg in valid_messages:
        # 第一条消息
        if not current_conv:
            current_conv.append(msg)
            last_time = msg['time']
            continue

        time_diff = msg['time'] - last_time

        # 检查是否开始新对话
        if time_diff > Config.CONTEXT_WINDOW:
            if current_conv:
                # 确保对话以user开始，以assistant结束
                if current_conv[0]['role'] == 'user' and any(m['role'] == 'assistant' for m in current_conv):
                    conversations.append({
                        "messages": current_conv.copy(),
                        "start_time": current_conv[0]['time'],
                        "end_time": current_conv[-1]['time']
                    })
            current_conv = [msg]
        else:
            current_conv.append(msg)

        last_time = msg['time']

    # 添加最后一个对话
    if current_conv and current_conv[0]['role'] == 'user' and any(m['role'] == 'assistant' for m in current_conv):
        conversations.append({
            "messages": current_conv,
            "start_time": current_conv[0]['time'],
            "end_time": current_conv[-1]['time']
        })

    logger.info(f"构建完成! 共 {len(conversations)} 个对话")
    return conversations


# ======================
# 对话精炼与格式转换
# ======================
def refine_conversation(conversation):
    """精炼对话结构以满足训练格式要求"""
    messages = conversation['messages']

    # 1. 创建系统提示
    system_prompt = f"你正在扮演{Config.TARGET_USER_NAME}在群聊中的角色。请用自然、真实的语气回复其他群成员。"
    refined_messages = [{"role": "system", "content": system_prompt}]

    # 2. 重组消息序列
    current_user_content = []

    for msg in messages:
        if msg['role'] == 'user':
            # 收集所有user消息
            current_user_content.append(msg['content'])
        elif msg['role'] == 'assistant' and current_user_content:
            # 合并所有user消息为一条
            user_content = "\n".join(current_user_content)
            refined_messages.append({"role": "user", "content": user_content})

            # 添加assistant回复
            refined_messages.append({"role": "assistant", "content": msg['content']})

            current_user_content = []

    # 3. 确保至少有一对有效对话
    if len(refined_messages) < 3:  # system + user + assistant
        return None

    # 4. 确保对话质量
    if not is_high_quality_conversation(refined_messages):
        return None

    # 5. 构建metadata
    assistant_count = sum(1 for msg in refined_messages if msg['role'] == 'assistant')

    # 提取对话中的用户昵称
    participants = set()
    for msg in conversation['messages']:
        if msg['role'] == 'user':
            participants.add(msg['nickname'])

    return {
        "messages": refined_messages,
        "metadata": {
            "original_start": conversation['start_time'].isoformat(),
            "original_end": conversation['end_time'].isoformat(),
            "turn_count": assistant_count,
            "participants": list(participants)
        }
    }


def is_high_quality_conversation(messages):
    """检查对话质量"""
    # 1. 跳过system消息
    content_messages = [msg for msg in messages if msg['role'] in ('user', 'assistant')]

    # 2. 检查有效消息数量
    if len(content_messages) < 2:  # 至少一对user-assistant
        return False

    # 3. 检查媒体消息比例
    media_count = sum(1 for msg in content_messages if msg['content'] == "[图片/表情]")
    if media_count / len(content_messages) > 0.3:  # 超过30%媒体消息
        return False

    # 4. 内容多样性检查
    unique_words = set()
    for msg in content_messages:
        if msg['content'] != "[图片/表情]":
            words = re.findall(r'[\w\u4e00-\u9fff]{2,}', msg['content'])
            unique_words.update(words)

    if len(unique_words) < 10:  # 至少10个不同的词
        return False

    return True


# ======================
# 主处理流程
# ======================
def process_to_jsonl(input_file, output_file):
    try:
        logger.info(f"开始处理文件: {input_file}")

        # 1. 加载数据
        logger.info("加载数据...")
        df = pd.read_csv(input_file)
        df['CreateTime'] = pd.to_datetime(df['StrTime'])
        df.sort_values('CreateTime', inplace=True)
        logger.info(f"加载完成: {len(df)} 条消息")

        # 2. 构建完整对话
        logger.info("构建对话结构...")
        conversations = build_complete_conversations(df)

        if not conversations:
            logger.error("没有构建出任何对话")
            return 0

        logger.info(f"构建完成: {len(conversations)} 个对话")

        # 3. 精炼对话并保存
        logger.info("精炼对话...")
        high_quality_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                refined = refine_conversation(conv)
                if refined:
                    json_line = json.dumps(refined, ensure_ascii=False)
                    f.write(json_line + '\n')
                    high_quality_count += 1

                    # 记录前3个示例
                    if high_quality_count <= 3:
                        logger.info(f"高质量对话示例 {high_quality_count}:")
                        for msg in refined['messages']:
                            content = msg['content']
                            if len(content) > 100:
                                content = content[:100] + "..."
                            logger.info(f"  {msg['role']}: {content}")

        logger.info(f"处理完成! 生成 {high_quality_count} 个高质量对话")
        return high_quality_count

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        return 0


# ======================
# 执行入口
# ======================
if __name__ == "__main__":
    import sys
    import time

    start_time = time.time()

    if len(sys.argv) < 2:
        input_file = "D:\MemoTrace\data\聊天记录\亲友群\merge.csv"
    else:
        input_file = sys.argv[1]
    output_file = Config.OUTPUT_FILE

    try:
        result = process_to_jsonl(input_file, output_file)

        if result > 0:
            logger.info(f"成功生成 {result} 个高质量对话，已保存到 {output_file}")
            logger.info(f"总耗时: {time.time() - start_time:.2f}秒")
            sys.exit(0)
        else:
            logger.error("未生成有效对话")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.error("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}", exc_info=True)
        sys.exit(1)