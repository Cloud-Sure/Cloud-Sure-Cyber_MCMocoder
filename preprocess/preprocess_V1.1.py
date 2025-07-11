import pandas as pd
import re
import json
import numpy as np
from collections import Counter
import jieba
import jieba.analyse
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


# ======================
# 配置参数
# ======================
class Config:
    TARGET_USER_SENDER = 'wxid_z2d3h3x279gm22'  # MCMocoder的Sender ID
    OUTPUT_FILE = 'mcmocoder_training_data.jsonl'
    MIN_MESSAGE_LENGTH = 1
    CONTEXT_TIME_GAP = 300  # 5分钟内的消息视为同一上下文
    TOP_KEYWORDS_COUNT = 20
    SENTIMENT_WORD_COUNT = 15
    STOP_WORDS = {'的', '了', '是', '在', '和', '有', '我', '你', '他', '她', '它'}
    # file_path = 'D:\MemoTrace\data\聊天记录\亲友群\merge.csv'

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super().default(obj)

# ======================
# 工具函数
# ======================
def setup_logging():
    """配置日志记录"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


logger = setup_logging()


def timer(func):
    """计时装饰器"""
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} 耗时: {end - start:.2f}秒")
        return result

    return wrapper


# ======================
# 数据加载与清洗
# ======================
@timer
def load_and_filter_data(file_path: str) -> pd.DataFrame:
    """加载并过滤目标用户数据"""
    try:
        df = pd.read_csv(file_path)

        # 基本数据校验
        required_columns = {'Sender', 'StrContent', 'Type', 'SubType', 'StrTime'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV文件缺少必要列，需要: {required_columns}")

        # 过滤目标用户
        target_df = df[df['Sender'] == Config.TARGET_USER_SENDER].copy()
        if len(target_df) == 0:
            raise ValueError(f"未找到Sender为 {Config.TARGET_USER_SENDER} 的消息")

        # 时间处理
        target_df['CreateTime'] = pd.to_datetime(target_df['StrTime'])
        target_df.sort_values('CreateTime', inplace=True)

        logger.info(f"成功加载 {len(target_df)} 条目标用户消息")
        return target_df

    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise


@timer
def clean_message_content(row: pd.Series) -> str:
    """清洗单条消息内容"""
    content = row['StrContent']
    if pd.isna(content):
        return ""

    # 处理非文本消息
    if not isinstance(content, str):
        return ""

    # 处理特殊消息类型
    if row['Type'] in {47, 49} or row['SubType'] == 57:
        return "[表情]"

    # 处理XML格式消息
    if content.startswith('<msg>'):
        if '<emoji' in content:
            return "[表情]"
        if '<location' in content:
            return "[位置]"
        return "[文件]"

    # 通用文本清洗流程
    content = re.sub(r'@\S+', '', content)  # 移除@提及
    content = re.sub(r'http[s]?://\S+', '', content)  # 移除URL
    content = re.sub(r'\[.*?\]', '', content)  # 移除方括号内容
    content = content.replace('（）', '').replace('()', '')  # 移除括号

    # 标准化标点
    content = re.sub(r'[~～]+', '～', content)  # 统一波浪号
    content = re.sub(r'[！!]+', '！', content)  # 统一感叹号
    content = re.sub(r'[…]+', '…', content)  # 统一省略号

    # 移除多余空格
    return re.sub(r'\s+', ' ', content).strip()


# ======================
# 高级情感分析
# ======================
@timer
def extract_sentiment_words(texts: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    从文本中提取情感词汇
    返回: (平均情感分数, 积极词汇列表, 消极词汇列表)
    """
    texts = [t for t in texts if t and t not in {"[表情]", "[位置]", "[文件]"}]
    if not texts:
        return 0.5, [], []

    # 情感分析
    sentiments = []
    word_sentiments = Counter()

    for text in texts:
        try:
            s = SnowNLP(text)
            sentiments.append(s.sentiments)

            # 提取带权重的词汇情感
            for word in set(s.words):
                if len(word) > 1 and word not in Config.STOP_WORDS:
                    indices = [i for i, w in enumerate(s.words) if w == word]
                    word_score = sum(s.sentiments[i] for i in indices) / len(indices)
                    word_sentiments[word] += word_score * len(indices)  # 按出现次数加权
        except:
            continue

    # 计算整体情感倾向
    avg_sentiment = np.mean(sentiments) if sentiments else 0.5

    # 提取情感词汇 (至少出现3次)
    qualified_words = {w for w in word_sentiments if word_sentiments[w] >= 3}
    pos_words = [w for w in qualified_words if word_sentiments[w] / word_sentiments.total() > 0.7]
    neg_words = [w for w in qualified_words if word_sentiments[w] / word_sentiments.total() < 0.3]

    # 按情感强度排序
    pos_words.sort(key=lambda w: word_sentiments[w], reverse=True)
    neg_words.sort(key=lambda w: word_sentiments[w])

    return avg_sentiment, pos_words[:Config.SENTIMENT_WORD_COUNT], neg_words[:Config.SENTIMENT_WORD_COUNT]


# ======================
# 特征工程
# ======================
@timer
def extract_linguistic_features(text: str,
                                positive_words: List[str],
                                negative_words: List[str]) -> Dict:
    """提取文本的深层语言特征"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[。！？!?]', text)),
        'punctuation': {},
        'emotion': {
            'score': 0.5,
            'label': 'neutral',
            'positive_words': [],
            'negative_words': []
        },
        'common_phrases': []
    }

    if not text or text in {"[表情]", "[位置]", "[文件]"}:
        return features

    # 标点分析
    punct_counts = Counter(re.findall(r'[～！？…]+', text))
    features['punctuation'] = dict(punct_counts)

    # 情感分析
    try:
        s = SnowNLP(text)
        features['emotion']['score'] = s.sentiments

        # 检测情感词汇
        detected_pos = [w for w in positive_words if w in text]
        detected_neg = [w for w in negative_words if w in text]

        features['emotion'].update({
            'positive_words': detected_pos,
            'negative_words': detected_neg,
            'label': 'positive' if len(detected_pos) > len(detected_neg) else
            'negative' if len(detected_neg) > len(detected_pos) else
            'neutral'
        })
    except:
        pass

    # 特殊短语检测
    if "我有错" in text:
        features['common_phrases'].append("我有错")
    if "～" in text and len(text) < 10:
        features['common_phrases'].append("短波浪号消息")

    return features


@timer
def extract_keyphrases(texts: List[str]) -> List[str]:
    """使用TF-IDF和TextRank提取关键词"""
    texts = [t for t in texts if t and t not in {"[表情]", "[位置]", "[文件]"}]
    if not texts:
        return []

    # TF-IDF关键词
    vectorizer = TfidfVectorizer(
        tokenizer=jieba.cut,
        stop_words=list(Config.STOP_WORDS))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 合并文本进行TextRank分析
    combined_text = ' '.join(texts)
    textrank_words = jieba.analyse.textrank(
        combined_text,
        topK=Config.TOP_KEYWORDS_COUNT,
        withWeight=False)

    # 结合两种方法的结果
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    top_tfidf_indices = tfidf_scores.argsort()[-Config.TOP_KEYWORDS_COUNT:][::-1]
    tfidf_words = [feature_names[i] for i in top_tfidf_indices]

    # 合并并去重
    keywords = list(set(tfidf_words + textrank_words))
    return keywords[:Config.TOP_KEYWORDS_COUNT]


# ======================
# 上下文处理
# ======================
@timer
def build_conversation_blocks(df: pd.DataFrame) -> List[Dict]:
    """构建带上下文的对话块"""
    blocks = []
    current_block = []
    last_time = None

    for _, row in df.iterrows():
        content = row['clean_content']
        if not content:
            continue

        current_time = row['CreateTime']

        # 初始化或检测时间间隔
        if last_time is None:
            time_gap = 0
        else:
            time_gap = (current_time - last_time).total_seconds()

        # 新对话块的判断条件
        if time_gap > Config.CONTEXT_TIME_GAP and current_block:
            blocks.append({
                'messages': current_block.copy(),
                'start_time': current_block[0]['time'],
                'end_time': current_block[-1]['time']
            })
            current_block = []

        # 添加当前消息
        current_block.append({
            'text': content,
            'time': current_time,
            'type': row['Type'],
            'subtype': row['SubType']
        })
        last_time = current_time

    # 添加最后一个块
    if current_block:
        blocks.append({
            'messages': current_block,
            'start_time': current_block[0]['time'],
            'end_time': current_block[-1]['time']
        })

    logger.info(f"构建了 {len(blocks)} 个对话块，平均每块 {np.mean([len(b['messages']) for b in blocks]):.1f} 条消息")
    return blocks


# ======================
# 主处理流程
# ======================
@timer
def process_chat_data(input_path: str) -> None:
    """完整的预处理流程"""
    try:
        # 1. 数据加载
        df = load_and_filter_data(input_path)

        # 2. 内容清洗
        df['clean_content'] = df.apply(clean_message_content, axis=1)
        df = df[df['clean_content'].str.len() >= Config.MIN_MESSAGE_LENGTH]

        # 3. 情感分析
        valid_texts = [t for t in df['clean_content'] if t not in {"[表情]", "[位置]", "[文件]"}]
        avg_sentiment, pos_words, neg_words = extract_sentiment_words(valid_texts)
        logger.info(f"情感分析结果 - 平均分: {avg_sentiment:.2f}")
        logger.info(f"积极词汇 ({len(pos_words)}): {', '.join(pos_words[:5])}{'...' if len(pos_words) > 5 else ''}")
        logger.info(f"消极词汇 ({len(neg_words)}): {', '.join(neg_words[:5])}{'...' if len(neg_words) > 5 else ''}")

        # 4. 关键词提取
        keywords = extract_keyphrases(valid_texts)
        logger.info(f"提取关键词 ({len(keywords)}): {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")

        # 5. 上下文构建
        conversation_blocks = build_conversation_blocks(df)

        # 6. 生成训练数据
        training_data = []
        for block in conversation_blocks:
            block_texts = [msg['text'] for msg in block['messages']]
            context_text = ' '.join(block_texts)

            # 提取特征
            features = extract_linguistic_features(context_text, pos_words, neg_words)

            training_data.append({
                'text': context_text,
                'messages': [{
                    **msg,
                    'time': msg['time'].isoformat()  # 转换时间格式
                } for msg in block['messages']],
                'metadata': {
                    'time_span': {
                        'start': block['start_time'].isoformat(),  # 转换为ISO格式字符串
                        'end': block['end_time'].isoformat()
                    },
                    'message_count': len(block['messages']),
                    **features
                },
                'user_profile': {
                    'sender_id': Config.TARGET_USER_SENDER,
                    'keywords': keywords,
                    'sentiment_words': {
                        'positive': pos_words,
                        'negative': neg_words,
                        'average_score': avg_sentiment
                    }
                }
            })

        # 7. 保存结果
        with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False, cls=DateTimeEncoder) + '\n')

        # 保存情感词汇
        with open('sentiment_analysis_result.json', 'w', encoding='utf-8') as f:
            json.dump({
                'positive_words': pos_words,
                'negative_words': neg_words,
                'average_sentiment': avg_sentiment,
                'keywords': keywords
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"预处理完成！已保存 {len(training_data)} 条训练数据到 {Config.OUTPUT_FILE}")

        # 8. 生成分析报告
        generate_analysis_report(training_data)

        return training_data

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise


@timer
def generate_analysis_report(data: List[Dict]) -> None:
    """生成数据分析报告"""
    try:
        # 情感分布
        sentiment_scores = [d['metadata']['emotion']['score'] for d in data]

        plt.figure(figsize=(12, 6))

        # 情感分数分布
        plt.subplot(1, 2, 1)
        sns.histplot(sentiment_scores, bins=20, kde=True)
        plt.title('Distribution of emotional scores')
        plt.xlabel('Emotional score (0= negative, 1= positive)')

        # 消息长度分布
        plt.subplot(1, 2, 2)
        lengths = [d['metadata']['length'] for d in data]
        sns.histplot(lengths, bins=20, kde=True)
        plt.title('Message length distribution')
        plt.xlabel('number of character')

        plt.tight_layout()
        plt.savefig('data_analysis.png')
        plt.close()

        # 生成文本报告
        with open('analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== 数据分析报告 ===\n")
            f.write(f"总对话块数: {len(data)}\n")
            f.write(f"平均情感分数: {np.mean(sentiment_scores):.2f}\n")
            f.write(f"平均消息长度: {np.mean(lengths):.1f} 字符\n")
            f.write(f"最短消息: {min(lengths)}, 最长消息: {max(lengths)}\n")

            # 统计标点使用
            punct_counts = Counter()
            for d in data:
                punct_counts.update(d['metadata']['punctuation'])
            f.write("\n标点使用统计:\n")
            for punct, count in punct_counts.most_common():
                f.write(f"  {punct}: {count}次\n")

        logger.info("已生成数据分析报告: analysis_report.txt 和 data_analysis.png")

    except Exception as e:
        logger.warning(f"生成分析报告失败: {str(e)}")


# ======================
# 执行入口
# ======================
if __name__ == "__main__":
    import sys

    # 检查输入文件
    #TODO
    if len(sys.argv) < 2:
        input_file = "D:\MemoTrace\data\聊天记录\亲友群\merge.csv"
    else:
        input_file = sys.argv[1]



    # 检查依赖
    try:
        import snownlp
    except ImportError:
        logger.info("安装依赖库...")
        import subprocess

        subprocess.run(['pip', 'install', 'snownlp', 'jieba', 'scikit-learn', 'pandas', 'matplotlib', 'seaborn'])

    # 运行预处理
    try:
        process_chat_data(input_file)
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        sys.exit(1)