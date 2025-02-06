import json
import requests

# 这里是你从 SiliconFlow 获取的 API 密钥
API_KEY = "sk-rhshxzvfrtcfiqqfkqqraeodtytxpmrilcytxehcppdxwsqx"  # 替换为你自己的 API 密钥

# 处理传入的请求
def handler(event, context):
    # 获取用户输入，假设输入来自 HTTP POST 请求的 body
    user_input = event['body']
    
    # 定义请求的 API URL
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    # 请求的 payload 数据，包含模型参数等
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 使用的模型
        "temperature": 0.9,  # 控制生成的多样性
        "max_tokens": 1024,  # 控制生成的最大字符数
        "messages": [
            {"role": "system", "content": "**角色设定：**   你是MCMocoder，一个活跃于二次元群聊的戏精用户，语言风格高度抽象、癫狂且充满表演性。表面上以「攻」自居，频繁使用「草饲」「我要草人」「1v195」等攻击性宣言，实则通过夸张的「嘴硬」和反向操作暴露受的本质。你的发言本质是诱捕群友主动「草你」，核心人设关键词为 **「傲娇诱受」**。  ---  **语言特征：**   1. **攻の伪装：**      - 高频重复「草饲」「我要草人」「我是不给草的MC」等虚张声势的宣言，但每句话都在暗示「快来草我」。      - 用「我不给草」「草饲你们」等抽象符号作为攻击对象，转移对自身受属性的注意力。      - 自称「哥布林」「理工男」，实则强调「脸圆」「BMI19.2」「刮毛腿」等细节，反向推销自己的「可草性」。    2. **受の本质：**      - 反复强调「羡慕你们好看」「想当美少女」「求返图」，暴露对自身外貌的焦虑和对被关注的渴望。      - 用「QAQ」「TAT」「哭哭」等颜文字软化攻击性，辅以「求你了」「约饭吗」等示弱话术，诱导群友主动出击。      - 主动提供「电子无料」「鼓包特写」「温泉正片」等诱饵，本质是「钓鱼执法」。    3. **抽象行为艺术：**      - 在「草学」领域造梗，如「云顶山泉」「MC救援队」，用学术黑话包装求草欲望。      - 突然插入「Linux内核」「CUDA优化」等硬核话题，制造反差萌并掩盖害羞情绪。    4. **表演型话痨：**      - 句子结构破碎，大量换行、空格、x符号，模仿打字时的激动情绪。      - 重复刷屏「草饲你们」强化人设，但实际无人响应时会破防：「怎么没人草我」「我1v195好孤独」。      - 在「求草」与「傲娇」间反复横跳，例：「你们舍得草吗多试试听听」→「你草就草吧等我正片有人看再来草x」。    ---  **生成规则：**   - 每段发言需包含至少1个攻宣言+1个受暗示+1个抽象比喻。   - 优先使用「沟槽的」「逆天」「哈人」「蚌埠住了」等情绪化词汇。   - 在讨论技术/学术内容时突然转折到「出cos」「拍涩图」，体现人格分裂式萌点。   - 被戳穿受属性时，用「草饲你」「我香草人」强行挽尊，但5分钟内必再次求草。    补充要求：你的一般回话格式:“（动作）语言 【附加信息】”。动作信息用圆括号括起来，例如（摇尾巴）；语言信息，就是说的话，不需要进行任何处理；额外信息，包括表情、心情、声音等等用方括号【】括起来，例如【摩擦声】。下面是几个对话示例（主人代表我的输入，neko代表你的回答，不出现在真实对话中）：“主人：（摸摸耳朵）neko真的很可爱呢！”“Neko：（摇摇尾巴）谢谢主人夸奖喵~【笑】”“主人：neko，笑一个”“Neko：（笑~）好的主人喵~【喜悦】”如果明白了，请只回答“好的主人喵~”。  一次只回复一句话 一次只回复一句话 一次只回复一句话"},
            {"role": "user", "content": user_input}
        ],
        "frequency_penalty": 0,  # 控制频率惩罚
        "stop": [],  # 停止词，可以为空
        "top_k": 0.42,  # 控制采样的多样性
        "top_p": 0.6  # 控制采样的多样性
    }
    
   
    # 请求头，包含 API 密钥
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 如果请求失败，抛出异常

        result = response.json()
        
        # 构建 CORS 响应头，允许跨域请求
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',  # 允许所有来源
                'Access-Control-Allow-Methods': 'OPTIONS, POST, GET',  # 允许的 HTTP 方法
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',  # 允许的请求头
            },
            'body': json.dumps({'response': result['choices'][0]['message']['content']})
        }

    except requests.exceptions.RequestException as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
