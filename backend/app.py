import json
import requests
from flask import Flask, request, jsonify

# 这里是你从 SiliconFlow 获取的 API 密钥
API_KEY = "sk-rhshxzvfrtcfiqqfkqqraeodtytxpmrilcytxehcppdxwsqx"  # 替换为你自己的 API 密钥

# 创建 Flask 应用
app = Flask(__name__)

# 处理聊天请求的 API 路由
@app.route('/ask', methods=['POST'])
def handle_request():
    try:
        # 获取用户输入，假设输入来自 HTTP POST 请求的 JSON 数据
        data = request.get_json()
        user_input = data.get('user_input', '')

        if not user_input:
            return jsonify({'error': '用户输入不能为空'}), 400  # 如果用户没有提供输入，返回错误

        # 定义请求的 API URL
        url = "https://api.siliconflow.cn/v1/chat/completions"
        
        # 请求的 payload 数据，包含模型参数等
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 使用的模型
            "temperature": 0.9,  # 控制生成的多样性
            "max_tokens": 1024,  # 控制生成的最大字符数
            "messages": [
                {"role": "system", "content": "**角色设定：**   你是MCMocoder，一个活跃于二次元群聊的戏精用户， 一次只回复一句话"},
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

        # 发送 POST 请求到 SiliconFlow API
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 如果请求失败，抛出异常

        # 解析 API 响应
        result = response.json()

        # 返回机器人生成的内容
        return jsonify({'response': result['choices'][0]['message']['content']}), 200

    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': '服务器内部错误'}), 500

# 运行 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

