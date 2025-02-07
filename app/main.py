from flask import Flask, render_template, request, jsonify
from chatbot import get_chatbot_response  # 你的聊天逻辑

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # 渲染前端页面

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = get_chatbot_response(user_input)  # 调用聊天逻辑
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
