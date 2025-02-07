document.getElementById('send_button').addEventListener('click', function() {
    var userInput = document.getElementById('user_input').value;
    var chatLog = document.getElementById('chatlog');

    // 显示用户输入
    chatLog.innerHTML += "<div><b>你:</b> " + userInput + "</div>";

    // 调用后端 Flask API 获取聊天机器人的回答
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        chatLog.innerHTML += "<div><b>机器人:</b> " + data.response + "</div>";
        document.getElementById('user_input').value = '';  // 清空输入框
        chatLog.scrollTop = chatLog.scrollHeight;  // 滚动到底部
    })
    .catch(error => {
        console.error('请求错误:', error);
    });
});
