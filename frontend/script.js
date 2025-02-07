// 监听按钮点击事件
function sendMessage() {
    // 获取用户输入的消息
    const userInput = document.getElementById("chat-input").value;
    
    if (!userInput.trim()) {
        alert("请输入消息");
        return;
    }

    // 显示加载中的提示
    document.getElementById("loading").style.display = "block";

    // 清空输入框
    document.getElementById("chat-input").value = "";

    // 发送请求到后端
    fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            user_input: userInput
        })
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载中的提示
        document.getElementById("loading").style.display = "none";

        // 显示返回的聊天内容
        if (data.response) {
            const messageDiv = document.createElement("div");
            messageDiv.classList.add("message");
            messageDiv.textContent = data.response;
            document.getElementById("messages").appendChild(messageDiv);
        } else if (data.error) {
            const errorDiv = document.createElement("div");
            errorDiv.classList.add("message");
            errorDiv.textContent = `错误: ${data.error}`;
            document.getElementById("messages").appendChild(errorDiv);
        }
    })
    .catch(error => {
        document.getElementById("loading").style.display = "none";
        console.error("Error:", error);
    });
}

// 切换主题（浅色/深色）
function toggleTheme() {
    document.body.classList.toggle("dark-theme");
}

// 展开/收起下拉菜单
function toggleDropdown(event) {
    const dropdownMenu = document.getElementById("dropdownMenu");
    dropdownMenu.style.display = dropdownMenu.style.display === "block" ? "none" : "block";
}
