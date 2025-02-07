// 监听按钮点击事件
function sendMessage() {
    // 获取用户输入的消息
    const userInput = document.getElementById("chat-input").value;
    
    // 如果用户没有输入内容，则弹出提示
    if (!userInput.trim()) {
        alert("请输入消息");
        return;
    }

    // 显示加载中的提示
    document.getElementById("loading").style.display = "block";

    // 清空输入框
    document.getElementById("chat-input").value = "";

    // 发送请求到后端
    fetch("https://msdocs-python-webapp-quickstart-mc-cfbudtbbe0bua9ee.eastasia-01.azurewebsites.net/ask", { // 修改为你实际的 Azure URL
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

        // 如果返回的是响应内容，则展示在聊天窗口
        if (data.response) {
            const messageDiv = document.createElement("div");
            messageDiv.classList.add("message");
            messageDiv.textContent = data.response;
            document.getElementById("messages").appendChild(messageDiv);
        } else if (data.error) {
            // 如果返回的是错误信息，则展示错误
            const errorDiv = document.createElement("div");
            errorDiv.classList.add("message");
            errorDiv.textContent = `错误: ${data.error}`;
            document.getElementById("messages").appendChild(errorDiv);
        }
    })
    .catch(error => {
        // 隐藏加载中的提示
        document.getElementById("loading").style.display = "none";
        
        // 如果发生网络或其他错误，显示错误信息
        console.error("Error:", error);
        const errorDiv = document.createElement("div");
        errorDiv.classList.add("message");
        errorDiv.textContent = "请求失败，请稍后重试";
        document.getElementById("messages").appendChild(errorDiv);
    });
}

// 切换浅色/深色主题
function toggleTheme() {
    document.body.classList.toggle("dark-theme");
}

// 展开或收起下拉菜单
function toggleDropdown(event) {
    const dropdownMenu = document.getElementById("dropdownMenu");
    dropdownMenu.style.display = dropdownMenu.style.display === "block" ? "none" : "block";
}

// 监听回车键来发送消息
document.getElementById("chat-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
