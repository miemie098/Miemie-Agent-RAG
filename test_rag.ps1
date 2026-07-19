# 1. 组装符合你 main.py 规范的流式入参
$body = @{ "question" = "什么是高并发代理？" } | ConvertTo-Json -Compress

# 2. 建立针对 /chat/stream 路由的长连接请求
$request = [System.Net.WebRequest]::Create("http://localhost:8000/chat/stream")
$request.Method = "POST"
$request.ContentType = "application/json"

# 3. 强行灌入参数字节流
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$request.ContentLength = $bytes.Length
$stream = $request.GetRequestStream()
$stream.Write($bytes, 0, $bytes.Length)
$stream.Close()

# 4. ⚡ 核心提效：以流式（Stream）逐行捕获 K8s 集群吐出来的 Token 瀑布
try {
    $response = $request.GetResponse()
    $reader = [System.IO.StreamReader]::New($response.GetResponseStream(), [System.Text.Encoding]::UTF8)
    
    Write-Host "=== 🚀 正在接收来自 K8s 分布式集群的 LangGraph 实时生成流... ===" -ForegroundColor Cyan
    
    while ($null -ne ($line = $reader.ReadLine())) {
        if ($line.StartsWith("data: ")) {
            # ✂️ 剔除前缀 "data: "
            $jsonStr = $line.Substring(6).Trim()

            # 🛑 过滤结束标志
            if ($jsonStr -eq "[DONE]") { continue }

            if ($jsonStr) {
                try {
                    # ⚡ 终极修复：使用 PowerShell 原生反序列化引擎，强行将 \uXXXX 转义码还原为纯正的中文物理字体
                    $jsonObj = $jsonStr | ConvertFrom-Json
                    if ($jsonObj.answer) {
                        # 以绿色高亮打字机流式输出纯正中文
                        Write-Output $jsonObj.answer
                    }
                } catch {
                    # 防止由于流片段不完整导致的临时解析错误
                }
            }
        }
    }
    
    $reader.Close()
    $response.Close()
} catch {
    Write-Host "⚠️ 传输中断或响应异常: $_" -ForegroundColor Red
}