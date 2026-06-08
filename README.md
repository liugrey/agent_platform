# 智能体平台

> 接收前端文字 → 调用大模型 API → 返回回答
> 支持多 Agent 配置、角色管理、SSE 流式输出、SQLite 会话记忆、配置知识库、向量数据库

---

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入 API Key 和选择 LLM Provider
```

### 3. 启动服务
```bash
python main.py
# 或
uvicorn main:app --reload
```

服务启动后访问：
- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

---

## 切换 LLM Provider

只需修改 `.env` 中的 `LLM_PROVIDER`，重启服务即生效，代码无需修改：

| Provider | .env 配置 | 说明 |
|---|---|---|
| OpenAI | `LLM_PROVIDER=openai` | 默认，兼容 DeepSeek / 通义 / Azure |
| 阿里云百炼 | `LLM_PROVIDER=bailian` | 使用独立的百炼配置 |
| Anthropic | `LLM_PROVIDER=anthropic` | Claude 系列 |
| Ollama | `LLM_PROVIDER=ollama` | 本地运行，无需 API Key |

**接入阿里云百炼**，推荐使用独立 provider：
```bash
LLM_PROVIDER=bailian
BAILIAN_API_KEY=sk-xxxxxxxx
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_MODEL=qwen-turbo            # 或 qwen-plus / qwen-max
```

如果你想继续走 OpenAI 兼容配置，也仍然可以使用：
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo
```

---

## 接口说明

### 基础对话 `/chat`

#### 单次对话（阻塞，无记忆）
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，请介绍一下你自己"}'
```

#### 单次对话（SSE 流式，无记忆）
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "用 100 字介绍 Python"}'
```

#### 多轮对话（阻塞，带会话记忆）
```bash
# 第一轮（不传 session_id，自动创建）
curl -X POST http://localhost:8000/chat/session \
  -H "Content-Type: application/json" \
  -d '{"message": "我叫小明"}'
# 返回 {"reply": "...", "session_id": "abc-123", ...}

# 第二轮（传入上一轮的 session_id）
curl -X POST http://localhost:8000/chat/session \
  -H "Content-Type: application/json" \
  -d '{"message": "我叫什么名字？", "session_id": "abc-123"}'
```

#### 多轮对话（SSE 流式，带会话记忆）
```bash
curl -X POST http://localhost:8000/chat/session/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "继续我们的话题", "session_id": "abc-123"}'
```

#### 查询历史消息 / 会话列表
```bash
curl http://localhost:8000/chat/history/abc-123
curl http://localhost:8000/chat/sessions
```

---

### Agent 管理 `/agents`

每个 Agent 有独立的角色、模型、系统提示词配置，不同 Agent 可以使用不同模型。

#### 查看所有预置角色类型
```bash
curl http://localhost:8000/agents/roles
```

内置角色列表：

| role_type | 角色名 | 适用场景 |
|---|---|---|
| `assistant` | 通用助手 | 日常问答、信息查询 |
| `customer_service` | 客服专员 | 用户咨询、投诉、售后 |
| `code_reviewer` | 代码审查员 | Bug 分析、优化建议 |
| `translator` | 翻译专家 | 多语言翻译 |
| `data_analyst` | 数据分析师 | 数据解读、业务洞察 |
| `writer` | 文案写手 | 营销文案、报告、文章 |
| `teacher` | 知识导师 | 复杂概念讲解、辅助学习 |
| `custom` | 自定义 | 完全自定义系统提示词 |

#### 创建 Agent
```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "客服小助",
    "role_type": "customer_service",
    "model": "qwen-max",
    "provider": "openai",
    "description": "负责处理用户售后咨询"
  }'
```

`system_prompt` 不传时自动使用角色预设，传了则完全覆盖预设：
```bash
# 完全自定义提示词（role_type 用 custom）
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "法律顾问",
    "role_type": "custom",
    "model": "qwen-plus",
    "provider": "openai",
    "system_prompt": "你是一名专业律师，擅长合同纠纷和劳动法..."
  }'
```

#### 查询 / 更新 / 删除 Agent
```bash
# 查询列表（可按角色类型筛选）
curl "http://localhost:8000/agents?role_type=customer_service"

# 查询单个
curl http://localhost:8000/agents/{agent_id}

# 更新（部分字段，只传需要改的）
curl -X PUT http://localhost:8000/agents/{agent_id} \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-turbo", "temperature": 0.5}'

# 删除（软删除，数据保留）
curl -X DELETE http://localhost:8000/agents/{agent_id}
```

#### 使用 Agent 对话（阻塞）
```bash
curl -X POST http://localhost:8000/agents/{agent_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我的订单三天没发货怎么办"}'
```

#### 使用 Agent 对话（SSE 流式）
```bash
curl -X POST http://localhost:8000/agents/{agent_id}/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我审查这段代码", "session_id": "可选，不传自动创建"}'
```

---

### SSE 流式事件格式

所有 `/stream` 接口统一使用以下事件格式：

```
data: {"type": "chunk",  "content": "文字片段"}                    ← 逐字推送
data: {"type": "done",   "session_id": "...", "model": "..."}      ← 结束
data: {"type": "error",  "message": "错误信息"}                    ← 出错
```

**前端接入示例（JavaScript）：**
```javascript
const res = await fetch('http://localhost:8000/agents/{agent_id}/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: '你好', session_id: null })
})

const reader = res.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break

  for (const line of decoder.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue
    const event = JSON.parse(line.slice(6))

    if (event.type === 'chunk') {
      output.textContent += event.content       // 追加文字
    } else if (event.type === 'done') {
      currentSessionId = event.session_id       // 保存会话 ID 供下轮使用
    } else if (event.type === 'error') {
      console.error(event.message)
    }
  }
}
```

---

## 目录结构

```
agent_platform_v1/
├── main.py                   ← 启动入口，路由注册
├── .env.example              ← 环境变量配置模板
├── requirements.txt          ← 依赖清单
│
├── config/
│   └── settings.py           ← 读取 .env，统一配置管理
│
├── core/
│   ├── llm_client.py         ← LLM 调用封装（阻塞 + 流式，多 provider）
│   └── agent_runner.py       ← Agent 执行器（按配置运行对话）
│
├── storage/
│   ├── db.py                 ← SQLite 会话/消息存储
│   └── agent_store.py        ← Agent 配置 CRUD + 角色预设
│
└── api/
    ├── chat.py               ← /chat 基础对话接口（阻塞 + SSE）
    └── agents.py             ← /agents Agent 管理 + Agent 对话接口
```
