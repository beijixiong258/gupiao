# A 股 T+3 量化研究员使用说明

这是一个在 Windows 命令行中使用的 A 股研究助手。你可以像使用 ChatGPT 一样连续提问，也可以直接分析一只股票，或从指定行业、概念板块中选股并查看未来 T+1、T+2、T+3 个交易日的研究预测。

本程序只研究中国大陆 A 股，不会连接券商、读取证券账户或自动下单。

## 1. 第一次安装

打开 PowerShell 7，进入项目目录：

```powershell
cd C:\Users\user\PycharmProjects\gupiaoyanjiu
```

如果项目里已经有 `venv`，跳到下一节。没有时执行：

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .\agent\.env.example .\agent\.env
```

以后每次重新打开 PowerShell，只需要进入项目并激活虚拟环境：

```powershell
cd C:\Users\user\PycharmProjects\gupiaoyanjiu
.\venv\Scripts\Activate.ps1
```

命令行开头出现 `(venv)` 就表示激活成功。

## 2. 配置模型和股票数据

用 PyCharm 打开 `agent\.env`。DeepSeek、OpenAI API、ChatGPT Pro 三种方式只启用一种。

### 使用 DeepSeek

```dotenv
LANGCHAIN_PROVIDER=deepseek
LANGCHAIN_MODEL_NAME=deepseek-chat
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 使用 OpenAI API Key

```dotenv
LANGCHAIN_PROVIDER=openai
LANGCHAIN_MODEL_NAME=gpt-5.6
OPENAI_API_KEY=你的OpenAI_API_Key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 使用 ChatGPT Pro 登录

先在已经激活虚拟环境的 PowerShell 中运行：

```powershell
gpyj openai-login
```

按照浏览器提示完成登录，然后在 `agent\.env` 中设置：

```dotenv
LANGCHAIN_PROVIDER=openai_codex
LANGCHAIN_MODEL_NAME=gpt-5.3-codex
OPENAI_CODEX_BASE_URL=https://chatgpt.com/backend-api/codex
```

退出 ChatGPT 登录：

```powershell
gpyj openai-logout
```

### 配置 Tushare

把 Tushare Token 写入 `agent\.env`：

```dotenv
TUSHARE_TOKEN=你的Tushare_Token
```

不配置或接口权限不足时，部分数据会自动改用 AKShare。检查当前配置：

```powershell
gpyj settings
```

## 3. 连续聊天

激活虚拟环境后直接运行：

```powershell
gpyj
```

`gpyj chat` 效果相同。程序默认续接最近一次会话，可以连续追问：

```text
你 > 分析一下贵州茅台的基本面和技术面
你 > 那它未来三个交易日怎么看？
你 > 和刚才白酒板块里的第二只比较一下
```

聊天中的常用命令：

| 命令 | 用途 |
|---|---|
| `/new` | 新建空白会话 |
| `/clear` | 清空当前会话 |
| `/sessions` | 查看最近 10 个会话和会话 ID |
| `/resume 会话ID` | 切换到指定会话 |
| `/history` | 查看当前会话最近内容 |
| `/help` | 查看会话命令 |
| `/exit` | 保存并退出 |

启动时直接新建或打开指定会话：

```powershell
gpyj chat --new
gpyj chat --session 20260715_120000_abcdef
```

会话使用 UTF-8 保存在 `%USERPROFILE%\.gupiaoyanjiu\duihua\`。其中不会保存 API Key、Tushare Token 或证券账户信息。

## 4. 常见提问方式

分析一只股票：

```text
分析贵州茅台的基本面、估值和技术走势
看看 600519.SH 目前风险大不大
宁德时代未来三个交易日怎么看
```

从指定板块选股：

```text
从白酒板块选 3 只股票，并预测未来三个交易日
从人工智能概念中找短线量化表现最好的 5 只
分析银行板块，模型没优势就不要推荐
```

板块名称必须明确。程序不会在没有范围的情况下扫描整个 A 股市场。

## 5. 单次提问

不进入连续聊天，执行一次后返回 PowerShell：

```powershell
gpyj run -p "分析 600519.SH 的基本面和技术面"
gpyj run -p "从白酒板块选 3 只股票，并预测未来 3 个交易日"
```

## 6. 不使用大模型，直接运行量化工具

直接分析单股：

```powershell
gpyj gupiao 600519.SH
gpyj gupiao 贵州茅台
```

直接进行板块选股：

```powershell
gpyj bankuai 白酒 --top-n 3
gpyj bankuai 人工智能 --type gainian --top-n 5
gpyj bankuai 银行 --type hangye --top-n 3
```

数据源默认使用 `auto`。需要排查数据差异时可以指定：

```powershell
gpyj gupiao 600519.SH --source tushare
gpyj gupiao 600519.SH --source akshare
gpyj bankuai 白酒 --source akshare --top-n 3
```

## 7. 查看运行记录

```powershell
gpyj list
gpyj list --limit 50
```

## 8. MCP 用法

让支持 MCP 的其他智能体通过 stdio 调用：

```powershell
gpyj-mcp
```

启动仅监听本机的 HTTP MCP：

```powershell
gpyj-mcp --transport http --host 127.0.0.1 --port 8765
```

HTTP 地址：`http://127.0.0.1:8765/mcp`。

MCP 只提供单股分析和板块选股两个只读工具，不提供下单功能。

## 9. 调整研究参数

根目录有两个可以直接编辑的 JSON 文件：

- `lianghua_peizhi.json`：历史天数、股票过滤、模型参数、最多研究股票数和 T+1/T+2/T+3 权重。
- `jiaoyi_chengben.json`：佣金、最低佣金、过户费、印花税和滑点。

修改后重新运行命令即可生效。预测周期只能是 T+1、T+2、T+3；自动交易相关选项必须保持关闭，否则程序会拒绝运行。

## 10. 等待时间和中断

单股分析通常需要 10 到 30 秒。板块选股需要逐只拉取行情并训练模型，通常需要 30 秒到 3 分钟，网络重试或板块股票较多时会更久。

终端仍显示“正在拉取”“正在训练”或没有重新出现 PowerShell 提示符时，通常表示程序还在工作。按 `Ctrl+C` 可以中断当前一轮；连续聊天中已经完成的历史不会丢失，也不会产生任何交易动作。

## 11. 常见问题

### 找不到 `gpyj` 命令

确认当前目录正确并且已经激活虚拟环境：

```powershell
cd C:\Users\user\PycharmProjects\gupiaoyanjiu
.\venv\Scripts\Activate.ps1
python -m pip install -e .
```

### Preflight 显示 LLM 失败

运行 `gpyj settings`，检查 `agent\.env` 中选择的 Provider、模型名称和对应密钥。修改 `.env` 后必须退出程序并重新启动。

### Tushare 显示未配置或权限不足

检查 `TUSHARE_TOKEN` 是否已经写入 `agent\.env`。120 积分并不拥有所有接口权限，程序会在可行时自动使用 AKShare 补充。

### AKShare 或东方财富出现代理 443 错误

保持 `lianghua_peizhi.json` 中的 `akshare_bypass_proxy` 为 `true`。这样程序访问中国大陆数据源时会临时绕过 Clash 代理，调用结束后恢复，不影响 DeepSeek 或 OpenAI。

### 中文乱码

优先使用 PowerShell 7。仍然乱码时，在当前终端执行：

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
```

然后重新运行 `gpyj`。

## 12. 怎样理解预测结果

- T+1、T+2、T+3 指后续第 1、2、3 个交易日，不是自然日。
- 模型验证不通过、扣除成本后没有优势或股票不可执行时，程序会明确不推荐。
- 预测是基于历史数据的研究结果，不保证未来收益。
- 所有买入、卖出、仓位和风险决定都由用户在程序之外人工完成。
