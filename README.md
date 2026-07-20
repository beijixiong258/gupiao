# A 股 T+3 量化研究员使用说明

这是一个在 Windows 命令行中使用的 A 股研究助手。你可以像使用 ChatGPT 一样连续提问，也可以直接分析一只股票，或从指定行业、概念板块中选股并查看 T+1、T+2、T+3 三个可卖出周期的研究预测。

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
LANGCHAIN_MODEL_NAME=gpt-5.6-luna
LANGCHAIN_REASONING_EFFORT=medium
LANGCHAIN_SERVICE_TIER=fast
OPENAI_CODEX_BASE_URL=https://chatgpt.com/backend-api/codex
```

`fast` 会提高响应速度，也会按 OpenAI 当前规则消耗更多 ChatGPT 额度；程序会把它转换成后端实际接受的 `priority` 请求值。不需要时删除该行。

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

程序按以下顺序读取第一份存在的环境配置：`%USERPROFILE%\.gupiaoyanjiu\.env`、`agent\.env`、当前目录的 `.env`。一般只编辑 `agent\.env`；如果修改后没有生效，先检查用户目录下是否已有优先级更高的配置。

## 3. 连续聊天

激活虚拟环境后直接运行：

```powershell
gpyj
```

`gpyj chat` 效果相同。程序默认续接最近一次会话。

### 启动示范

程序显示 `你 >` 后，可以直接用股票名称、计划持有时间和问题一起提问：

```text
你 > 深科技现在怎么样，我想持有两个交易日，能不能买？
```

智能体会把“深科技”识别为 `000021.SZ`，并从完整语义理解用户要看两日上涨空间。它先调用 `gupiao_fenxi` 做行情时点、基本面、估值、技术面、波动、可交易性、同行和风险的全面诊断；因为问题还涉及能否上涨，再使用诊断返回的 `analysis_id` 调用 `gupiao_yuce`，按 T+2 计算用户真正需要的数值。链路由大模型按语义选择，不靠“持有两个交易日”等文字的正则硬匹配。

“能不能买”按扣除广义交易费用后是否仍有上涨空间理解；“能不能卖”先分析剩余上涨空间，如果缺少持仓信息，分析完成后会再询问买入价和股数或持仓金额，以便计算净盈亏。费用覆盖佣金及最低佣金、过户费、卖出印花税、双边滑点和合法申报数量。指定周期未通过样本外验证或预测时点已经失效时，第二阶段不会把内部原始点预测展示给用户。

首轮回答后可以继续追问，程序会保留当前会话上下文：

```text
你 > 如果明天高开，还适合买入吗？
你 > 哪些信号说明两日持有的逻辑已经失效？
你 > 把刚才提到的主要风险按重要程度排列
```

自然语言交互分为两条链路：

1. **量化分析链路**：问题需要当前行情、新股票或板块、不同持有期限、指标计算、模型预测或其他确定性结果时，智能体调用对应工具，再解释工具结果。
2. **直接聊天链路**：不需要量化工具时直接回答，例如解释已有结果、A 股概念或程序用法。问题与 A 股分析、预测和程序使用无关时，不调用工具，只用一句简短提示请用户回到程序主业，不展开闲聊。

链路选择由智能体结合完整问题和会话上下文进行语义判断，不使用关键词或正则表达式硬匹配。程序升级前保存的旧版单股工具结果和它生成的文字结论仍会自动标记为过期，不能作为新回答的依据。

聊天中的常用命令：

| 命令 | 用途 |
|---|---|
| `/new` | 新建空白会话 |
| `/clear` | 清空当前会话 |
| `/clear-history` | 清除全部历史会话，执行前需要输入“确认清除” |
| `/sessions` | 查看最近 10 个会话和会话 ID |
| `/resume 会话ID` | 切换到指定会话 |
| `/history` | 查看当前会话最近内容 |
| `/help` | 查看会话命令 |
| `/exit` | 保存并退出 |

程序启动时会直接显示全部斜杠命令。启动时直接新建或打开指定会话：

```powershell
gpyj chat --new
gpyj chat --session 20260715_120000_abcdef
```

会话使用 UTF-8 保存在 `%USERPROFILE%\.gupiaoyanjiu\duihua\`，智能体运行目录也会保存本轮输入和输出。程序不会主动把配置文件中的 API Key 或 Tushare Token 写进这些记录，但会完整保存用户消息、助手回答和工具结果。不要在聊天中粘贴密钥、交易密码或其他敏感信息。

## 4. 常见提问方式

分析一只股票：

```text
深科技现在怎么样，我想持有两个交易日，能不能买？
分析贵州茅台的基本面、估值和技术走势
看看 600519.SH 目前风险大不大
宁德时代当前技术趋势和波动风险怎么样
```

从指定板块选股：

```text
从白酒板块选 3 只股票，并比较入场后的 T+1、T+2、T+3 可卖出周期
从人工智能概念中找短线量化表现最好的 5 只
分析银行板块，只列出通过验证的候选
```

板块选股未说明数量时默认请求 Top 8，单批最多返回 8 只通过证据门槛的研究候选；如果用户明确要求超过 8 只，程序先说明单批上限，再正常返回第一批 Top 8。用户说“不满意、换一批、继续”时，程序使用稳定序列编号顺延到第 9～16 名，之后继续顺延，不重复前一批。8 是上限而不是必须凑满的数量，只有更少股票通过验证时就如实返回更少。

单股量化使用两个工具阶段。`gupiao_fenxi` 公开基本面、技术面、可交易性和风险诊断，但不公开内部预测；`gupiao_yuce` 只发布用户指定的 T+1、T+2 或 T+3 数值。持有期预测假设下一交易日开盘作为测算基准，到指定可卖出收盘计算收益；由于实际开盘价尚未知，程序不会伪造精确目标价。板块选股仍要求明确板块名称，不会无范围地扫描整个 A 股市场。

如果问题是“未来三个交易日大概怎么走”，智能体仍先完成诊断，再让 `gupiao_yuce` 按用户指定的周期输出结果。未来收盘模式以最近完整收盘价为基准，预测未来第 1、2、3 个市场交易日收盘的累计收益、参考收盘价、80% 经验区间和验证状态。这里的 T+1 就是下一个交易日，与“假设下一交易日开盘作为测算基准”的持有期情景相互独立。

## 5. 单次提问

不进入连续聊天，执行一次后返回 PowerShell：

```powershell
gpyj run -p "分析 600519.SH 的基本面和技术面"
gpyj run -p "从白酒板块选 3 只股票，并比较入场后的 T+1、T+2、T+3 可卖出周期"
```

## 6. 不使用大模型，直接运行量化工具

直接执行第一阶段单股诊断：

```powershell
gpyj gupiao 600519.SH
gpyj gupiao 贵州茅台
gpyj gupiao 深科技 --holding-days 2
gpyj gupiao 600519.SH --holding-days 3 --budget-yuan 50000 --history-calendar-days 1080
```

只查看未来三个交易日预测：

```powershell
gpyj yuce 600519.SH
gpyj yuce 贵州茅台 --source auto
gpyj yuce 深科技 --json
```

`gupiao` 只公开第一阶段诊断和 `analysis_id`，不直接暴露内部模型点预测。`yuce` 会先执行同一诊断，再分别经过验证门禁输出 T+1、T+2、T+3；未通过验证的周期只返回不可用原因。这里的 T+1、T+2、T+3 分别表示最近完整收盘日之后第 1、2、3 个市场交易日收盘。

直接进行板块选股：

```powershell
gpyj bankuai 白酒 --top-n 3
gpyj bankuai 人工智能 --type gainian --top-n 5
gpyj bankuai 银行 --type hangye --top-n 3
```

`--top-n` 默认 8 且单批最多 8。根据上一批 JSON 中的 `selection.selection_id` 和 `selection.next_offset` 顺延下一批：

```powershell
gpyj bankuai 白酒 --top-n 8 --offset 8 --selection-id sel_上一批返回的编号 --json
```

需要临时使用另一份量化配置时，可以执行：

```powershell
gpyj bankuai 白酒 --config .\lianghua_peizhi.json --top-n 3
```

日线行情来源默认使用 `auto`。需要排查行情差异时可以指定：

```powershell
gpyj gupiao 600519.SH --source tushare
gpyj gupiao 600519.SH --source akshare
gpyj bankuai 白酒 --source akshare --top-n 3
```

对于 `gpyj gupiao`，`--source tushare` 或 `--source akshare` 会固定股票名称解析和日线行情来源，`auto` 才按 Tushare、AKShare 顺序降级；对于 `gpyj bankuai`，它只控制成分股日线行情。基本面仍按 Tushare 优先、AKShare 补充的策略获取；板块成分也按 Tushare、AKShare 新浪、AKShare 东方财富的独立顺序获取，因此一次结果可能包含多个来源。

需要让其他程序读取结果时，`run`、`gupiao`、`yuce`、`bankuai` 都支持 `--json`。`run` 和 `chat` 还支持 `--max-iter` 调整单轮最多工具循环次数。

## 7. 查看运行记录

```powershell
gpyj list
gpyj list --limit 50
```

这里列出的是 `gpyj`、`gpyj chat` 和 `gpyj run` 产生的智能体运行目录。直接执行 `gpyj gupiao` 或 `gpyj bankuai` 不会创建这类运行记录。

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

默认地址只允许本机访问。程序也接受其他 `--host` 值；如果改成 `0.0.0.0` 等对外监听地址，需要自行配置网络访问控制。

MCP 只提供 `gupiao_fenxi`、`gupiao_yuce` 和 `bankuai_xuangu` 三个非交易工具，不提供下单功能，也不会修改证券账户状态；取数时可能更新项目内的本地缓存。

## 9. 调整研究参数

根目录有两个可以直接编辑的 JSON 文件：

- `lianghua_peizhi.json`：单股同行池、滚动验证、证据评估门槛，以及板块模型历史天数、股票数量、过滤规则、样本外验证门槛和 T+1/T+2/T+3 权重。
- `jiaoyi_chengben.json`：佣金、最低佣金、过户费、印花税和滑点。

单股分析和板块选股都会读取并校验这些量化参数，包括数值范围、样本数量、过滤条件、模型参数、成本费率和合法执行模式；无效的交易成本配置会明确记录错误并退回内置的非负默认成本。数据源不在 JSON 中设置，应使用每次命令的 `--source`。预测周期只能是 T+1、T+2、T+3。程序会拒绝启用自动交易相关配置；无论配置如何，本程序都没有券商连接和下单工具。

## 10. 等待时间和中断

单股分析需要建立同行历史面板并完成三段滚动验证，通常需要 30 秒到 3 分钟。板块选股同样需要逐只拉取行情并训练模型，通常需要 30 秒到 3 分钟；数据源重试、限频或可用股票较多时会更久。

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

- 程序在 T 日收盘数据完整后形成分析截面，持有期情景假设下一市场交易日开盘价作为测算基准；不会假设能够回到 T 日收盘价成交。
- 未来收盘模式是独立的走势预测：从 T 日完整收盘价预测未来第 1、2、3 个交易日收盘，不代表这些日期都能按新买入股票执行卖出。
- 如果行情源尚未更新，而第一个预测交易日已经收盘，程序会把未来三日预测标为不可用，等待最新完整日线后重算，不会把已经发生的日期称为“未来”。
- 输出中的 T+1、T+2、T+3 指入场后第 1、2、3 个**可卖出**交易日。受 A 股当日买入不能当日卖出约束，T+1 的退出日实际是信号后的第 2 个市场交易日。
- 单股问题中的“持有两个交易日”在语义上对应 T+2 数值，不再由技术指标启发式打分代替期限模型，也不靠正则表达式强制路由。
- 盘中询问时，实时价只用于检查成交与涨停状态。如果最近完整收盘信号对应的次日开盘已经过去，程序会标记原持有期情景入口已经失效，不会把盘中价格伪装成模型入口。
- 单股模型优先选取当前同行股票，再用少量全市场高流动性股票补足，使用三段扩展窗口滚动验证；当前成分回看历史仍存在幸存者偏差，结果会明确披露。
- 单股只输出“证据偏正面、证据中性、证据偏负面、证据不足”的分析概括，并同时展示形成该概括的原始指标。该标签不是买入、卖出或持有指令。
- 停牌股票如果缺少规定的入场日或退出日行情，该日期不会被下一根日线冒充。
- 如果计划入场日是一字涨停，程序会把该样本视为无法买入，不用它训练收益标签。
- 板块工具只有通过样本外验证、Top-N 扣成本收益为正的周期才参与排名；未通过周期仍可展示，但会标为不参与排名。
- 交易成本按目标资金、个股价格和所属板块的合法买入数量计算；资金不足以买入最低数量时会明确列为执行约束。
- 实际 T+1 开盘前无法可靠换算持有期情景的参考收盘价，因此 `predicted_close` 返回空值；程序不会给出“最高可接受入场价”或目标价。
- 模型验证不通过或预测时点失效时，程序不会展示内部原始点收益、经验正收益比例和区间；只说明没有通过的具体原因。
- 扣除成本后没有优势或股票存在执行约束时，程序会明确说明证据不足或偏负面，但不会替用户作交易决定。
- 预测是基于历史数据的研究结果，不保证未来收益。
- 程序只解释支持证据、反对证据、风险和不确定性；用户自行判断是否采取行动。
