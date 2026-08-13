# A 股分析与未来三交易日预测

这是一个给个人使用的中国大陆 A 股日 K 研究程序。产品定位只有两件事：

1. 分析一只股票未来涨跌的可能性，并说明依据。
2. 预测这只股票未来第 1、2、3 个交易日的方向、参考收盘价和收益估计。

程序运行时需要联网取得行情、交易日历和基本面数据；本地缓存只用于减少重复请求。除这两个研究功能外，程序不提供其它业务，也不执行自动交易、分钟级预测或超过三个交易日的预测。

完整的已实现行为、输出字段和限制见 [程序介绍](./程序介绍.md)；维护者设计决策见 [项目设计与实现](./项目设计与实现.md)。

## 快速开始

运行环境：Windows、PowerShell 7（`pwsh`）、Python 3.11 或更高版本。

```powershell
cd C:\Users\user\PycharmProjects\gupiaoyanjiu
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .\agent\.env.example .\agent\.env
```

在 `agent\.env` 配置一个模型 Provider，并按需配置 `TUSHARE_TOKEN`。例如：

```dotenv
LANGCHAIN_PROVIDER=deepseek
LANGCHAIN_MODEL_NAME=deepseek-chat
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
TUSHARE_TOKEN=你的Token
```

也可以使用 OpenAI API，或执行 `gpyj openai-login` 使用 ChatGPT/Codex 登录。`gpyj settings` 可查看当前 Provider 和数据源配置。

## 两项业务

```text
股票代码或名称
      |
      v
gupiao_fenxi  -> 方向结论、可能性依据、证据、analysis_id
      |
      v
gupiao_yuce(analysis_id) -> T+1、T+2、T+3 一次性预测
```

### 方向分析

```powershell
gpyj gupiao 600519.SH
gpyj gupiao 贵州茅台 --source auto --json
```

分析固定使用最近一个已确认的完整收盘日，结果包含行情时点、技术面、基本面、波动、可交易性、同行证据和风险提示，并给出“偏上涨”“偏下跌”或“中性/不确定”。第一阶段只输出因子证据，概率字段固定为空；三日概率和数值只在第二阶段返回，不会补造数字。

分析阶段不是只看均线。因子工程按以下 8 组计算并在结果中逐组解释（当前数据缺失的因子会标为“信息有限”）：

- 趋势结构：均线间距、趋势斜率、拟合质量、金叉速度；
- 动量与反转：多周期收益、RSI、MACD；
- K 线压力：跳空、日内收益、收盘位置、实体和上下影线；
- 价量确认：量比、成交额异常、价量相关、量价残差；
- 突破与回撤质量：突破距离、放量确认、回撤质量、停滞压力；
- 相对强弱：相对同行、行业、市场基准的超额收益和排名；
- 风险与流动性：ATR、波动率、振幅、回撤、成交额、换手和市值流动性；
- 市场背景：市场/行业平均收益、上涨广度、离散度和市场状态。

普通终端输出会先给一轮 LLM 通俗解读，再给出“因子分组结果”表；LLM 只翻译程序已计算的证据，不把分数或分位数改写成概率、收益率或目标价。

### 三日预测

```powershell
gpyj yuce 600519.SH
gpyj yuce 贵州茅台 --json
```

命令会先完成一次分析，再用同一 `analysis_id` 调用一次预测。预测一次返回 `T+1`、`T+2`、`T+3`，不接受更远期限或额外的资金、持仓参数。

## 自然语言与 MCP

连续对话：

```powershell
gpyj
```

连续对话中输入 `/exit` 会保存当前会话并退出；也可以按 `Ctrl+C` 退出当前输入。正常退出不会启动预测模型。

单轮提问：

```powershell
gpyj run -p "分析 600519.SH 的涨跌可能性"
gpyj run -p "预测贵州茅台未来三个交易日" --json
```

智能体只注册两个业务工具：

- `gupiao_fenxi(gupiao, source?, history_calendar_days?)`
- `gupiao_yuce(analysis_id)`

MCP 对外提供同名的两个工具；MCP 的第一阶段参数为 `gupiao`、`source` 和 `history_calendar_days`，第二阶段仍只接受 `analysis_id`。

启动 MCP：

```powershell
gpyj-mcp
gpyj-mcp --transport http --host 127.0.0.1 --port 8765
```

服务只返回研究数据，不连接券商、不读取账户凭据、不提交委托。

## 如何理解结果

- 方向结论是当前证据的统计倾向，不是涨跌承诺。
- 第一阶段的 `positive_probability` / `negative_probability` 固定为空：它只给因子证据；这两个数值只在第二阶段三日预测中由模型或历史样本给出。
- `predicted_close` 是从信号收盘推导的参考值，不是目标价或成交承诺。
- 第一阶段的 `confidence` 描述证据完整度；第二阶段的 `confidence` 才描述模型验证和样本质量。验证未通过时仍可能发布模型估计，但必须降低可信度。
- 交易决定、价格和仓位始终由用户自行判断。

## 代码入口

| 路径 | 职责 |
|---|---|
| `agent/src/tools/gupiao_fenxi_tool.py` | 第一阶段公开契约和方向摘要 |
| `agent/src/tools/gupiao_yuce_tool.py` | 第二阶段固定三日预测契约 |
| `agent/src/ashare/gupiao_yanjiu.py` | 单股数据时点、基本面、技术面和结果装配 |
| `agent/src/ashare/dangu_yuce.py` | 日 K 因子、同行面板、滚动验证和预测底稿 |
| `agent/src/ashare/yinzi_gongcheng.py` | 因子清洗、稳定性筛选和训练窗特征工程 |
| `agent/src/agent/context.py` | 两阶段语义路由和解释边界 |
| `agent/cli.py` | PowerShell CLI |
| `agent/mcp_server.py` | 两工具 MCP 服务 |

## 安全边界

程序永久保持 `research_only`：不连接证券账户，不读取或保存交易密码，不控制交易终端，不提交、修改或撤销委托，不自动执行买卖。输出仅用于个人研究和人工复核。
