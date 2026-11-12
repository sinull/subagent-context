# subagent-context

让 Hermes Agent 的 `delegate_task` 子代理**默认继承主 agent 的上下文摘要**的插件。

Hermes 的子代理是全新会话("Subagents Know Nothing"),任务上下文只来自模型填写的 `goal` + `context` 字段。本插件不改核心代码,用两个公开钩子机械地补上这块:

| 钩子 | 作用 |
|---|---|
| `pre_llm_call` | 每个用户回合开始时(工具循环之前)拿到 `conversation_history`,渲染成有界文本摘要,按 `session_id` 缓存 |
| `pre_tool_call` | 拦截 `delegate_task` 调用,返回 `{"action": "modify", "args": {...}}` 改写入参:把摘要前置注入 `context` 字段(单任务形式与批量 `tasks` 形式都处理) |

摘要放在模型手写的任务上下文**之前**,同一批次的兄弟子代理共享相同前缀,仍能复用彼此的 prompt cache。

## 安装

```bash
mkdir -p ~/.hermes/plugins/subagent-context
# 把 plugin.yaml 和 __init__.py 拷进去,然后:
hermes plugins enable subagent-context
hermes plugins doctor subagent-context   # 可选:隔离校验
```

重启 Hermes 生效。之后模型每次调用 `delegate_task`,子代理收到的第一条消息自动变为:

```
[parent-agent context digest]
user: …(主会话最近的用户/助手消息摘要,有界)
assistant: …

[task context from parent agent]
(模型原本填写的 context,如有)
```

## 配置(环境变量,写入 `<HERMES_HOME>\.env`)

| 变量 | 默认 | 说明 |
|---|---|---|
| `SUBAGENT_CONTEXT_MODE` | `digest` | **`full` = 全量继承**:所有消息(含工具结果)、不截断、不限条数 |
| `SUBAGENT_CONTEXT_MAX_CHARS` | 200000 | full 模式安全钳位(字符);`0` = 真·无限(不建议,见下) |
| `SUBAGENT_CONTEXT_BUDGET` | 6000 | digest 模式每任务注入的摘要字符上限 |
| `SUBAGENT_CONTEXT_TAIL` | 30 | digest 模式参与摘要的父会话消息条数上限 |
| `SUBAGENT_CONTEXT_DISABLE` | 未设 | 设为 `1` 即整体停用(逃生门) |

### full 模式的钳位为什么必须有

已核实 v0.23.1 源码:`delegate_task` 的 `context` 字段**没有字符上限**(16k/32k 上限只约束插件生命周期 API)。真正的天花板是**子代理模型的上下文窗口**——而子代理的第一条消息(context+goal)在它自己的上下文压缩中是**逐字保留**的,塞超了就无解。默认 20 万字符 ≈ 5 万 token,128k 窗口的子模型留足工作空间。超钳位时从**最旧端**丢弃并标注省略量(保留最近的上下文)。若你的子模型是长上下文(200k+ token),可自行调大或设 `0`。

full 模式的代价:每个子代理都要为完整父历史付输入 token(批量×N 个任务;同批兄弟因共享前缀可吃 prompt cache);父会话很长时,一次 10 任务批量 = 10 份全量上下文的费用。

## 设计取舍与已知边界

- **只保留 user/assistant 消息**——工具原始输出对子代理是噪音,白白烧掉它的全新上下文窗口
- **纯字符串操作,无 I/O / 无 LLM 调用**——`pre_tool_call` 默认 30s 超时且超时 fail-closed(直接阻断工具),handler 必须快
- **幂等**:注入头 `[parent-agent context digest]` 兼作标记;摘要构建时会跳过已含标记的回流消息,防止多轮委托雪球式膨胀
- **有内存上限**:只缓存最近 64 个 session 的摘要,FIFO 淘汰
- **隔离是有意削弱**:这是本插件的目的,但意味着每个子代理多花 BUDGET 字符的输入 token;上下文投毒面也随之扩大到子代理(批量并行时注意成本)
- **覆盖不到 `/review`**:`/review` 走内部后台路径而非 `delegate_task` 工具分发,钩子拦不到
- **孙代委托**:子代理在同进程内运行,插件钩子大概率同样作用于子代理自己的 `delegate_task` 调用(级联继承),未官方验证,建议实测评估
- 若想要**更聪明的摘要**(LLM 压缩而非截断):在 `pre_llm_call` 里用 `ctx.llm` 生成摘要是官方支持的口子,但注意它同样受 `hook_callback_timeout`(默认 30s,上限 600s)约束——切勿把 LLM 调用放进 `pre_tool_call`

## 依据(官方文档)

- [Observer Hooks](https://hermes-agent.nousresearch.com/docs/developer-guide/observer-hooks):`pre_tool_call` 可返回 `{"action": "modify", "args": {...}}` 改写工具入参;`pre_llm_call` 载荷含 `conversation_history` / `session_id`
- [Subagent Delegation](https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation):子代理上下文只来自 `goal`/`context`;第一条消息 = context 在前 goal 在后,且压缩时逐字保留;兄弟子代理共享 prompt cache
- [Subagent Lifecycle API](https://hermes-agent.nousresearch.com/docs/developer-guide/subagent-lifecycle-api):goal/context 有大小上限(fail-closed),BUDGET 默认 6000 字符是保守值
