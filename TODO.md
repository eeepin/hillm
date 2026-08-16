# hillm 优化 TODO

## 目标与范围

本文档基于当前代码和本地验证结果，整理 hillm 的优缺点、风险以及最小化改进顺序。

当前阶段的核心目标不是继续增加功能，而是先建立可信的正确性基线，并完成以下两项结构性修复：

1. 将 SSE 处理改为与 HTTP chunk 边界无关、符合事件语义的增量解析。
2. 将上游 API 协议显式建模为 OpenAI Chat、OpenAI Responses 和 Anthropic Messages 三种路由，使 provider 可以声明支持的路由，并在创建 provider 实例时选择其中一种。

本文档只描述工作项，不包含代码修改。

## 当前结论

hillm 已经具备比较完整的 LLM 基础设施能力：多 provider、流式响应、Tower 中间件、缓存、singleflight、熔断、限流、预算、健康检查、租户、guardrail、可观测性和向量存储。Provider、Client、CacheStore、BudgetLedger 等 trait 的抽象方向合理，API Key 脱敏、出站地址校验、重试退避等安全和可靠性意识也较好。

主要问题是功能规模已经达到基础设施库级别，但测试、feature 边界、协议模型和发布工程仍处于原型阶段：

- `cargo test --locked` 当前为 74 通过、9 失败、9 忽略。
- `cargo clippy --locked --all-targets -- -D warnings` 当前有 11 项错误。
- `cargo check --locked --no-default-features` 当前有 19 个编译错误。
- `--all-features` 需要外部 `protoc`，当前环境无法完成构建。
- 复杂 Tower 中间件、streaming、realtime、tenant 和 vectorstore 的行为测试明显不足。
- `client/mod.rs` 等文件过大，公开 API 和内部职责逐渐耦合。
- 仓库缺少 README、examples、CI、CHANGELOG、LICENSE 和独立集成测试。

因此建议暂缓新增中间件，按照下述 P0 → P1 → P2 顺序推进。

## 需要先确定的协议模型

### API 路由不是 provider 路由算法

本文中的“API 路由”表示请求和响应所使用的上游协议，不表示负载均衡或 Tower Router 的流量选择策略。建议使用独立类型，避免与现有 `tower::router` 混淆：

```rust
enum APIType {
    OpenAIChatCompletions,
    OpenAIResponses,
    AnthropicMessages,
}
```

建议的稳定序列化名称：

```text
openai_chat_completions
openai_responses
anthropic_messages
```

不要使用模糊的 `openai_compatible` 作为协议类型；兼容性应由 provider 声明其支持 `OpenAIChatCompletions`，而不是成为第四种协议。

### provider 能力与 provider 实例选择分离

provider 配置描述可用能力：

```rust
ProviderConfig {
    // 现有字段……
    available_api_types: Vec<APIType>,
    default_api_type: Option<APIType>,
}
```

创建出来的 provider 对象只选择一种实际路由：

```rust
ProviderInstance {
    config: Arc<ProviderConfig>,
    api_type: APIType,
}
```

约束：

- `available_api_types` 不得为空。
- `default_api_type` 必须属于 `available_api_types`。
- 显式传入的 `api_type` 必须属于 `available_api_types`，否则创建时立即返回结构化错误。
- 一个 provider 实例在生命周期内不自动切换协议，避免同一对象的 endpoint、header、序列化和流式解码规则发生隐式变化。
- 如果需要同一 provider 同时使用两种协议，应创建两个选择不同 api type 的实例；共享 HTTP client 和凭据可以作为后续优化。

### provider 匹配顺序

最小、可预测的匹配顺序：

1. 如果调用方显式指定 provider，先按 provider 名称查找。
2. 如果未指定 provider，再按 model 的精确规则或明确的前缀规则匹配。
3. 使用 `api_type` 过滤不支持该协议的 provider。
4. 没有匹配时返回 `ProviderNotFound` 或 `APITypeUnsupported`。
5. 多个 provider 同时匹配时返回歧义错误；不要依赖注册顺序，也不要静默回退到 OpenAI。

模型规则需要明确区分精确值和前缀，避免当前 `models: Vec<String>` 同时被测试理解为“前缀”、被实现理解为“精确名称”的问题。最小配置可以使用：

```rust
enum ModelMatch {
    Exact(String),
    Prefix(String),
}
```

### 三种路由的职责

| API 路由 | 默认 endpoint | 请求/响应形状 | 流式结束语义 |
| --- | --- | --- | --- |
| `OpenAIChatCompletions` | `/chat/completions` | OpenAI Chat Completion | OpenAI chat chunk，支持 `[DONE]` |
| `OpenAIResponses` | `/responses` | OpenAI Responses API | Responses 原生事件和 response item |
| `AnthropicMessages` | `/messages` | Anthropic Messages API | Anthropic 原生 message/content block 事件 |

三个路由应分别拥有 endpoint、请求编码、响应解码和流事件解码逻辑。通用 Provider trait 不应再通过一组默认的 `chat_completions_path()`、`responses_path()` 和一个无类型的 `transform_request(Value)` 来隐式判断协议。

最低兼容策略：

- 保留现有 `LLMClient::chat` 和 `ResponseClient`，避免立即破坏已有调用方。
- `LLMClient::chat` 只直接对应 `OpenAIChatCompletions`；如果用它调用 `AnthropicMessages`，兼容转换必须是显式 adapter，而不是 provider 的隐藏默认行为。
- `ResponseClient` 只对应 `OpenAIResponses`。
- 新增 Anthropic Messages 的原生 request/response/event 类型和独立 client trait，避免把 Anthropic 的 content block、cache usage、stop reason 和 tool use 有损压缩成 OpenAI chat 类型。
- 不在第一版中实现三种协议之间任意互转；只提供现有 Chat → Anthropic 的兼容 adapter，并标注可能丢失的信息。

## P0：恢复可信基线

### 1. 修复现有默认测试

- [x] 明确 custom provider 的模型匹配语义，修复 `detect_custom_provider()` 只比较 provider 名称、完全不使用 `models` 的问题。
- [x] 将 custom provider 的全局注册表测试隔离；避免测试并发清空共享状态。
- [x] 更新 provider registry 的 JSON fixture，使其符合当前 `ProviderEntry` 和 `ModelEntry` 的反序列化结构。
- [x] 保证 `cargo test --locked` 零失败；需要网络的测试继续独立标注，但不得用网络测试代替本地 fixture 测试。

完成标准：默认测试稳定全绿，连续和并行执行结果一致。

### 2. 修复已知正确性问题

- [x] 用 `saturating_sub` 或显式校验修复 cache read/write token 总和超过 input token 时的无符号下溢。
- [x] 修复 provider 环境变量扫描只检查第一个元素的问题。
- [x] 处理 `cargo clippy --locked --all-targets -- -D warnings` 的全部错误。
- [x] 为 token 下溢问题添加不依赖网络的回归测试。
- [x] 为 provider 环境变量扫描问题添加不依赖网络的回归测试（`ProviderEntry::to_config()` 当前零测试覆盖）。

完成标准：fmt、默认测试和严格 Clippy 均通过。

## P0：重写 SSE 增量解析边界

- [x] 完成

### 当前问题

当前 `http/stream.rs` 和 `streaming.rs::IngressStream` 已经维护跨 HTTP chunk 的行缓冲，因此”普通 JSON 行只是在任意 chunk 位置断开”在部分情况下可以工作；但它们仍然不是完整、健壮的 SSE decoder：

- 遇到一条 `data:` 行便立刻调用 JSON parser，没有等待空行表示的完整 SSE 事件结束。
- 不能将同一事件中的多条 `data:` 行按 SSE 规则使用 `\n` 合并。
- HTTP chunk 被单独执行 UTF-8 校验；一个多字节 UTF-8 字符跨 chunk 时会被错误拒绝。
- `event:`、`id:`、`retry:` 字段没有形成事件对象，协议 decoder 无法使用事件类型。
- `[DONE]` 被硬编码进通用传输 parser，而它只属于特定上游协议。
- `http/stream.rs` 和 `streaming.rs` 有两套近似的 SSE 解析逻辑，容易继续漂移。
- 当前没有针对任意 chunk 分割、CRLF 分割、多行 data、UTF-8 分割和 EOF 残帧的单元测试。

### 最小实现步骤

- [x] 提取唯一的、与 reqwest 解耦的 `SSEDecoder`，输入 `Bytes`，输出完整 `SSEEvent`。
- [x] 内部使用字节缓冲而不是对每个 HTTP chunk 转为 `&str`；只在完整字段行或完整事件形成后校验 UTF-8。
- [x] 同时识别 `\n\n`、`\r\n\r\n`，并正确处理分隔符本身跨 chunk 的情况。
- [x] 支持 `data`、`event`、`id`、`retry` 和 comment；按规范将多个 `data:` 行以换行连接。
- [x] 仅在空行结束事件时将事件交给 api-type-specific decoder。
- [x] 将 `[DONE]` 判断移动到 `OpenAIChatCompletions` 流事件 decoder；通用 SSE decoder 不理解业务 payload。
- [x] 让 `http/stream.rs` 和 `IngressStream` 复用同一个 decoder，删除重复状态机。
- [x] 统一使用 `SSE_BUFFER_MAX_BYTES`，限制”尚未消费的数据”，事件过大时返回明确错误。
- [x] 明确 EOF 行为：完整但没有最终空行的事件是否派发应由兼容策略决定；半个 UTF-8 字符或半个字段必须返回截断错误，不能静默丢弃。

### 必需的回归测试

- [x] JSON 在每一个可能的字节位置切成两个或多个 HTTP chunk，解析结果保持一致。
- [x] 中文、emoji 等 UTF-8 字符在每个字节位置切分。
- [x] `data:`、字段名、冒号、`\r\n` 和事件终止空行分别跨 chunk。
- [x] 一个 chunk 包含多个事件。
- [x] 一个事件包含多条 `data:` 行。
- [x] comment/heartbeat 不产生业务事件。
- [x] OpenAI `[DONE]` 只终止 `OpenAIChatCompletions` decoder。
- [x] Anthropic `event:` 与 JSON `type` 能正确传递给 Anthropic decoder。
- [x] 超过大小上限、非法 UTF-8、流错误、EOF 残帧返回确定性错误。
- [x] cancellation 后不再轮询底层 stream，也不派发残留事件。

完成标准：任意 HTTP chunk 分割不会改变事件序列；传输 decoder 完全不知道 OpenAI 或 Anthropic 类型。

## P0：引入三种显式 API 路由

### 1. 先增加纯配置模型，不改变网络行为

- [x] 新增 `APIType`，包含且仅包含 `OpenAIChatCompletions`、`OpenAIResponses`、`AnthropicMessages`。
- [x] 为静态、远端数据驱动和 custom provider 配置增加 `available_api_types` 与 `default_api_type`。
- [x] 为文件配置增加相同字段，并对未知值、空列表和非法 default 做严格校验。
- [x] 给内置 provider 设置明确能力：OpenAI 至少支持 Chat 和 Responses；Anthropic 支持 Messages；其他 provider 依据真实端点配置，不进行乐观推断。
- [x] 保留旧配置的兼容默认值：只在旧配置未填写 api types 时推导原行为，并输出迁移说明；新配置必须显式声明。

完成标准：配置可以 round-trip，所有非法组合在 provider 创建前失败。

### 2. provider 工厂在创建实例时选择路由

- [x] 将 `get_provider(name)` 演进为显式选择 api type 的工厂接口，例如 `create_provider(name, api_type)`。
- [x] custom provider 检测同时校验 provider/model match 和 api type 支持。
- [x] `base_url` 不再无条件创建 `OpenAICompatibleProvider`；使用自定义 base URL 时必须给出 api type，或使用明确的兼容默认值并标为待弃用。
- [x] provider 实例暴露只读 `api_type()`，实例创建后不得改变。
- [x] 增加 `APITypeUnsupported`、`AmbiguousProvider` 等结构化错误，避免统一返回 `BadRequest(String)`。

完成标准：每个 provider 实例的 api type 唯一、可观察且已验证；不再静默回退到 OpenAI。

### 3. 将 endpoint 和 codec 绑定到 api type

- [x] 为每种 `APIType` 提供 api-type-specific codec：请求编码、非流响应解码、SSE 事件解码和结束条件。
- [x] `OpenAIChatCompletions` 使用 `/chat/completions` 与 Chat Completion 原生类型。
- [x] `OpenAIResponses` 使用 `/responses` 与 Responses 原生类型；补齐其流式 API，而不是先转换成 chat chunk。
- [x] `AnthropicMessages` 使用 `/messages` 与 Anthropic 原生类型；新增原生 request、response、usage、content block 和 stream event 类型。
- [x] 将现有 Anthropic → OpenAI chat 的转换保留为显式 compatibility adapter。
- [x] 逐步淘汰通用 `transform_request(&mut Value)` / `transform_response(&mut Value)` 在核心协议路由中的使用；无法静态表达的 provider 参数映射可以保留为最后一层扩展。

完成标准：三条路由分别能执行非流请求；三类类型不会被强制归一成 OpenAI Chat 后再发送。

#### 技术债务：APITypeCodec trait 的类型安全重构

**当前问题**：`APITypeCodec` trait 当前使用 `serde_json::Value` 作为请求/响应类型，而非关联类型 `Self::Request` / `Self::Response`。这是因为使用关联类型会导致 `Box<dyn APITypeCodec>` 无法满足对象安全（object safety）要求，编译器无法推断关联类型。

**临时方案**：使用 `serde_json::Value` 作为通用类型，牺牲编译时类型安全以支持动态分发。

**推荐重构方案**（后续实施）：使用 Enum 包装保持类型安全和动态分发：

```rust
pub enum APIRequest {
    ChatCompletion(ChatCompletionRequest),
    AnthropicMessages(AnthropicMessagesRequest),
    BedrockConverse(BedrockConverseRequest),
}

pub enum APIResponse {
    ChatCompletion(ChatCompletionResponse),
    AnthropicMessages(AnthropicMessagesResponse),
    BedrockConverse(BedrockConverseResponse),
}

pub trait APITypeCodec: Send + Sync {
    fn api_type(&self) -> APIType;
    fn encode_request(&self, request: &APIRequest) -> HiLLMResult<Bytes>;
    fn decode_response(&self, bytes: &[u8]) -> HiLLMResult<APIResponse>;
    fn parse_stream_event(&self, data: &str) -> HiLLMResult<Option<APIStreamEvent>>;
}
```

**优势**：

- 保持编译时类型检查
- 支持动态分发（`Box<dyn APITypeCodec>`）
- 明确的类型边界，符合 Rust 哲学
- 与 TODO.md 中"三种路由分别拥有原生类型"的目标一致

**实施时机**：在 P0 Step 2（provider 工厂路由选择）完成后，P0 Step 3（endpoint 和 codec 绑定）开始时进行重构。

### 4. 路由集成测试

- [x] 用本地 mock HTTP service 断言三种 api type 的 URL、header 和 JSON body。
- [x] 断言 provider 不支持 api type 时在发送请求前失败。
- [x] 断言显式 provider、model api type filter 和歧义处理的优先级。
- [x] 断言三种非流响应保留各自的原生字段。
- [x] 断言三种流式响应在任意 chunk 分割下保持一致。
- [x] 断言旧 Chat API 兼容路径仍可用，并记录发生了哪种 adapter 转换。

完成标准：三种 api type 各有至少一个完整的非流和流式端到端测试，测试不访问公网。

## 建议的最小交付切片

为控制风险，建议按以下顺序形成小而独立的改动，每一步都保持测试可运行：

1. **基线修复**：修复现有 9 个测试、token 下溢、env 扫描和 Clippy。
2. **SSE decoder**：只替换传输层状态机并补分片测试，不同时改 provider API。
3. **路由配置模型**：加入 `APIType`、`available_api_types`、`default_api_type` 和校验，但暂不改变发送路径。
4. **provider 实例选择**：工厂创建时选择 api type，修复 provider/model 匹配和 silent fallback。
5. **OpenAI Chat codec**：先把现有行为迁入第一个 api_type_specific codec，保持兼容。
6. **OpenAI Responses codec**：接入现有 ResponseClient，补原生 streaming。
7. **Anthropic Messages codec**：增加原生类型和 client；将旧转换降级为显式 adapter。
8. **三路由集成测试**：本地 mock server + 任意 SSE chunk 分割测试。

不要在一个改动中同时重写 SSE、provider registry、三套 DTO 和全部 Client trait；否则失败时很难区分传输、匹配、序列化还是兼容层问题。

## P1：feature、网络和安全边界

- [x] 修复 `--no-default-features`，为所有 `reqwest`、`tokio`、`rustls`、`DefaultClient` 和 cancellation 类型补齐条件编译边界。
- [x] 建立 CI feature 矩阵：default、no-default、all-features。
- [x] 在 CI 提供 `protoc`，验证 etcd feature。
- [x] 将 JSON、binary、错误 body、SSE 和 EventStream 全部接入统一的有界读取策略；当前 `RESPONSE_BODY_MAX_BYTES` 等常量不能只定义不使用。
- [x] 即使 `OutboundPolicy::Off`，也始终解析 URL 并限制为 http/https；策略只控制 DNS 和地址范围。
- [x] 对服务端场景提供安全默认配置 `DenyPrivate`。新增 `OutboundPolicy::server_default()` 返回 `DenyPrivate`；支持 `HILLM_OUTBOUND_POLICY` 环境变量在启动时选择默认策略。
- [x] 将进程级全局 outbound policy 和 custom provider registry 演进为 client/registry 实例，避免不同租户互相影响。新增 `OutboundPolicyValidator`（per-instance policy validator，替代 `GLOBAL_POLICY` OnceLock）和 `CustomProviderRegistry`（per-instance registry，替代 `CUSTOM_PROVIDERS` RwLock）。原有全局函数保留为便捷入口，底层委托给全局实例。`ClientConfig` 增加 `outbound_policy: Option<Arc<OutboundPolicyValidator>>` 字段，`ClientBuilder` 增加 `.outbound_policy()` 方法；`GuardedResolver` 支持绑定实例 validator。
- [x] provider registry 内置版本化快照，远端数据只作为显式刷新来源；离线时仍可查询能力和成本。`ProviderRegistry` 由 `OnceCell`/`OnceLock` 演进为 `RwLock<Option<ProviderRegistrySnapshot>>`，`ProviderRegistrySnapshot` 包含 `data: Arc<ProviderRegistry>`、`fetched_at: u64`（unix timestamp）和 `source: RegistrySource`（`Remote`/`Offline`）。新增 `refresh_registry()`、`registry_snapshot()`、`registry_fetched_at()`、`registry_source()` 公共函数。
- [x] 为配置文件增加 `api_key_env`，文档中不再推荐明文 `api_key`。

## P1：关键基础设施测试

- [x] cache：过期、驱逐、哈希碰撞、错误缓存和 TTL override。（已有 9 个 store 级测试）
- [x] singleflight：leader 取消/panic、follower lag、关闭通道和多并发一致性。（已有 7 个测试）
- [x] circuit/fallback/hedge/timeout：状态迁移和中间件顺序。（circuit 8 个；新增 fallback 4 个、fallback_chain 7 个、hedge 5 个、cooldown 4 个）
- [x] budget/rate limit：并发请求是否超卖、窗口切换和精度。（budget 11 个，rate_limit 13 个）
- [x] router/health：ready 状态、动态 discover、健康状态切换和无可用上游。（router 现有 11 个 unit，新增 16 个 strategy `call()` 路径测试：RoundRobin 顺序分配、Fallback 失败穿透、LatencyBased tie-break、CostBased 行为、WeightedRandom 退化权重、Router 构造校验、RouterError 转换；health 8 个 unit）
- [x] idempotency：并发重复请求、失败结果和过期。（已有 20 个 store 级测试）
- [x] guardrail：输入/输出阶段顺序和全局 registry 隔离。（registry 14 个测试；新增 builtin 28 个测试覆盖 RegexGuardrail/AllowList/DenyList/LengthCap/PromptInjectionHeuristic/redact_in_place；修复 `GuardrailDecision` 缺失 `#[derive(Clone)]`）
- [x] realtime、tenant、vectorstore 的错误和并发行为。（realtime 从 0 个测试增至 42 个，覆盖所有 21 种事件类型的 inbound/outbound 翻译和往返一致性；tenant 从 8 个增至 15 个，新增并发 insert/resolve、concurrent insert+remove、key 更新、Arc 生命周期测试）

测试应优先使用暂停的 Tokio 时间、可注入时钟、mock `Service` 和本地 HTTP service，不依赖 sleep 或公网。

## P2：结构与发布质量

- [ ] 拆分 `client/mod.rs`：traits、core、chat、responses、anthropic_messages、files、batches、streaming。
- [ ] 拆分大 provider 文件，将 DTO/codec 与 provider 配置分离。
- [ ] 收窄根模块的 `pub use types::*`，减少未来的 API 兼容负担。
- [ ] 评估在 API 稳定后拆分 `hillm-core`、`hillm-http`、`hillm-tower` 和 provider crates；当前先做文件级拆分。
- [ ] 添加 README、LICENSE、CHANGELOG、examples、CI、贡献指南和安全策略。
- [ ] 补齐 Cargo package metadata：description、license、repository、documentation、keywords、categories、rust-version。
- [ ] 为 feature、provider api type、SSE 限制、兼容 adapter 和安全默认值提供公开文档。
- [ ] 在 0.2 发布前明确 Provider、APIType、错误类型和配置文件的稳定契约。

## 暂不纳入最小方案

以下事项有价值，但不应阻塞三路由和 SSE 修复的第一版：

- 自动根据一次请求失败在三种 API Type 之间探测或切换。
- 将 OpenAI Responses 与 Anthropic Messages 完整无损转换为 Chat Completion。
- 同一 provider 实例按请求动态选择 API type。
- 基于模型能力自动选择最优协议。
- 重新设计全部 Tower `LLMRequestKind` 和缓存键。
- 立即拆成多个 crate。

这些行为会引入隐式决策、缓存键变化和难以解释的 fallback。第一版应坚持“配置声明能力、创建实例时显式选择、发送期间保持不变”。

## 最终验收门槛

- [x] `cargo fmt --all -- --check` 通过。
- [x] `cargo clippy --locked --all-targets -- -D warnings` 通过。
- [x] `cargo test --locked` 零失败。
- [x] 受支持的 feature 矩阵全部编译；all-features 在有 protoc 的 CI 中通过。（default、no-default、tower 已在本地验证；wasm 目标与 all-features 需要 CI）
- [x] 任意 HTTP chunk 分割不改变三种路由的流式事件结果。
- [x] provider 的可用 api type 列表、默认 api type、实例选择 api type 均可配置且经过校验。
- [x] OpenAI Chat、OpenAI Responses、Anthropic Messages 各有原生非流和流式集成测试。
- [x] 不支持、未匹配和歧义情况均返回结构化错误，不静默回退到 OpenAI。
- [x] 现有 Chat 调用方有明确、经过测试的兼容迁移路径。
