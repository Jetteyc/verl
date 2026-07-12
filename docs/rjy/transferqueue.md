# TransferQueue 仓库全景解读 —— 及其在 verl 中的应用

---

## 一、这个仓库是什么？

TransferQueue 是一个**面向后训练（post-training）场景的异步流式数据管理模块**。它解决的核心问题是：

> 在大规模分布式 RLHF/PPO 训练中，数据的生产者和消费者之间如何高效、解耦地传递 tensor 数据？

它不是另一个训练框架，而是训练框架的**数据基础设施层**——你可以把它理解为分布式训练的"数据网关"。

---

## 二、为什么需要它？—— verl 的痛点

### 2.1 verl 原来的架构

verl 采用**单控制器模式**：`RayPPOTrainer` 作为中央调度器，所有阶段（generate → compute_log_prob → compute_reward → compute_advantage → update_actor）的 `DataProto` 对象都必须经过它中转。

```
Rollout Worker ──DataProto──→ RayPPOTrainer ──DataProto──→ Ref Worker
                                    ↕
Actor Worker ←──DataProto──         ↓          ──DataProto──→ Reward Worker
```

### 2.2 瓶颈在哪？

当规模从 8 卡扩展到 128×H100 时：

- **单点带宽瓶颈**：所有 tensor（prompt_ids, response_ids, log_prob, reward …）都要经控制器序列化/反序列化，控制器成为数据传输的漏斗。
- **内存压力**：控制器需要同时持有所有中间数据副本，内存利用率极差。
- **同步屏障**：各阶段通过控制器耦合，一个阶段慢会拖慢整个流水线。

这不是 Ray 的问题——Ray 作为分布式运行时本身是高效的。**问题在于 verl 把所有数据都路由到单点的架构设计**。

### 2.3 TransferQueue 的解法

核心思想：**数据不经控制器中转，生产者直写存储，消费者直读存储；控制器只管元数据**。

```
Rollout Worker ──write──→ [Distributed Storage] ←──read── Ref Worker
                                ↕
                         Controller (只管元数据)
                                ↕
Actor Worker ←──read──  [Distributed Storage]  ──read──→ Reward Worker
```

这个架构转变在 verl 集成中带来了 **49.1% 的端到端性能提升**（128×H100，多模态 post-training）。

---

## 三、TransferQueue 的架构

TransferQueue 分为**控制平面**和**数据平面**，以及面向用户的 **API 层**。

### 3.1 控制平面 —— TransferQueueController

> 代码：[controller.py](transfer_queue/controller.py)

`TransferQueueController` 是一个 Ray Actor，但它**不搬运任何 tensor 数据**。它的职责是：

**① 元数据追踪**：记录每个样本的每个字段（field）是否已被生产、已被哪个任务消费。

每个样本在控制器中用 `global_index` 唯一标识。字段级的生产状态用位图跟踪：

```python
# controller.py 中的 FieldMeta
class FieldMeta:
    global_indexes: set[int]     # 该字段覆盖了哪些样本
    dtype: Any                   # 数据类型
    shape: tuple                 # 形状（nested tensor 为 None）
    is_nested: bool              # 是否变长 tensor
    per_sample_shapes: dict      # 变长 tensor 每样本的 shape
```

**② 消费历史追踪**：不同任务（如 `compute_log_prob`、`update_actor`）可能需要同一份数据的不同字段，控制器为每个任务独立维护消费进度，互不干扰。

**③ 数据分片管理**：通过 `PartitionIndexManager` 管理 partition（逻辑数据集），支持索引的分配与回收复用。

**④ 采样调度**：通过可插拔的 `Sampler` 决定哪些就绪样本应该被哪些消费者拿走。

控制器通过 **ZMQ** 与客户端通信（而非 Ray remote call），保证元数据交互的低延迟。

### 3.2 数据平面 —— StorageManager + StorageBackend

> 代码：[storage/](transfer_queue/storage/)

数据平面负责实际的 tensor 存储和传输。核心抽象是 `StorageManager`：

```python
class StorageManager:
    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None
    async def get_data(self, metadata: BatchMeta) -> TensorDict
    async def clear_data(self, metadata: BatchMeta) -> None
```

**可插拔存储后端**，目前已支持：

| 后端 | 说明 | 状态 |
|------|------|------|
| **SimpleStorage** | CPU 内存存储，默认后端 | 稳定 |
| **Yuanrong** | 华为原生数据系统，支持 HBM/DRAM/SSD 分层 | Beta |
| **MooncakeStore** | 高性能 KV 存储，支持 RDMA | Beta |
| **RayRDT** | Ray 直接传输 | Alpha |

#### SimpleStorageUnit 的数据结构

> 代码：[simple_storage.py](transfer_queue/storage/simple_storage.py)

`SimpleStorageUnit` 是默认的存储后端，采用**二维数据结构**：

```
              field1        field2        field3
sample_0   [prompt_ids]  [response_ids]   [log_prob]
sample_1   [prompt_ids]  [response_ids]   [log_prob]
sample_2   [prompt_ids]                    [log_prob]   ← field2 还没写入
```

- 每行 = 一个训练样本（由 `global_index` 标识）
- 每列 = 一个数据字段（如 prompt_ids, response_ids 等）
- 支持**字段级并发读写**：不同任务可以同时写不同字段，或读已就绪的字段

底层用 `dict[str, dict]` 实现，`O(1)` 的读写和删除：

```python
# StorageUnitData 内部结构
field_data = {
    "field_name1": {global_index_0: tensor1, global_index_3: tensor2},
    "field_name2": {global_index_0: tensor3, global_index_3: tensor4},
}
```

每个 `SimpleStorageUnit` 是一个 Ray Actor，可部署在不同节点上，通过 ZMQ 提供服务。由 `AsyncSimpleStorageManager` 协调多个 storage unit 的数据路由。

### 3.3 元数据 —— BatchMeta 与 KVBatchMeta

> 代码：[metadata.py](transfer_queue/metadata.py)

`BatchMeta` 是控制器和客户端之间的"通行证"，描述一批数据的位置和状态：

```python
class BatchMeta:
    global_indexes: list[int]       # 样本的全局索引
    partition_ids: list[str]        # 所属分区
    production_status: list[int]    # 每样本的生产位图
    field_schema: dict              # 字段级元信息 {name: {dtype, shape, is_nested}}
    consumption_status: dict        # 每任务的消费记录
    custom_meta: dict               # 用户自定义元数据
```

`KVBatchMeta` 是 KV 接口专用的子集，增加了 `keys` 字段，更贴合 Redis 风格的使用方式。

### 3.4 采样器系统 —— Sampler

> 代码：[sampler/](transfer_queue/sampler/)

采样器决定**哪些就绪样本被哪些消费者取走**，是控制平面的核心调度逻辑。

```
BaseSampler (抽象基类)
├── SequentialSampler      # 默认：顺序取，FIFO
├── RankAwareSampler       # 分布式：同 DP 组的 rank 取相同数据
├── SeqlenBalancedSampler  # 按 token 长度均衡分配，减少 padding 浪费
└── GRPOGroupNSampler      # GRPO 算法：同一 prompt 的 N 个 response 必须同时就绪
```

`BaseSampler` 的核心接口：

```python
def sample(self, ready_indexes, batch_size, **kwargs) -> tuple[list[int], list[int]]:
    """返回 (采样索引列表, 标记为已消费的索引列表)"""
```

---

## 四、用户 API 层

> 代码：[interface.py](transfer_queue/interface.py)、[\_\_init\_\_.py](transfer_queue/__init__.py)

TransferQueue 提供三个层级的 API，从简到繁：

### 4.1 KV 接口（高层，推荐入门）

Redis 风格的 Put/Get/List/Clear 语义，零学习成本：

```python
import transfer_queue as tq

# 初始化系统
tq.init()

# 写入数据 —— 生产者（如 Rollout Worker）
tq.kv_put(key="sample_0", partition_id="train", fields=TensorDict({
    "prompt_ids": prompt_tensor,
    "response_ids": response_tensor,
}))

# 读取数据 —— 消费者（如 Ref Worker）
data = tq.kv_batch_get(keys=["sample_0"], partition_id="train", fields=["prompt_ids", "response_ids"])

# 查看状态
tq.kv_list(partition_id="train")

# 清理
tq.kv_clear(keys=["sample_0"], partition_id="train")
```

**关键特性**：
- **字段级粒度**：可以只读取/更新特定字段，无需整行操作
- **分区隔离**：不同 partition 互不干扰
- **元数据标签**：轻量级状态追踪
- **异步版本**：所有方法都有 `async_` 前缀的协程版本

### 4.2 StreamingDataLoader（高层，流式训练场景）

> 代码：[streaming_dataloader.py](transfer_queue/dataloader/streaming_dataloader.py)、[streaming_dataset.py](transfer_queue/dataloader/streaming_dataset.py)

PyTorch `DataLoader` 的 drop-in replacement，支持完全流式的生产-消费管道：

```python
dataset = StreamingDataset(
    config=config,
    batch_size=32,
    micro_batch_size=4,
    data_fields=["prompt_ids", "response_ids", "log_prob", "reward"],
    partition_id="train",
    task_name="update_actor",
    dp_rank=0,
)

dataloader = StreamingDataLoader(dataset, num_workers=2, prefetch_factor=2)

for batch, batch_meta in dataloader:
    # batch: TensorDict，包含请求的字段
    # batch_meta: 元数据，用于后续写回
    loss = model(batch)
    loss.backward()
    
    # 写回结果
    tq.kv_batch_put(keys=batch_meta.keys, partition_id="train", fields=output)
```

**两种模式**：
- **流式模式**（默认）：无限流，持续轮询新数据，适合在线训练
- **有限模式**：`should_check_consumption_status=True`，消费完 partition 后自动停止

**注意**：`StreamingDataset.__iter__` 在 DataLoader 的 worker 子进程中创建独立的 `TransferQueueClient`（通过 ZMQ 直连，不依赖 Ray），避免 fork 安全问题。

### 4.3 原生客户端 API（底层，最大灵活度）

> 代码：[client.py](transfer_queue/client.py)

直接操作 `TransferQueueClient`，获取元数据 → 读取数据 → 写入数据 → 清理数据的全流程：

```python
client = tq.get_client()

# 1. 获取就绪数据的元数据
meta = client.get_meta(partition_id="train", task_name="update_actor", batch_size=32)

# 2. 从存储读取数据
data = client.get_data(meta)

# 3. 计算并写回
client.put_data(output_data, output_meta)

# 4. 标记消费完成
client.clear_data(meta)
```

---

## 五、在 verl 中的集成方式

### 5.1 集成历史

| 时间 | 里程碑 |
|------|--------|
| 2025.07 | 在 verl 社区发起 [RFC#2662](https://github.com/volcengine/verl/discussions/2662) |
| 2025.10 | 早期集成 [PR#3649](https://github.com/volcengine/verl/pull/3649) |
| 2025.12 | 64 节点 1024 卡 DAPO 算法大规模测试 |
| 2026.04 | **官方正式集成** [verl/pull/5401](https://github.com/verl-project/verl/pull/5401) |

### 5.2 数据流对比

**集成前**（所有数据经 RayPPOTrainer 中转）：

```
Generate → [DataProto] → RayPPOTrainer → [DataProto] → Ref Worker
                                    ↓
                              [DataProto] → Reward Worker
                                    ↓
                              [DataProto] → Actor Worker
```

**集成后**（数据直传，控制器只管元数据）：

```
Generate → kv_put → [TQ Storage] ← kv_batch_get ← Ref Worker
                          ↕
                    TQ Controller (side-controller，只管元数据)
                          ↕
                    [TQ Storage] ← kv_batch_get ← Reward/Actor Worker
```

### 5.3 典型 PPO 流程中的数据流

以 verl 的 PPO 训练为例，展示 TransferQueue 如何编排各阶段：

```python
import transfer_queue as tq

# ====== 初始化（只需在 driver 进程调用一次）======
tq.init()

# ====== Step 1: Rollout（生成 response）======
# Rollout Worker 作为生产者
response_data = TensorDict({"response_ids": generated_ids})
kv_meta = tq.kv_batch_put(keys=sample_keys, partition_id="ppo_batch", fields=response_data)

# ====== Step 2: Ref Model（计算参考 log_prob）======
# Ref Worker 作为消费者，直接从存储读取
data = tq.kv_batch_get(keys=sample_keys, partition_id="ppo_batch", 
                        fields=["prompt_ids", "response_ids"])
ref_output = TensorDict({"ref_log_prob": ref_log_prob})
tq.kv_batch_put(keys=sample_keys, partition_id="ppo_batch", fields=ref_output)

# ====== Step 3: Reward（计算奖励）======
data = tq.kv_batch_get(keys=sample_keys, partition_id="ppo_batch",
                        fields=["response_ids"])
reward_output = TensorDict({"reward": rewards})
tq.kv_batch_put(keys=sample_keys, partition_id="ppo_batch", fields=reward_output)

# ====== Step 4: Actor Update（更新策略）======
# 消费所有字段
data = tq.kv_batch_get(keys=sample_keys, partition_id="ppo_batch",
                        fields=["prompt_ids", "response_ids", "old_log_prob", 
                                "ref_log_prob", "reward", "advantage"])
# ... 训练 ...
# 清理本批次数据
tq.kv_clear(keys=sample_keys, partition_id="ppo_batch")
```

**关键点**：各阶段之间**没有数据中转**，每个 worker 直接与存储交互。控制器在幕后确保"字段就绪后才可消费"。

### 5.4 完整的 recipe 示例

仓库提供了 verl 风格的完整 demo：[recipe/simple_use_case/single_controller_demo.py](recipe/simple_use_case/single_controller_demo.py)，模拟了 PPO 训练中的 generate → compute_log_prob → compute_ref_log_prob → compute_reward → compute_advantage → update_actor 全流程。

---

## 六、核心概念速查表

| 概念 | 说明 | 对应代码 |
|------|------|----------|
| `TransferQueueController` | 控制平面，追踪元数据，调度采样 | [controller.py](transfer_queue/controller.py) |
| `BatchMeta` | 元数据"通行证"，描述数据的位置和状态 | [metadata.py](transfer_queue/metadata.py) |
| `StorageManager` | 数据平面抽象，定义 put/get/clear 接口 | [storage/managers/](transfer_queue/storage/managers/) |
| `SimpleStorageUnit` | 默认存储后端，CPU 内存，2D 数据结构 | [simple_storage.py](transfer_queue/storage/simple_storage.py) |
| `BaseSampler` | 采样器基类，决定谁消费什么数据 | [sampler/base.py](transfer_queue/sampler/base.py) |
| `RankAwareSampler` | 分布式采样器，保证同 DP 组取相同数据 | [rank_aware_sampler.py](transfer_queue/sampler/rank_aware_sampler.py) |
| `StreamingDataLoader` | PyTorch 风格流式数据加载器 | [streaming_dataloader.py](transfer_queue/dataloader/streaming_dataloader.py) |
| `KVBatchMeta` | KV 接口的元数据，带 key 字段 | [metadata.py](transfer_queue/metadata.py) |
| `PartitionIndexManager` | 分区索引管理，支持分配/回收/复用 | [controller.py](transfer_queue/controller.py#L56) |
| ZMQ 通信层 | 控制器与客户端之间的低延迟消息通道 | [zmq_utils.py](transfer_queue/utils/zmq_utils.py) |

---

## 七、快速上手路径

1. **安装**：`pip install TransferQueue`
2. **看教程**：[tutorial/basic.ipynb](tutorial/basic.ipynb) → [tutorial/02_kv_interface.py](tutorial/02_kv_interface.py)
3. **看完整 demo**：[recipe/simple_use_case/single_controller_demo.py](recipe/simple_use_case/single_controller_demo.py)
4. **深入控制器**：[tutorial/04_understanding_controller.py](tutorial/04_understanding_controller.py)
5. **自定义采样器**：[tutorial/05_custom_sampler.py](tutorial/05_custom_sampler.py)
6. **流式训练**：[tutorial/06_streaming_dataloader.py](tutorial/06_streaming_dataloader.py)

---

*这份文档覆盖了仓库的全貌。如果你想深入某个模块，建议按以下顺序阅读指南：*

- [架构概述](4-architecture-overview) — 整体架构的系统性讲解
- [控制器元数据追踪](5-controller-metadata-tracking) — 控制平面如何工作
- [存储管理器抽象](8-storage-manager-abstraction) — 数据平面的设计
- [StreamingDataLoader API](12-streamingdataloader-api) — verl 集成的核心接口
- [原生客户端 API](13-native-client-api-low-level) — 底层操作的全景