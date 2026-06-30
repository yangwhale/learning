# 5. Training Loop（训练循环）

## 概述

将前面构建的所有组件组合在一起：数据加载、模型、优化器。

---

## 5.1 Data Loader（数据加载）

### 数据格式

Tokenized 数据是一个单一的 token 序列 $x = (x_1, x_2, \ldots, x_n)$。

Data loader 将这个长序列转换为 batch 流。每个 batch 包含：
- $B$ 个长度为 $m$ 的输入序列
- 对应的 next-token targets

**示例：** 当 $B = 1$, $m = 3$ 时：

```
输入:  [x_2, x_3, x_4]
目标:  [x_3, x_4, x_5]
```

即 targets 是 inputs 向右移一位。

---

### Problem: 实现数据加载 (2 points)

**函数参数：**
- `x` — numpy array，token ID 的整数数组（一维长序列）
- `batch_size` — batch 大小 $B$
- `context_length` — 上下文长度 $m$
- `device` — PyTorch device string，如 `'cpu'` 或 `'cuda:0'`

**返回：**
- 一对张量 `(inputs, targets)`，shape 均为 `(batch_size, context_length)`

**测试：**
- Adapter: `adapters.run_get_batch`
- 测试命令：

```bash
uv run pytest -k test_get_batch
```

---

> ### Low-Resource Tip: CPU 或 Apple Silicon 上的数据加载
>
> - **CPU**: 使用 `'cpu'` 作为 device
> - **Apple Silicon** (M1/M2/M3/M4 芯片): 使用 `'mps'` 作为 device
>
> **MPS 相关资源：**
> - [PyTorch MPS Backend](https://docs.pytorch.org/docs/stable/mps.html)
> - [PyTorch MPS Notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
> - [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

---

### Memory-Mapped 数据加载

对于大型数据集，无法将整个数据集加载到内存中。使用 **memory-mapped** 模式可以按需加载数据的各个部分。

**实现方式：**
- 使用 `np.memmap` 创建 memory-mapped 数组
- 或使用 `np.load` 的 `mmap_mode='r'` 参数

**重要提醒：**
- **训练时从 numpy 数组采样时，务必使用 memory-mapped 模式加载**
- 确保指定正确的 `dtype`（例如 `np.uint16` 或 `np.int32`）

---

## 5.2 Checkpointing（模型检查点）

### 背景

保存模型以便恢复训练是非常重要的，因为训练过程可能因多种原因中断：
- Job 超时
- 机器故障
- 需要调整超参数后继续训练

### Checkpoint 内容

一个 checkpoint 至少应包含：
- **模型权重** — 所有可学习参数的当前值
- **优化器状态** — 如 AdamW 的一阶矩 ($m$) 和二阶矩 ($v$) 估计
- **迭代号** — 当前训练到第几步

### PyTorch 序列化工具

| 功能 | API |
|------|-----|
| 获取模型参数字典 | `nn.Module.state_dict()` |
| 加载模型参数 | `nn.Module.load_state_dict(state_dict)` |
| 获取优化器状态字典 | `torch.optim.Optimizer.state_dict()` |
| 加载优化器状态 | `torch.optim.Optimizer.load_state_dict(state_dict)` |
| 保存 Python 对象 | `torch.save(obj, dest)` |
| 加载 Python 对象 | `torch.load(src)` |

---

### Problem: 实现模型 Checkpointing (1 point)

**保存 checkpoint：**

```python
def save_checkpoint(model, optimizer, iteration, out):
    """
    保存训练 checkpoint。

    参数:
        model: nn.Module           模型
        optimizer: torch.optim.Optimizer  优化器
        iteration: int             当前迭代号
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  输出路径/文件
    """
```

**加载 checkpoint：**

```python
def load_checkpoint(src, model, optimizer):
    """
    加载训练 checkpoint。

    参数:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  checkpoint 路径/文件
        model: nn.Module           模型（会被原地更新）
        optimizer: torch.optim.Optimizer  优化器（会被原地更新）

    返回:
        int  保存时的迭代号
    """
```

**测试：**
- Adapter: `adapters.run_save_checkpoint` + `adapters.run_load_checkpoint`
- 测试命令：

```bash
uv run pytest -k test_checkpointing
```

---

## 5.3 Training Loop（训练循环）

### Problem: 组合在一起 (4 points)

编写一个完整的训练脚本，将前面实现的所有组件串联起来。

**建议支持的功能：**

1. **超参数配置** — 能够配置和控制各种模型/优化器超参数（如 `d_model`, `num_layers`, `learning_rate`, `batch_size` 等）

2. **高效数据加载** — 使用 `np.memmap` 高效加载大型训练和验证数据集，不将整个数据集读入内存

3. **Checkpoint 序列化** — 支持将 checkpoint 保存到用户指定的路径，以便中断后恢复训练

4. **性能日志记录** — 定期记录训练和验证性能，支持：
   - 输出到控制台
   - 和/或发送到外部服务（如 [Weights and Biases](https://wandb.ai)）

> **脚注 9：** Weights and Biases (wandb.ai) 是一个常用的实验跟踪平台，可以用于记录训练指标、可视化学习曲线等。

**Deliverable**: 可运行的训练脚本。
