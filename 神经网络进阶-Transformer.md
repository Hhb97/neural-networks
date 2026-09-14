# 神经网络进阶：Transformer（当前主流方向）

> 承接《[从机器学习到神经网络：零基础入门](从机器学习到神经网络-零基础入门.md)》第 15 节"第三步"的最后一个方向。
> 也承接《[RNN 与 LSTM](神经网络进阶-RNN与LSTM.md)》结尾提到的那句话——**"现在多被 Transformer 替代"**。这一篇讲清楚：替代的原因是什么，Transformer 又是怎么工作的。
> ChatGPT、GPT、Claude 这些大模型，底层结构都是 Transformer。

---

## 目录

1. [RNN 的两个瓶颈，Transformer 怎么解决](#1-rnn-的两个瓶颈transformer-怎么解决)
2. [Self-Attention：每个词都去"问"所有其他词](#2-self-attention每个词都去问所有其他词)
3. [Query / Key / Value 三兄弟](#3-query--key--value-三兄弟)
4. [手算一遍 Self-Attention](#4-手算一遍-self-attention)
5. [多头注意力：多找几个角度](#5-多头注意力多找几个角度)
6. [位置编码：没有顺序怎么办](#6-位置编码没有顺序怎么办)
7. [一个 Transformer Block 长什么样](#7-一个-transformer-block-长什么样)
8. [Encoder-only / Decoder-only / Encoder-Decoder](#8-encoder-only--decoder-only--encoder-decoder)
9. [完整代码：手写一个最小的 Self-Attention](#9-完整代码手写一个最小的-self-attention)
10. [黑话词典](#10-黑话词典)
11. [检查清单](#11-检查清单)

---

## 1. RNN 的两个瓶颈，Transformer 怎么解决

上一篇讲到 RNN/LSTM 的两个硬伤：

**瓶颈一：只能一步一步算，没法并行。** 算 `h5` 必须先算完 `h1 h2 h3 h4`，训练速度被严重拖慢，GPU 的并行能力用不上。

**瓶颈二：信息要一步步"传话"才能到远处。** 就算 LSTM 缓解了梯度消失，第 1 个词的信息传到第 100 个词，中间还是要经过 99 次传递，多少会衰减、失真。

**Transformer 的解法很直接**：**别一步步传了，让每个词直接和序列里所有其他词"对话"一次，一步到位。** 这样：

- 所有词可以**同时**计算（大幅并行，训练快）
- 任意两个位置之间**只隔一步**（第 1 个词和第 100 个词直接建立联系，没有衰减）

这个"直接对话"的机制，就叫 **Self-Attention（自注意力）**。

---

## 2. Self-Attention：每个词都去"问"所有其他词

### 直觉

一句话："猫躺在垫子上，因为**它**很软。"

"它"指的是猫还是垫子？人类靠上下文判断——"软"更可能形容垫子。Self-Attention 做的就是这件事：**"它"这个词会去看句子里其他所有词，根据相关程度，决定应该重点参考谁来理解自己。**

### 用"关注度"重新组合信息

Self-Attention 给每个词分配一组"关注度分数"（对句子里所有词，包括自己），分数高的词对当前词的意思贡献更大。最终每个词的新表示，是**所有词的加权平均**，权重就是这组关注度。

```
"它" 的新表示 = 0.05×"猫" + 0.70×"垫子" + 0.02×"躺" + ... 
                              ↑ 关注度最高，说明模型判断"它"主要指垫子
```

---

## 3. Query / Key / Value 三兄弟

具体怎么算出"关注度分数"？每个词会生成三个向量：

| 向量 | 类比 | 作用 |
|---|---|---|
| **Query（问）** | "我想找跟我相关的信息，我的问题是……" | 代表"当前词在找什么" |
| **Key（钥匙/标签）** | "如果有人来找我，看这个标签知道我是谁" | 代表"这个词能提供什么" |
| **Value（值/内容）** | "如果你选中了我，这是我真正要给你的内容" | 真正被加权求和的信息 |

三个向量都是**同一个词向量**分别乘上三个不同的权重矩阵得到的：

```
Q = x · Wq
K = x · Wk
V = x · Wv
```

**关注度分数的计算**：拿"我的 Query"去跟"所有词的 Key"做点积（点积越大，说明越匹配）：

```
注意力分数 = Q · K^T / √d_k        ← 除以 √d_k 是为了防止点积数值过大，让后面的 softmax 更稳定
注意力权重 = softmax(注意力分数)    ← 复用第二篇的 Softmax，把分数变成加起来等于 1 的权重
输出 = 注意力权重 · V               ← 按权重把所有词的 Value 加权求和
```

---

## 4. 手算一遍 Self-Attention

用一个极简的例子：句子只有 2 个词，每个词的向量只有 2 维（真实模型是几百上千维，这里为了手算缩小）。

```
词 A 的 Query: qA = [1, 0]
词 A 的 Key:   kA = [1, 0]      词 B 的 Key:   kB = [0, 1]
词 A 的 Value: vA = [1, 2]      词 B 的 Value: vB = [3, 4]
```

**第 1 步：算词 A 的注意力分数（Q 与每个 K 做点积）**

```
分数(A,A) = qA · kA = 1×1 + 0×0 = 1
分数(A,B) = qA · kB = 1×0 + 0×1 = 0
```

（这里维度小，先跳过除以 √d_k 这一步，直接进入 softmax）

**第 2 步：Softmax 归一化（复用第二篇第 2 节的公式）**

```
e^1 = 2.718,  e^0 = 1.000
总和 = 3.718

权重(A,A) = 2.718 / 3.718 = 0.731
权重(A,B) = 1.000 / 3.718 = 0.269
```

**第 3 步：按权重加权求和 Value**

```
词 A 的新表示 = 0.731 × vA + 0.269 × vB
             = 0.731 × [1, 2] + 0.269 × [3, 4]
             = [0.731, 1.462] + [0.807, 1.076]
             = [1.538, 2.538]
```

**读法**：词 A 的 Query 跟自己的 Key 更匹配（点积 1 > 0），所以词 A 的新表示里，**73% 来自自己，27% 参考了词 B**——这就是"注意力"的字面意思：模型自己算出该给每个词分配多少关注度。

---

## 5. 多头注意力：多找几个角度

一次 Self-Attention 只能学到"一种"相关性模式（比如只学会了"指代关系"）。但语言里同时存在很多种关系：语法关系、语义关系、指代关系……

**多头注意力（Multi-Head Attention）** 的做法：**把 Q、K、V 切成好几份（比如 8 份），每一份独立算一次 Self-Attention（一个"头"），最后把 8 个头的结果拼起来，再过一个线性层融合。**

```
头1 可能专门学会了"主谓关系"
头2 可能专门学会了"指代关系"
头3 可能专门学会了"相邻词关系"
...
最后拼接 8 个头的输出 → 融合成一个更丰富的表示
```

类比：一个头是"一个人读句子，只从一个角度理解"；多头是"8 个人同时读，每人关注点不同，最后综合 8 个人的理解"。

---

## 6. 位置编码：没有顺序怎么办

Self-Attention 有个明显副作用：它计算的是"所有词两两之间的关系"，**跟词在句子里的先后顺序完全无关**——打乱词的顺序，attention 的计算结果结构上不变。但"猫追狗"和"狗追猫"意思完全不同！

**解法：位置编码（Positional Encoding）**。在每个词的向量上，**额外加一个专门编码"这是第几个位置"的向量**：

```
词的最终输入 = 词本身的向量 + 位置编码向量
```

位置编码常用不同频率的正弦/余弦函数生成（原始论文的做法），也可以让模型自己学（可学习位置编码）。**只要保证每个位置的编码都不一样，模型就能顺带学到"顺序"这个信息**，弥补 Self-Attention 本身对顺序不敏感的缺陷。

---

## 7. 一个 Transformer Block 长什么样

真正的 Transformer 不是只有一层 Self-Attention，而是把下面这个"块"堆叠很多层（原始论文堆了 6~12 层，大模型堆几十上百层）：

```
输入
  │
  ├──────────────┐
  ▼              │
Multi-Head        │  ← 残差连接（回忆 CNN 篇里 ResNet 的做法）
Self-Attention     │
  │              │
  ▼              │
  + ◄────────────┘
  │
LayerNorm（归一化，帮助训练稳定）
  │
  ├──────────────┐
  ▼              │
前馈网络 FFN       │  ← 就是两层全连接 + 激活函数（第一篇讲过的最基础结构）
（Linear→ReLU→Linear）  │
  │              │
  ▼              │
  + ◄────────────┘
  │
LayerNorm
  │
  ▼
输出（送给下一个 Transformer Block）
```

**看到没有**：这个"块"里其实全是前几篇讲过的零件——残差连接（CNN 篇）、全连接 + ReLU（入门笔记）、Softmax（第二篇）。Transformer 真正"新"的东西只有 **Self-Attention 和位置编码**，其余都是老朋友重新组合。

---

## 8. Encoder-only / Decoder-only / Encoder-Decoder

原始 Transformer 论文是 Encoder-Decoder 结构（做机器翻译），后来分化出三条路线：

| 结构 | 代表模型 | 适合任务 |
|---|---|---|
| **Encoder-only** | BERT | 理解类任务：分类、抽取式问答（一次看到整句话） |
| **Decoder-only** | GPT 系列、Claude | 生成类任务：只能看到"当前词之前"的内容，逐词往后生成（这叫**自回归**） |
| **Encoder-Decoder** | 原始 Transformer、T5 | 输入一段、输出另一段的任务，如翻译、摘要 |

**当前的大语言模型（包括 GPT、Claude）基本都是 Decoder-only**：靠"预测下一个词"这一个简单目标训练，靠堆叠层数和参数量把这件事做到极致。

---

## 9. 完整代码：手写一个最小的 Self-Attention

不依赖任何"注意力层"封装，把第 3、4 节的公式直接写成代码，方便对照理解：

```python
import torch
import torch.nn.functional as F

def self_attention(x, Wq, Wk, Wv):
    """
    x:  (序列长度, 词向量维度)
    Wq, Wk, Wv: (词向量维度, 内部维度) 的可学习权重矩阵
    """
    Q = x @ Wq              # (seq_len, d_k)
    K = x @ Wk              # (seq_len, d_k)
    V = x @ Wv              # (seq_len, d_v)

    d_k = K.shape[-1]
    scores = (Q @ K.T) / (d_k ** 0.5)     # 第3节：Q·K^T / √d_k
    weights = F.softmax(scores, dim=-1)   # 第2篇的 Softmax，按行归一化
    output = weights @ V                  # 按权重加权求和 Value

    return output, weights

torch.manual_seed(0)
seq_len, d_model, d_k = 4, 8, 4          # 4 个词，每个词 8 维，内部维度 4

x  = torch.randn(seq_len, d_model)        # 4 个词的输入向量
Wq = torch.randn(d_model, d_k)
Wk = torch.randn(d_model, d_k)
Wv = torch.randn(d_model, d_k)

output, weights = self_attention(x, Wq, Wk, Wv)
print("每个词新的表示，形状:", output.shape)      # (4, 4)
print("注意力权重矩阵（每行加起来=1）:\n", weights)

# ============ 对照：直接用 PyTorch 内置的多头注意力 ============
import torch.nn as nn
mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
x_batched = x.unsqueeze(0)                # 加上 batch 维度: (1, seq_len, d_model)
mha_out, mha_weights = mha(x_batched, x_batched, x_batched)   # Self-Attention: Q=K=V=x
print("\nPyTorch 内置多头注意力输出形状:", mha_out.shape)
```

**逐行对照本篇知识点**：`Q = x @ Wq` 等三行是第 3 节的 Q/K/V 生成；`scores = Q@K.T / √d_k` 和 `weights = softmax(scores)` 就是第 3、4 节手算的两步；`nn.MultiheadAttention` 内部把第 5 节的"多头拆分再拼接"也封装好了。

---

## 10. 黑话词典

| 术语 | 人话解释 |
|---|---|
| **Self-Attention** | 序列里每个位置都直接和所有其他位置计算相关性，一步到位地交换信息 |
| **Query / Key / Value** | 分别代表"当前词在找什么"“别人能提供什么"“真正被加权求和的内容” |
| **多头注意力 Multi-Head Attention** | 把 Q/K/V 切成多份并行算多次 attention，从不同角度捕捉关系 |
| **位置编码 Positional Encoding** | 额外加到词向量上、专门编码"这是第几个位置"的向量 |
| **LayerNorm** | 对每个样本自身的特征做归一化，帮助深层网络训练稳定（和 BatchNorm 类似但归一化的维度不同） |
| **自回归 Autoregressive** | 只能看到"之前"的内容，逐词往后生成，GPT 系列的生成方式 |
| **Encoder / Decoder** | Encoder 负责"理解"整段输入，Decoder 负责"逐词生成"输出 |

---

## 11. 检查清单

- [ ] 能说出 RNN 的两个瓶颈，以及 Self-Attention 分别怎么解决
- [ ] 能用自己的话解释 Query / Key / Value 的作用，不看类比也能讲清楚
- [ ] 能手算一遍 2 个词的 Self-Attention（照第 4 节的数字重新算一遍）
- [ ] 知道为什么要做多头，而不是一个头把维度加大
- [ ] 知道为什么 Self-Attention 需要额外加位置编码
- [ ] 能画出一个 Transformer Block 的结构图，并指出哪些部件是"老朋友"
- [ ] 能说出 Encoder-only / Decoder-only / Encoder-Decoder 的区别和各自的代表模型

打勾之后，第三步的三个方向（CNN / RNN·LSTM / Transformer）就都过了一遍，可以进入入门笔记第 15 节的**第四步**：动手做一个完整项目。

---

*建议把第 4 节的数字改一改（比如换一组 Query/Key），重新手算一遍，感受"关注度"是怎么随数字变化的。*
