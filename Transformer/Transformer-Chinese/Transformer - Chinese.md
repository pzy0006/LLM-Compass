# Transformer - Chinese

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image.png)

传统的Transformer的组成主要分为两个部分，一个图片中左边的部分我们叫做encoder，一个右边部分叫decoder。 在实际环境中，encoder和decoder可以重复好几次，例如一个Transformer可能有6个encoder和6和decoder，一层加上一层的关系。

# Encoder Inputs

Encoder的主要负责：把输入序列（words / tokens）映射成 语义丰富的上下文表示。

其实Encoder的inputs是有两个部分组成的embedding + positional Encoding

## Embedding

整个embedding的过程就是把一些人类能读懂的语言转化成电脑能读懂的数字。类似于Java code编码过程，从code到byte code。

> 在进入 embedding 之前，首先需要进行 tokenization，即把原始文本拆分为更小的单元 —— tokens。在英文中，最简单的做法是基于空格来切分单词，并将标点符号单独作为 token。但在实际应用中，这种方式往往不够灵活，因为词表会过大，且难以处理生僻词或新词。为了解决这个问题，通常会引入subword segmentation 方法，如 BPE（Byte Pair Encoding） 或 WordPiece。它们可以把词拆分为更小的子词单元，从而减少未登录词（OOV）问题，同时控制词表规模。强烈推荐Andrej Karpathy视频：https://www.youtube.com/watch?v=zduSFxRajkE&t=5200s
> 

> 在完成 tokenization 之后，我们会得到一串离散的 token。然而，神经网络并不能直接理解这些符号，因此需要将每个 token 转换为对应的整数索引。为此，我们会构建一个 词表（vocabulary），其中包含所有允许出现的 token，并为每个 token 分配一个唯一的整数 ID。
> 
> 
> 例如，句子 “I love cream crackers” 经过词表映射后可能变成：["I": 101, "love": 102, "cream": 103, "crackers": 104]。
> 

### Step 1：

在得到一串整数ID后，比如[101, 102 ,103, 104]。这时候要通过embedding层把他们转成稠密向量表示。每一个 token id就是矩阵的一行索引。例如：如果embedding matrix 是100,000 * 512， ID = 101就取矩阵的第101行，得到一个512维度的向量。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%201.png)

以上步骤我们叫做embedding lookup。整个过程是发生在transformer原图中的“input embedding”中。

> Quick Review: embedding dimension越大，模型有潜力容纳的信息越多，但并不是无限越大越好。维度越高，embedding 矩阵和后续 Transformer 层的参数量成倍增加，训练和推理成本更高。如果训练数据不够大，而维度太高，模型容易学到噪声而不是泛化特征。
> 

我们得到这些向量看起来是这样子的，我这里用2D给大家展示一下

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%202.png)

在经过 Embedding 映射后，语义相近的词会拥有相近的向量表示。

例如：

- 向量 “Apple” 和 “Banana” 的距离较近，因为它们都属于水果；
- 向量 “BMW” 和 “Toyota” 也较为接近，因为它们都表示汽车品牌。

换句话说，具有相似语义的词，其向量在高维空间中的位置也会相近。

这一步的最终结果，是每个 token 都被表示为一个连续的 浮点向量（通常是数百维或上千维）。

假设我们使用一个非常简单的分词算法，句子：“I love eating apples and bananas.”
经过分词与 embedding 映射后，可能得到类似这样的结果（仅示意）：

| Token | Embedding（简化示例） |
| --- | --- |
| I | [0.12, -0.05, 0.33, ...] |
| love | [0.51, 0.22, -0.11, ...] |
| eating | [0.09, 0.47, -0.02, ...] |
| apples | [0.62, 0.13, 0.77, ...] |
| and | [-0.03, 0.08, 0.04, ...] |
| bananas | [0.60, 0.15, 0.75, ...] |

你可以看到，“apples” 和 “bananas” 的向量在数值上较为接近，表示它们的语义相似。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%203.png)

这一步的最终输出就是一个简单的matrxi， 其中包含了每个单词的向量，图上假设每个向量的维度是512.（向量维度可以是别的数字）

那么我们得到每个词的向量后是不是直接可以送进transformer中训练呢？答案是不行的！

Transformer并不知道每一个词的位置信息。例如你输入：I love cream crackers 和你输入：love cream I crackers, 对于transformer来说是一样的。我们必须想一个办法让Transformer理解每个单词之间是有位置关系的。请看Step2

> 如果你忘记RNN，你可以复习一下。
> 

### Step2 ：Positional Embedding

在transformer本文中，作者用的是Absolute Positional Encoding:

![Screenshot 2025-09-22 at 1.34.29 PM.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/Screenshot_2025-09-22_at_1.34.29_PM.png)

**各个符号的含义：**

- **`d`**：embedding 的维度。
    - 例如 BERT-base 的输入 embedding 是 768 维；如果你设定 `d = 512`，那就是一个 512 维的向量。
- **`i`**：维度的索引。
    - 因为公式是对偶数和奇数维分别用 sin 和 cos，所以 `i` 通常取范围 `0,1,2,…,d/2-1`。
    - 举个例子：
        - `i = 0` → 计算第 0 和第 1 个维度（sin/cos 一对）。
        - `i = 210` → 计算第 420 和第 421 个维度。
- **`pos`**：token 的位置索引（第几个 token），比如第 0 个词、第 1 个词……
- **结果**：每个位置 `pos` 最终会得到一个长度为 `d` 的位置向量，交替填充了不同频率的 sin 和 cos。

> 除了绝对位置编码，我们还有可学习位置编码和相对位置编码。之后我会对这三个算法做比较，但是我想重点讲解RoPE相对位置编码算法。
> 

当我们算出每一个向量的绝对位置信息后：

$$
P=[p_1,p_2,...,p_n],p_i∈R^d
$$

会跟 input embedding：

$$
X=[x_1,x_2,...,x_n],x_i∈R^d
$$

相加：

$$
Z=X+P
$$

得到最终的序列表示

$$
 Z=[z_1,z_2,...,z_n]
$$

其中每个

$$
z_i=x_i+p_i
$$

通过各种位置编码算法计算，我们真正的输入其实是向量Z。向量Z的size是：

$$
Z = (B×N×D)
$$

其中：

- **batch size** = B：一次送入模型训练/推理的 **样本数**。
- **序列长度** = N：每个样本中 **token 的数量**（即句子长度）
- **embedding 维度** = D：每个 token 的向量表示长度（即每个 token 的特征维度）。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%204.png)

> 上面的步骤展示了如何将位置信息向量加入到原始词向量中，本质上就是对这两类向量进行加权求和的过程。不过，这种直接相加的方法存在一个问题：位置信息可能会干扰甚至破坏原始语义向量所携带的意义。因此，我们需要一种更巧妙的方式，既能保留原始向量的语义，又能有效地融入位置信息。关于这一点，我将在后续的文章中详细介绍**RoPE（Rotary Position Embedding）**算法，它正是为了解决这一问题而提出的。
> 

### Step 3

这一步是重点。在注意力机制中我们会把Z分成三个部分Q，K和V（注意他们三个matrix的size是一样的）。这三个matrix包含了不同信息，作用也是不同的。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%205.png)

在进入multi head attention 里面首先我们对Z进行加权重计算，以至于我们能获得对应的Q，K和V。

$$
Q=ZW_Q,K=ZW_K,V=ZW_V
$$

其中：

$$
W_q, W_v, W_k
$$

是可学习的（即在训练过程中，这三个矩阵不断变化，最终找到一个最佳矩阵。）权重矩阵。

$$
Q,K,V
$$

是 **查询（Query）**、**键（Key）**、**值（Value）** 矩阵。

Q是当前想要提问的向量，K是当前 token 的“特征标签”，V是真正携带的信息。

> 矩阵从Z到Q，K和V的转换属于线性转换。
> 

### Step 3.1：Attention Score

这一步我们主要讲 Q，K 和V 怎么相互作用，我们能从中得到什么信息。原文paper中用的是multi head attention。我在这里先用single head attention为例子，看看整个计算过程是什么样的。

$$
\begin{equation}\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V\end{equation}
$$

上边式子就是我们如何计算attention的。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%206.png)

例子：[”I”, “love”, “cream”, “crackers”] 对于每一单词，都有对应的q,k和v，他们的size是1*5， 这里的5就是向量维度。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%207.png)

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%208.png)

上图对应是的式子就是

$$
Q*K^T
$$

意思就是对于任何一个单词的$q$ 都要去跟所有单词（包括自己）的$k$相乘。例如：

单词“I”的$q$跟每一个单词[”I”, “love”, “cream”, “crackers”] 的$k$相乘。

$$
[q_1*k_1, q_1*k_2,q_1*k_3,q_1*k_4]
$$

单词“love”的$q$跟每一个单词[”I”, “love”, “cream”, “crackers”] 的$k$相乘。

$$
[q_2*k_1, q_2*k_2,q_2*k_3,q_2*k_4]
$$

以此类推。

那么我们为什么要这样计算呢？其实这里是有数学意义的：

当我们计算

$$
Q*K^T
$$

时，实际上是每个Query和所有Key的inner product。

- Inner product越大，说明Query和Key的越匹配，相关性越强。
- Inner product越小甚至为负，说明Query对那个Key不感兴趣。

换句话说，第I个token想要“关注”第J个token的程度，这就是我们所说的**Attention Score。**

### Step 3.2：数值缩放

为什么$Q*K^T$要除以 $\sqrt{d}$  ? （注意这里的d是向量维度，在多头注意力机制中，这里应该是 $d_k$，后续我们会讲。）

在Q 和K 相乘过程中，有些矩阵中的数值会变的非常大，为了控制数字在一个何以范围内，我们需要除以 $\sqrt{d}$ （也就是缩放），把Inner product的数值控制在一个合理的范围。

### Step 3.3: Softmax

我们先看公式：

$$
\begin{equation}\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{j'} \exp(s_{ij'})}\end{equation}
$$

其中：

- $s_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d}}$是 token i 对 token j 的相关性分数
- $\alpha_{ij}$就是 注意力权重

当你算出来一个单词对所有单词的attention score，并且等比例缩小（除以sqrt（d））之后，这些数值只是相对的相似度，还不能直接表示注意力分布。Softmax 将这些 attention score 转换为一组加起来为 1 的attention weights，从而明确了每个单词应该“关注”其他单词的比例。

Softmax的作用就是**把一行分数转成概率分布**：

- 每个$α_{ij}∈(0,1)$：正态分布，每个注意力权重就像“概率”，不会出现负权重或者大于 1 的情况。
- 并且  $∑α_{ij}=1$：保证所有权重合起来就是 100%

比如 “love” 的 Query 得到的分数是 [3.2, 0.8, -5.7]：

- Softmax 结果 = [0.71, 0.28, 0.01]
- 含义：love 70% 关注 “I”，28% 关注自己，1% 关注 cream，几乎不看 crackers。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%209.png)

### Step 3.3

通过步骤3.2，我们获得一个权重矩阵，表示每一个单词对所有单词的注意力。

假设最终final output 是 $Z$，权重矩阵是 $A$ ，还有一个矩阵 $V$。（这里为了方便计算，V矩阵大小为4*2）

$$
Z =\begin{bmatrix}0.7 & 0.1 & 0.1 & 0.1 \\0.2 & 0.5 & 0.2 & 0.1 \\0.25 & 0.25 & 0.25 & 0.25 \\0.4 & 0.1 & 0.4 & 0.1\end{bmatrix}\cdot\begin{bmatrix}1 & 0 \\0 & 1 \\1 & 1 \\0 & 2\end{bmatrix}
$$

逐步计算

第一行

$$
z_1=0.7[1,0]+0.1[0,1]+0.1[1,1]+0.1[0,2]=[0.8,0.3]
$$

第二行

$$
z_2=0.2[1,0]+0.5[0,1]+0.2[1,1]+0.1[0,2]=[0.4,0.9]

$$

第三行

$$
z_3=0.25[1,0]+0.25[0,1]+0.25[1,1]+0.25[0,2]=[0.5,1.0]
$$

第四行

$$
z_4=0.4[1,0]+0.1[0,1]+0.4[1,1]+0.1[0,2]=[0.8,0.6]
$$

最终输出矩阵：

$$
Z =\begin{bmatrix}0.8 & 0.3 \\0.4 & 0.9 \\0.5 & 1.0 \\0.8 & 0.6\end{bmatrix}\quad \in \mathbb{R}^{4 \times 2}
$$

这个矩阵意思就是：每个 token 的输出向量 = 它对所有其他 token 的注意力权重 × 各 token 的 Value 向量，然后加权求和。换句话说，最终输出的矩阵的第i 行（也就是第i 个 token 的输出向量），表示这个 token 在“关注”所有其他 token 后，综合它们的 Value 信息后得到的上下文表示。

### Multi head attention

单头注意力只能从一个“投影空间”中学习关联，而多头注意力通过多个不同的线性变换（即多个“头”），在不同子空间中并行计算注意力，从而增强模型的表示能力。 

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2010.png)

$$
head_i = Attention(Q_i, K_i, V_i) = softmax\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i, \quad i \in [1, H]
$$

其中：

$i$：第i个注意力头 （head index）， 一共有 $H$个

$Q_i$：第i个头的Query矩阵，形状 $[n * d_k]$

$K_i$：第i个头的Key矩阵，形状 $[n * d_k]$

$V_i$：第i个头的Value矩阵，形状 $[n * d_k]$

$d_k$:Key/Query向量维度 $d_k = d_{model} / H$

> 如何计算 $d_k$? 假设你的输入Q的维度是 $d_{model} = 518$, 多头数量是 $h = 4$, 那么 $d_k = d_{model} / h = 128$
> 

$$
MultiHeadAttention(Q, K, V) = Concat(head_0, head_1, \ldots, head_H)
$$

这个公式就是把所有 $head$的结果拼接在一起，得到multi head attention 的整体输出。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2011.png)

为什么使用multi-head？

**在不同子空间（切割后的矩阵）并行建模关系**

把总维度

$$
d_{model} 
$$

划分成h个头，每个头各自学习一套

$$
W^Q,W^K,W^V
$$

等于在不同表示子空间里计算注意力；这样能同时捕捉短程/长程、句法/语义等不同类型的依赖（例如head 1 捕捉相邻单词的关系，head2捕捉主谓宾关系，head3捕捉语法结构等），再把各头输出拼起来形成更丰富的表示。这是原论文与教材对多头的核心动机。

> 一句话总结多头注意力机制就是为了捕捉单词与单词之间关系，提升模型对上下文的理解能力。
> 

### Step 4：Normalization and residual network

现在我们讲解这一部分，图中Add表示的是residual network， norm表示normalization。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2012.png)

> 什么是深层网络？为什么需要深层网络？深度网络问题。
> 

Deep Neural Network是由多层非线性变换堆叠而成的神经网络。在现代大模型中，网络层数往往达到几十层甚至上百层。从理论上讲，网络越深，它能够学习到的信息就越丰富：底层侧重捕捉简单的模式（如边缘、颜色），中层逐步组合这些模式形成更复杂的特征（如纹理、局部结构），高层则进一步抽象为语义层面的信息（如物体类别），就像人类理解世界一样，从感知到概念逐步过渡。

但是，深层网络并非简单地“越堆越深”就能无限提升性能。随着层数增加，我们会遇到两个主要问题：Gradient Exploding/Vanishing 和 degradation。前者指的是在反向传播时梯度数值过大或过小，导致训练不稳定甚至无法收敛；这类问题通常可以通过 归一化（Normalization） 或 合适的激活函数 得到缓解。真正更棘手的是 degradation：随着层数加深，非线性激活函数在信息传播过程中会引入不可逆的损失，结果是网络越深，反而信息丢失越多，性能下降。

> 网络退化解决方案：residual network
> 

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2013.png)

残差网络层是：

$$
y = f(x) + x
$$

其中：x是输入（原始信息），f（x）：这层学习到的信息， y：原始信息+学习到的信息。

无论 f(x) 内部如何变化，输入的原始信息 x 都能够直接传递到下一层。如果这一层的 f(x) 没有学到任何有用的特征，至少x 还能保留下来，不至于让网络性能下降。这正是 Residual Network 的核心思想：让模型只学习在输入基础上的改进，而不是完全替代输入。

> 强烈推荐学习：前向传播和反向传播，ResNet论文，residual network在前向和反向传播的作用和激活函数是什么。
> 

> 什么是Normalization？
> 

**Normalization（归一化/标准化）** 就是对数据或中间特征进行数值变换，让数据落在一个更“规范”的范围或分布内。例如把数据范围统一在0和1之间。

在现在大语言模型训练中，模型通常都是几十层或者100多层，这就极其容易导致输入和中间特征的数值分布可能差别非常大或者非常小，这就导致训练不稳定。

> 数值大：梯度爆炸。数值小：梯度消失。
> 

**Normalization 的作用**：

1. **稳定训练**：控制数值范围，避免梯度爆炸或消失。
2. **加快收敛**：减少内部协变量偏移，让训练更快收敛。
3. **提升泛化能力**：减少过拟合，让模型在新数据上表现更好。
4. **支持更深网络**：帮助残差结构在深层模型中保持有效梯度流。

> 一句话总结：Normalization 就是通过统一特征分布，让网络训练更稳、更快、更准。
> 

我们了解到Normalization是对数值做归一化，即控制数值在某一个范围内。那么问题来了，我们对什么样的数值做归一化处理？在原文Transformer中，作者用的Layer normalization，它对单一样本的所有特征维度做归一化。

例如：我们有一个输入token事512维度的

$$
x=[x_1,x_2,x_3...x_{512}]
$$

Layer Normalization 就单独对这个token做归一化处理。下面是LN式子和大致步骤。

先算这个512维度向量的均值：

$$
\mu = \frac{1}{512} \sum_{i=1}^{512} x_i
$$

在计算方差：

$$
\sigma^2 = \frac{1}{512} \sum_{i=1}^{512} (x_i - \mu)^2
$$

把每个元素归一化：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}   \text{where i = 1,2...512}
$$

最后做线性变换（可学习的缩放和平移)

$$
y_i = \gamma_i \cdot \hat{x}_i + \beta_i\quad \text{where } \gamma, \beta \in \mathbb{R}^{512}
$$

以上Layer normalization步骤，放在Transformer中它的表达式是：

$$
\text{Output} = \mathrm{LayerNorm}\!\big(x + \mathrm{Sublayer}(x)\big)
$$

其中sublayer就是f（x）。

> 在Transformer原文中，作者用的post-norm，其实我们现在用的pre-norm比较多。
> 

### Step 5: FFN - Feed forward network

截至目前，我们所接触到的公式大多是线性的。人们总希望找到一种直观且线性的表达方式，但现实世界中，很多现象本质上却是非线性的，比如语言。在大模型的训练过程中，为了让模型能够习得更多知识，具备更丰富的表达能力，我们通常会设法在模型中引入非线性的公式。

> FFN（前馈网络）是一种逐位置的两层全连接神经网络，用于对每个输入向量进行非线性变换和特征提炼。如果对FFN概念不熟悉建议单独学习，同时别忘记学习RNN概念
> 

接下来，让我们看看Transformer中怎么使用FFN

$$
\text{FFN}(x) = W_2 \, \sigma \!\left( W_1 x + b_1 \right) + b_2
$$

Step1:

$$
W_1x+b_1
$$

假设你有一个向量

$$
x∈R^{d_{model}},d_{model}=4
$$

这个向量目前只包含了基本的语义信息，但维度太小，模型很难从中捕捉到复杂而丰富的模式。

为了让模型具备更强的表达能力，我们需要一个更大的空间来暂时保存和加工信息。于是我们将向量的维度从 $d_{model}$ 扩展为 $d_{ff} = 4 * d_{model} = 16$ (通常是4倍)

具体做法是引入一个权重矩阵:

$$
W_1∈R^{16×4},b_1∈R^{16}
$$

通过一次线性变换：

$$
h_1=W_1x+b_1,h_1∈R^{16}
$$

这样，原本只有 4 个数的向量，就被映射成了一个 16 维的向量。扩展后的向量在高维空间中可以承载更多的组合特征和潜在模式。

Step 2:

随后我们再对其施加非线性激活函数（如 ReLU 或 GELU）：

$$
h_2=σ(h_1)
$$

> 激活函数是神经网络中作用在线性变换输出上的非线性函数，它既能打破多层线性叠加仍然等价于单层的限制，又能引入非线性能力，使模型在更高维空间中筛选、变换特征，从而学习和表达更复杂、更抽象的模式。常见的激活函数公式有sigmoid，ReLU，GELU等，之后我会单独讲解。
> 

这一步让模型具备了更强的非线性表达能力。最后，为了回到原始的维度，我们再使用一个压缩矩阵：

$$
W_2∈R^{4×16},b_2∈R^4
$$

得到输出：

$$
y=W_2h_2+b_2,y∈R^4
$$

这样，一个 4 维向量在中间过程被扩展到 16 维（更大容量），经过非线性变换后再压缩回 4 维，最终的表示y 就比最初的输入x 更加丰富和抽象。

> 为什么FFN要先升维后降维？
> 

以上FFN层的步骤，现在我们讨论下FFN核心作用：

- **增强非线性表示能力**
    
    注意力层本身是线性的（点积 + Softmax），FFN 的非线性部分能让模型学习复杂的模式和特征组合。
    
- **扩展与压缩**
    - 扩展到更高维度（ $d_{ff}$）时，模型有了“更大的空间”去挖掘和组合特征。
    - 再压缩回原维度，确保输出和输入的形状一致，方便堆叠层。
- **逐位置（position-wise）的特征抽取**
    
    在注意力层已经整合上下文信息之后，FFN 就像是每个 token 的“独立加工厂”，帮助它进一步提炼和抽象信息。注意这里并不是说使用FFN我们就不能并行处理token了。
    
- **提高模型深度**
    
    自注意力负责“信息交互”，而 FFN 负责“局部加工”，二者交替堆叠，赋予 Transformer 足够的计算深度和表达能力。
    

我额外想补充一点是两个权重矩阵 $W_1$ 和 $W_2$ 是可学习参数。它们在训练过程当中通过backpropagation计算梯度，并由优化器（如 Adam，这里有梗 who is Adam）不断更新，从而逐步学会如何把输入向量变换成更有用的表示。

> 又是一个坑，什么是backpropagation？简单理解就是：**神经网络在得到预测误差后，把这个误差一层层往回传，算出每个参数该怎么调，然后更新参数，让下次预测更准确。这里要用到求导了。忘记如何求导的同学可以复习一下。**
> 

### Step6

FFN层结束后，就是正常的residual network 和normalization了，这一层的作用和之前一样的。同样这里用的是post norm：

$$
z=LayerNorm(x+FFN(x))
$$

### Encoder Final output

截至目前，我们只完成了一个 Encoder Block。实际上，Encoder 是由多个 Encoder Block 堆叠而成的，例如第 1 个 Block 的输出会作为第 2 个 Block 的输入，它们层层堆叠，直到得到最终的输出。这个最终的 Encoder 输出是一个与输入序列长度相同的矩阵，其中每一行对应一个 token 的上下文表示。该表示不仅保留了原始语义信息，还通过自注意力机制融合了全局上下文，从而成为更加丰富的语义特征。在原始 Transformer 架构中，Decoder 的一部分输入（交叉注意力中的 Key 和 Value）正是来源于 Encoder 的最终输出。

## Decoder

从 Transformer 框架 中我们可以看出，Decoder 的整体结构 与 Encoder 基本相似，同样由多层堆叠的子层组成，每一层内部都包含残差连接（Residual Connection）、层归一化（Layer Normalization）以及前馈全连接网络（Feed-Forward Network）等模块。这些与 Encoder 相同的部分在此不再赘述。

**不同点** 主要体现在以下几个方面：

1. **Masked Multi-Head Self-Attention**
    - Decoder 在输入端首先会经过一个带有 **Mask机制** 的多头自注意力层。
    - 掩码的作用是确保每个位置在预测时只能看到当前及之前的词，而不能看到未来的词，从而保证自回归生成（auto-regressive generation）的正确性。
2. **Encoder–Decoder Attention（交叉注意力层）**
    - 与 Encoder 不同，Decoder 在自注意力之后还会引入一层额外的 **多头注意力机制**。
    - 这层注意力以 Encoder 的输出作为 Key 和 Value，而 Decoder 自身的表示作为 Query，从而使 Decoder 能够有效地“查询”源序列的信息，实现对输入序列的条件建模。
3. **输出层（Prediction Layer）**
    - Decoder 在经过若干层堆叠后，最终会输出一系列隐藏表示。
    - 这些表示会送入一个线性层（Projection Layer），再经过 Softmax，得到对词表中每个 token 的概率分布，用于生成下一个词。

在正式进入讲解Decoder中的components之前，我还想要引入两个概念：Teacher Forcing 和自回归结构。

**什么是Teacher Forcing？**

这个是一种训练模型策略，它可以引导和加速模型训练。在训练decoder时候，输入不是前一步模型自己预测出来的token，而是前一步的真实目标token，我们叫做ground truth。

**什么是自回归模式？**

这个模式正好跟teacher forcing相反，在模型进行下一个token 预测时候，每一步的输出都只能依赖于之前已经生成的输出，并且只能基于“过去”来预测“未来”。

### Step 1：Input

在Decoder 的Input跟Encoder 是不一样的。Decoder的输入：

$$
<BOS> I \ love \ cream \ crackers 
$$

其中BOS stands for beginning of sequence。 这种形式的输入其实就是常说的 **shifted right**：将目标序列整体右移一位。

在理想情况下，我们希望在一次前向传播中就能得到序列中所有位置的预测结果，以充分利用 GPU 并行计算能力。
然而，如果 Decoder 严格按照自回归方式逐步生成，那么训练时就必须一词一词地解码，效率极低。

To improve training efficiency, we want to leverage parallelism so that the model can predict all tokens in a sequence within a single training step. To achieve this, researchers introduced **Teacher Forcing**, a technique that allows the decoder to take the entire target sequence as input during training, enabling parallel decoding of all outputs at once.

More specifically, Teacher Forcing means that at each step, instead of feeding the decoder with its own previous prediction, we feed it with the ground truth token from the training data. This mechanism ensures that the Transformer can produce predictions for all positions in parallel during training, without relying on sequential decoding. As a result, training becomes significantly faster.

In practice, the decoder input is not just a single label at each step but a concatenated shifted sequence, as illustrated in the figure.

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2014.png)

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2015.png)

**掩码自注意力（Masked Self-Attention）**

在Transformer中，我们能看到两种不同的masking 机制。一个masking机制是在准备输入时候，一个masking机制是在masked self attention中。这两种masking机制分别叫说 Padding masking 和sequence masking。

- Padding Masking 在自然语言处理模型中非常常见。由于不同句子的长度通常不一样，为了能够批量输入并保持张量形状一致，我们需要将较短的句子补齐到统一的长度。这时会在句子后面添加一个特殊符号 `<PAD>` 作为占位符。例如，如果我们希望输入矩阵的长度统一为 4，那么句子 “I love you” 会被处理成 “I love you `<PAD>`”，而句子 “I love cream crackers” 则刚好占满 4 个位置，不需要补齐。
需要注意的是，`<PAD>` 并没有实际语义，仅仅起到占位作用。如果在计算注意力分布（如 Transformer 的 self-attention）时不做处理，模型可能会错误地将注意力分配到 `<PAD>` 上，进而干扰正常的预测。为此，引入了 Padding Mask 机制。在计算注意力分数时，模型会将 `<PAD>` 位置对应的数值替换为一个极小的值（例如 -1e9），这样经过 softmax 后，这些位置的权重几乎为零，相当于被忽略掉，从而保证模型只关注真实的有效 token。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2016.png)

- Sequence Masking: Sequence Masking 也是 Transformer 中常见的一种掩码机制。与 Padding Mask 不同，Sequence Mask 主要用于保证解码器在训练和推理过程中只能利用已经生成的词，而不能“偷看”未来的信息。因为在自回归的生成模式下，每一步的预测都必须依赖于之前的 token，而不是后面的 token。举个例子，假设目标序列是 “I love you”。在预测第一个词时，模型只能看到 `<BOS>`；在预测第二个词时，模型只能利用 `<BOS> I`；在预测第三个词时，模型只能利用 `<BOS> I love`。如果没有 Sequence Mask，模型在计算 self-attention 时就可以同时看到 “you”，这会破坏自回归生成的设定。为了解决这个问题，我们使用了一个 上三角掩码矩阵（causal mask），将未来时刻的注意力分数屏蔽掉，通常做法是把这些位置的数值设为一个极小的负值（如 -1e9）。这样在 softmax 之后，这些未来位置的权重就几乎为零，模型就只能关注当前和过去的 token。通过这种方式，Sequence Mask 保证了 Transformer 解码器的自回归特性，既能防止信息泄露，又能让训练与推理保持一致。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2017.png)

> 以上我们讲解了两种不同形式的masking，为什么decoer要sequence masking？这是因为Transformer 采用的是Autoregressive模式（逐个 token 地生成序列，每一步都基于前面已经生成的 token 来预测下一个 token）。这个模式保证三个好处：1 确保生成逻辑正确，2:保持训练和推理阶段一致性，3:防止information leakage。注意sequence masking 只针对自回归模式的训练和推理时候的prefill 阶段，推理时候的 decoder不需要masking机制。
> 

我们来看一下它的计算步骤：

1. **输入嵌入（Embedding）**
    
    将输入序列通过 embedding 和位置编码，得到向量表示 $X$ 。
    
2. **线性变换**
    
    $X$分别乘以三个权重矩阵 $W^Q,W^K,W^V$，得到 $Q,K,V$ 三个矩阵。
    
3. **注意力分数计算**
    
    $Q$ 与 $K^T$ 相乘，得到注意力打分矩阵 $QK^T$。
    
4. **应用掩码（Mask）**
    
    将 $QK^T$与一个 Mask 矩阵逐元素相加。
    
    - Mask 矩阵的下三角部分保持不变，表示当前位置及之前的 token 可以被看到。
    - 上三角部分设为 −∞，即“屏蔽未来信息”。
    - 经过 softmax 后，被掩码的位置权重变为 0。
5. **加权求和**
    
    将 Mask 后的注意力权重矩阵与  $V$相乘，得到新的上下文表示矩阵 $Z$。
    
    换句话说， $Z$ 的每一行就是对当前位置之前所有词向量的加权平均。
    

### Step 2: Add & Norm

这一步骤就是跟Encoder的作用是一样的。不再过多阐述。

### Step3: **Encoder–Decoder Attention**

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2018.png)

前面我们讲过 Self-Attention，它的本质是序列内部“自己和自己”的信息交换。而我们使用模型的目的在于预测，要实现预测必须结合 历史信息。在 Encoder-Decoder 框架中，历史信息存储在 Encoder 的输出里，那么如何保证预测时的生成信息能够和这些历史信息建立联系呢？这时就引入了 Cross-Attention。

Cross-Attention 的核心思想是在 Encoder 和 Decoder 之间建立桥梁，让源序列与目标序列进行交互。它通过计算 目标序列中每个位置与源序列所有位置的相关性或相似度，来决定在生成某个目标 token 时，应当更关注源句子中的哪些部分。换句话说，源序列提供上下文语义，目标序列负责生成预测，两者通过交叉注意力紧密结合。

**Q / K / V 的来源**

在交叉注意力中：

- **Q（Query）** 来自 Decoder 的输出（经过 Masked Self-Attention，已经包含历史译文信息）。
- **K（Key）** 和 **V（Value）** 来自 Encoder 的输出（包含源文本的上下文表示）。

这样，交叉注意力就像一个信息检索过程：

- Encoder 的输出可以看作一个“数据库”（V，包含源序列的全部信息）。
- Decoder 的每一个位置发出一个“查询”（Q），去寻找数据库中最相关的部分（通过与 K 的相似度计算）。
- 最终得到的加权结果告诉 Decoder：在生成当前 token 时，应该侧重源句子中的哪些词。

**大致步骤：**

- **源序列（Encoder 输入）**：`I love cream crackers`
- **目标序列（Decoder 输出）**：`我 爱 奶油 饼干`

当 Decoder 要预测 `"饼干"` 时：

1. 它的 **Query** 来自前文 `"我 爱 奶油"` 的隐藏表示。
2. 它去 Encoder 的输出（`I love cream crackers` 的表示）里查找最相关的位置：
    - 发现 `"crackers"` 的 Key 最相关。
3. 所以会把 `"crackers"` 的 Value 拿出来，融合到 Decoder 当前的隐状态中。
4. 最终帮助 Decoder 生成目标词 `"饼干"`。

**计算过程：**

Same as encoder

### Step 4 FFN + Add & Norm

Same as encoder

### Step 5: Linear layer and softmax = generator

走到这一层时候，我们已经算出来下一个token是什么了，但是此时token的表现形式还是数字，即向量。我们必须想到一个办法让这个向量转成文字，人们能够读得懂。

![image.png](Transformer%20-%20Chinese%202719e9af5d5a808394c8e4fd3a4f7ec3/image%2019.png)

**Linear**

在线性层（通常称为 Language Model Head 或 LM Head）中，主要目标是将解码器生成的隐向量映射到词汇表的维度上，从而为下一个词的预测做好准备。换句话说，线性层的本质功能是 维度转换：将解码器输出的向量转化为与词表大小相同的向量（logits）。

你也可以理解为Linear层是把预测出来的向量跟词汇表中的每一个单词做inner product，等到和每一单词的相似度，即logits。logits 是一个候选词的得分向量，表示每个词在当前位置作为下一个 token 的可能性。向量的每个维度都对应词表中的一个词，数值越大，表示该词是正确输出的概率越高。

> 得到 logits 后，模型并不是简单地总是选择概率最大的那个词，否则生成的句子会非常机械、缺乏多样性（例如“我爱你” → 永远回答“我也爱你”）。为了解决这个问题，通常会引入采样策略，如Top-k或Top-p（Nucleus Sampling）从得分较高的一部分词中进行抽样。这样既能保证合理性，又能增加生成的多样性。
> 

**Softmax**

线性层的输出是logits，它们本身并不能直接解释为概率。为了解决这一问题，我们会在 logits 上应用softmax 函数，将向量的最后一维缩放到[0,1]的区间，并保证所有元素之和为 1。这样处理之后，模型的输出就可以被解释为一个概率分布，表示在当前位置上每个词作为下一个 token 的可能性。在多分类任务中，这一步尤为关键，因为它让预测结果具有概率意义。随后，模型会根据这个概率分布进行采样，从而选择最终生成的词。

**Sampling**

在自然语言生成任务中，**采样** 是指根据模型输出的概率分布，随机选择下一个词的过程，而不是一味地挑选概率最大的词。与贪心搜索相比，采样的核心优势在于它能引入一定的随机性，从而避免模型在生成时陷入机械和重复的模式。

之所以需要采样，是因为如果模型总是选择概率最高的那个词，生成的句子往往会非常僵硬、缺乏变化。例如，当你输入“我爱你”，模型可能永远只会输出“我也爱你”。而通过采样，模型可以在多个高概率候选词之间进行随机选择，从而生成更丰富的表达方式，比如“我也很喜欢你”或“我对你有同样的感觉”。

因此，采样不仅能有效提升句子的多样性，还让生成的结果更加自然、更贴近人类的语言习惯。不同的采样策略（如 Top-k 或 Top-p）还能在“合理性”和“多样性”之间灵活平衡，保证输出既不会太离谱，又能避免一成不变。

> 在这一步骤中又很不同的sampling algorithm，我就不介绍了，有兴趣可以自己学习。
> 

### Final output

采样之后，得到的 token 会被加入生成序列，并作为下一步的输入，整个过程不断循环，直到句子完成。

