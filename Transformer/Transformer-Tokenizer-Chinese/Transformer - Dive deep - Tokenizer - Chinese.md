# Transformer - Dive deep - Tokenizer - Chinese

![image.png](Transformer%20-%20Dive%20deep%20-%20Tokenizer%20-%20Chinese/image.png)

我们已经知道Transformer接受的variable只能是高维度向量，在一个段落/文字转化成embedding matrix 之前需要经历一个非常重要的阶段：分词。一个好的分词器能够决定你的模型的精确度和性能。

### 分词的主要步骤：

1. **Vocabulary Building**
    - 根据corpus统计出出现过的所有文本单位（可以是字符、词、子词等），去重后形成词表。
    - 不同分词算法（如 WordPiece、BPE、Unigram）会根据频率、合并规则等动态生成词表。
    - 每个 token 在词表中都会被分配一个唯一的整数 ID。
2. **Pre-tokenization**
    - 包括大小写归一化、去除特殊符号、标点标准化等。
    - 长文本通常会被分割成多个句子或固定长度的段落，以便输入模型时不会超出最大长度限制。
3. **Tokenization**
    - 将文本按照词表规则切分为更小的单元。
    - token 的粒度取决于算法类型：
        - **Word-level**：按词分割（容易遇到 OOV 问题）。
        - **Character-level**：按字符分割（粒度太细，效率较低）。
        - **Subword-level**（最常见，如 BPE 或 WordPiece）：在词与字符之间取得平衡，能很好地处理新词。
4. **Indexing**
    - 将每个 token 映射为词表中的整数索引（ID）。
    - 索引序列就是模型的输入。
    - 在这一阶段通常会加上特殊标记，如：
        - `[CLS]`：句子起始
        - `[SEP]`：句子分隔
        - `[PAD]`：填充到固定长度
5. **Embedding**
    - 模型通过查找 embedding 矩阵，将每个 token 的 ID 映射为对应的向量表示。
    - 向量捕捉 token 的语义信息、上下文关系及其他特征，是模型理解和生成语言的基础。

# Tokenizer Algorithm

Tokenizer算法有非常多。如果按照分词颗粒度分类的话，有三种：word base，char base and subword base

- Word base
    - 单词级别的分类。例如 I love cream crackers 分词后[”I”, ”love”, ”cream”, “crackers”]
    - 优点：可以很好保持语义完整度，算法实现简单
    - 缺点：需要列出所有单词组合，严重影响计算效率和消耗内存。会导致OOV（out of vocabulary）问题：模型词表中没有的词。
- char base
    - 字符级别的分类。例如[I, l, o, v, e, c, r….]
    - 优点：字符少，不会产生OOV问题
    - 缺点：无法获取丰富的语义等信息。模型很难学习词与词，句子与句子之间关系。
- subword base
    - 介于word和char之间，例如[I, lo, ve, cr, ea, m, cra, c, kers]
    - 优点：有效缓解 OOV 问题。相对于char base，更高的泛化能力。对于形态丰富语言（如德语、芬兰语、阿拉伯语）尤其有效
    - 缺点：生成任务复杂度增加：对生成式模型（如 GPT）而言，一个词需要多个 token，导致生成速度变慢、长度变长。

| **算法名称** | **典型使用模型** | **核心思想 / 特点** | **优点** | **缺点** |
| --- | --- | --- | --- | --- |
| **Word-level Tokenizer** | Word2Vec、GloVe、早期 RNN、ELMo | 按完整单词分割。每个词是一个 token。 | 实现简单、语义清晰 | 容易产生 OOV，词表巨大 |
| **Character-level Tokenizer** | CharCNN、DeepMoji、一些语音/拼写模型 | 每个字符作为一个 token | 无 OOV、跨语言 | 序列太长，语义弱 |
| **BPE（Byte Pair Encoding）** | GPT-2、GPT-3、RoBERTa、MarianMT | 基于统计频率合并最常见字符对形成子词单元 | 平衡 OOV 与词表大小 | 子词不总是语言学上自然 |
| **WordPiece** | BERT、ALBERT、DistilBERT、ELECTRA | 通过概率最大化方式选取子词组合（类似 BPE，但更精确） | 能更好地处理罕见词 | 构建词表复杂，计算稍慢 |
| **Unigram Language Model** | T5、XLM-R、mT5、ByT5 | 用语言模型选择最优子词集合，每个 token 是一个概率单元 | 子词划分更灵活 | 训练过程复杂 |
| **SentencePiece (基于BPE或Unigram)** | T5、mT5、XLNet、ALBERT（Google版） | 直接作用于原始字节，不依赖空格，支持多语言 | 跨语言通用，训练稳定 | 输出 token 不直观（包含前缀符号，如 "▁"） |
| **Byte-Level BPE** | GPT-2（OpenAI 原版）、GPT-3、GPT-4、LLaMA | 直接在字节层面进行 BPE 编码，可处理任意字符 | 无需额外字符集，完全避免 OOV | token 较多，计算稍重 |
| **SentencePiece (Character-level)** | ByT5、Charformer | 将文本转成 UTF-8 字节序列再建模 | 语言独立 | 语义碎片化 |
| **Tiktoken (OpenAI 特制版)** | GPT-3.5、GPT-4 系列（ChatGPT） | 改进版 byte-level BPE，优化编码效率 | 快速、兼容多语言、节省上下文窗口 | 封闭实现（不开源） |
| **SPM BPE + BERT Tokenizer Hybrid** | DeBERTa、ERNIE、NEZHA（中文模型） | 融合 SentencePiece 与 WordPiece/BPE，适合中文多粒度 | 处理中文效果好 | 调参复杂、词表难统一 |

# BPE

BPE算法最早是由google提出来的，文献链接：https://arxiv.org/abs/1508.07909

BPE（Byte Pair Encoding）算法先将每个文本词（Word）拆分成 Char 粒度的字母序列，然后从字符级别开始，每一步都将出现频数最大的一对相邻 token 合并为该数据中没有出现过的一个新 token。这样可以逐渐构建出更长的词汇或短语表示，形成一种基于统计频率的层级式子词结构。

这个过程会反复迭代，直到达到预设的词汇表大小（vocabulary size）或合并次数（merge operations） 为止。训练完成后，会生成一张 合并规则表（merge table），其中记录了每一次 token 合并的顺序和规则。后期在对文本进行分词（encoding）或还原（decoding）时，都需要使用这张表来重建原始数据的 token 序列。

BPE 的核心思想是一种贪心算法：它在每次迭代中都局部选择当前出现频率最高的 token 对进行合并，但这种策略不一定能保证全局最优；同时，频数也不一定是最佳的合并指标（例如有时会破坏语义边界）。

尽管如此，BPE 仍然是一个性能极高、结构简单且可控性强的分词算法。

它不仅能显著缓解 OOV（Out Of Vocabulary） 问题，还能方便地将 token 总数量控制在手动设定的范围内，在效率与效果之间取得很好的平衡。

因此，BPE 已成为许多现代 Transformer 模型（如 GPT-2、RoBERTa、MarianMT 等）默认使用的分词算法之一。

![Screenshot 2025-10-24 at 3.57.05 PM.png](Transformer%20-%20Dive%20deep%20-%20Tokenizer%20-%20Chinese/Screenshot_2025-10-24_at_3.57.05_PM.png)

（以上图片来源于BPE文献）

## Demonstration with an example for BPE

让我用一个例子讨论上边算法过程：

$$
{'low': 5, 'lowest': 2, 'new: 6, 'widest': 3}
$$

假设我们根据corpus，对一个文本进行pre- tokenization之后，得到的一个结果。其中这个文本包含5个low，2个lowest，6个new，3个widest

### 初步分割：subword + 终止符号

在这个阶段，我们需要把集合内每一个单词分割成最小单元：chars，并且为了让算法知道每一个单词的结尾，我们需要在分割后添加一个终止符号 $</w>$。这个就是我们的 Vocabulary set

$$
{’l o w </w>’:5,’l o w e s t </w>’:2,’n e w </w>’:6,’w i d e s t </w>’:3}
$$

> 为什么要终止符号？举个例子，我们有两个单词 string 和 widest。这两个词中都包含字母组合 st，但在 string 中，st 出现在词首，而在 widest 中，st 位于词尾。它们在不同位置所表达的含义完全不同。 因此，我们必须通过添加终止符号，让算法能够区分这两种 st，从而正确理解单词的结构。
> 

### Building Subword Vocabulary Set

最原始的Subword Vocabulary Set应该包括所有在文本中出现过的字母加上最终符号：

$$
['l', 'o', 'w', '</w>', 'e', 's', 't', 'n', 'i', 'd']
$$

目前状态由于压缩比例太高，基本上无法表示相应的含义，我们必须通过BPE 算法不断个迭代和合并Subword Vocabulary Set，直到达到我们理想的词表规模。

### Merging and Iteration

在 Byte Pair Encoding算法中，下一步操作是识别Frequency Set中出现频率最高的相邻character pair，并将其合并为新的符号。该过程会以迭代的方式反复执行，直到达到预先设定的 token 数量上限或迭代次数上限为止。

通过不断地合并高频字符对，BPE 能够以最少的 token 数量对语料库进行有效表示，从而实现其核心目标——数据压缩。在每次迭代中，算法首先统计当前语料中各字符对的出现频率，选取最频繁的一对进行合并，并将新生成的 token 添加至词表中。由于合并会改变语料的组成，算法会在每次迭代后重新计算各 token 的出现频率。

这一合并与更新的过程将持续进行，直至达到预设的终止条件。接下来，我们将详细探讨每次迭代中具体发生的步骤。

### Step 1: calculating frequencies for each character pair

在算法的第一次迭代中，我们首先根据base vocabulary对原始词汇集合进行细粒度的分词处理。通过这一过程，可以将每个单词拆分为最基本的字符单元，并据此统计各字符的出现频率。

接着，我们进一步统计相邻character pair的共现频率，从而为后续的合并操作提供依据。具体结果如下所示。

Frequency Set: 

$$
('l', 'o'): 7, ('o', 'w'): 7, ('w', '</w>'): 11, ('w', 'e'): 2, ('e', 's'): 5, ('s', 't'): 5, ('t', '</w>'): 5, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3
$$

### Step 2: find the highest frequent character pair

在完成字符对频率的统计后，接下来的步骤是选择出现频率最高的字符对并进行合并。在我们case中，最高频率的是：

$$
('w', '</w>'): 11
$$

我们需要讲这个两个character合并，并且添加到Subword Vocabulary中。

$$
['l', 'o',  'w</w>', 'e', 's', 't', 'n', 'i', 'd','w'，'</w>']
$$

重新统计和更新frequency set：

$$
('l', 'o'): 7, ('o', 'w</w>'): 5, ('o', 'w'): 2, ('w', 'e'): 2, ('e', 's'): 5, ('s', 't'): 5, ('t', '</w>'): 5, ('n', 'e'): 6, ('e', 'w</w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3
$$

最后合并Vocabulary set：合并low</w>和new</w>

$$
\texttt{{'l o w</w>': 5, 'l o w e s t </w>': 2, 'n e w</w>': 6, 'w i d e s t </w>': 3}}
$$

### Step 3: second iteration

根据最新的frequency set，找the highest frequency pair：

$$
('l', 'o'): 7
$$

合并Subword Vocabulary集合：

$$
['lo',  'w</w>', 'e', 's', 't', 'n', 'i', 'd', 'w','</w>']
$$

更新frequency set：

$$
{('lo', 'w</w>'): 5, ('lo', 'w'): 2, ('w', 'e'): 2, ('e', 's'): 5, ('s', 't'): 5, ('t', '</w>'): 5, ('n', 'e'): 6, ('e', 'w</w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3}
$$

合并Vocabulary set： l 和 o

$$
\texttt{{'lo w</w>': 5, 'lo w e s t </w>': 2, 'n e w</w>': 6, 'w i d e s t </w>': 3}}
$$

下一个iteration就是同样的步骤：最高频率是 n 和 e 。最终我们得到的 Subword Vocabulary set 是：

$$
['est</w>', 'new</w>', 'low</w>', 'wid', 'lo', 'w']

$$

Vocabulary Set:

$$
\texttt{{'low</w>': 5, 'lo w est</w>': 2, 'new</w>': 6, 'wid est</w>': 3}}
$$

我们继续重复以上步骤，直到达到预设的词表规模或者满足迭代条件，或者下一个最高频的字符对出现频率为 1。 

## Implementation  of BPE

See Stanford university cs 336 assignment 1

## Other Algorithms

WordPiece, Mini BPE, BBPE, UNILM

## Papers You may like

Byte Latent Transformer: Patches Scale Better Than Tokens

The Future of AI: Exploring the Potential of Large Concept Models