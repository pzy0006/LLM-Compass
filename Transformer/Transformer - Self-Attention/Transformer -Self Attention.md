# Transformer -Self Attention

Over the pastt decade, sequence modeling ahs been dominated by neural architectures such as RNNs and CNNs. While these models achieved notable success in tasks like machine translation and speech recognition, they suffer from inherent limitations: difficulty modeling long-range dependencies, limited parallelism, and increasing computational cost as sequence length grows.

The introduction of the Transformer architecture marked a fundamental shift in how sequence data is processed. Unlike previous models, Transformers completely abandon recurrence and convolution(not saying they are not useful anymore), replying instread on a mechanism known as self-attention. This seemingly simple idea - allowing each element in a sequence to directly attend to all other elements - has proven powerful enough to redefine the state of the art across natural language processing, computer vision, and beyond.

at its core, self-attention answers a basic but crucial questions:

> When processing a token in sequence, which other tokens should it pay attention to, and to what extent?
> 

Rather than compressing information through a fixed-size hidden state (as in RNNs) or local receptive fields (as in CNNs), self-attention enables a token to dynamically aggregate information from the entire sequence based on learned relevance scores. This design allows the model to capture both short-range and long-range dependencies in a single operation.

Another key advantage of self-attention is **parallelism**. Since attention does not depend on sequential computation, all tokens in a sequence can be processed simultaneously. This property not only accelerates training on modern hardware but also scales more effectively to large datasets and long sequences—one of the primary reasons Transformers have become the foundation of large language models such as BERT, GPT, and LLaMA.

In this article, we will take a systematic look at self-attention in Transformers. We will begin by building intuition around the mechanism, then break down its mathematical formulation, and finally discuss why it is so effective in practice. The goal is not only to explain *how* self-attention works, but also *why* it works—and why it has become the cornerstone of modern deep learning architectures.

## Intuition Behind Self-Attention

To understand self-attention, it is useful to step away from equations and htink about how humans process sequences, especially language.

Consider this:

> The animal didn’t cross the street because it was to tired
> 

when encountering the word “it”, a human reader instinctively understands that “it:” refers to the “animal”, not “the street”. This understanding does not come from reading the sentence strictly left to right, nor from remembering everything in a fixed memory slot. Instead, the reader selectively focuses on relevant parts of the sentence while interpreting the current word.

Self-attention is a computational mechanism designed to mimic this behavior.

### From Sequential Processing to Selective Focus

Traditional sequence models such as RNNs process tokens one at a time, accumulating information into a hidden state. While this approach preserves order, it forces the model to compress all past information into a single vector. As sequences grow longer, important details may fade or become diluted.

Self-attention takes a different approach. Instead of asking *“What information have I remembered so far?”*, it asks:

> Given the current token, which other tokens in the sequence are most relevant right now?
> 

Crucially, this question is asked **independently for every token** in the sequence.

Each token is allowed to:

- Look at all other tokens
- Decide how much attention to assign to each
- Combine their information into a new, context-aware representation

This is why the mechanism is called *self*-attention: the attention is computed **within the same sequence**, not between different sequences (as in encoder–decoder attention).

### Attention as a Relevance Scoring Problem

At an intuitive level, self-attention can be framed as a relevance scoring task.

For a given token, the model needs to determine:

- Which tokens matter?
- How strongly should each one influence the current representation?

To do this, self-attention introduces three conceptual roles for each token:

- one representation that *asks* what it is looking for,
- one that *describes* what the token contains,
- and one that *provides* the information to be aggregated.

These roles are later formalized as **Query (Q)**, **Key (K)**, and **Value (V)**.

For now, it is sufficient to think of them as different perspectives of the same token, each learned through linear transformations.

The interaction between queries and keys produces attention scores, which measure how relevant one token is to another. These scores are then used to weight the values, allowing the model to construct a context-dependent representation.

### Why This Design Is Powerful

This design offers several important advantages:

1. **Direct modeling of long-range dependencies**
    
    Any token can attend to any other token in a single step, regardless of their distance in the sequence.
    
2. **Dynamic, context-dependent representations**
    
    The meaning of a token is not fixed. Its representation changes depending on which other tokens it attends to.
    
3. **Interpretability**
    
    Attention weights provide a transparent signal of which parts of the input influence a given output, offering insights into model behavior.
    
4. **Parallel computation**
    
    Since attention for all tokens can be computed simultaneously, self-attention is well-suited for modern parallel hardware.
    

## Query, Key, and Value: Formalizing Self-Attention

In the previous section, we described self-attention as a relevance-based information aggregation mechanism. To turn this intuition into a computable operation, Transformers introduce a structured representation of each token using three distinct vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.

Although these vectors are often presented together, they serve different purposes and should be understood as playing complementary roles in the attention process.

### Why Do We Need Three Representations?

At first glance, it may seem unnecessary to represent each token three times. Why not compute attention directly on the token embeddings themselves?

The reason lies in **decoupling roles**.

For each token in a sequence, the model must answer three different questions:

1. **What am I looking for?**
    
    This is captured by the **Query**.
    
2. **What do I contain, and how should others match against me?**
    
    This is captured by the **Key**.
    
3. **What information should I contribute if I am deemed relevant?**
    
    This is captured by the **Value**.
    

A helpful way to understand the roles of Query, Key, and Value is to imagine a person searching for a book in a large library.

Suppose you walk into the library with a specific goal in mind—say, you want to find a book about modern neural networks. This intention corresponds to a **Query**. The query represents *what you are looking for*, not the books themselves.

Each book in the library, on the other hand, has a label on its spine describing its topic, author, or category. These labels serve as **Keys**. You do not read the entire content of every book to decide whether it is relevant; instead, you compare your search intent with these labels to determine which books are worth pulling from the shelf.

Once you identify the most relevant books, you open them and read their actual content. This content corresponds to the **Values**—the information you ultimately care about and want to extract.

By separating these roles, the model gains the flexibility to learn *how* relevance is measured independently from *what* information is ultimately exchanged.

### Constructing Q, K, and V

Let the input sequence be represented as a matrix

$X \in \mathbb{R}^{n \times d_{\text{model}}}$

where n is the sequence length and $d_{\text{model}}$ is the embedding dimension.

Self-attention applies three learned linear transformations to this input:

$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$

Here:

- $W_Q, W_K, W_V$ are trainable projection matrices
- All tokens share the same projection parameters
- Each projection emphasizes different aspects of the input representation

This means that **Q, K, and V originate from the same input**, but are shaped to serve different functional roles in attention.

### Queries and Keys: Measuring Relevance

The interaction between **queries** and **keys** determines *where attention is placed*.

For a given token i, its query vector $q_i$is compared with the key vectors $k_j$of all tokens j in the sequence. This comparison produces a scalar score that reflects how relevant token j is to token i.

Conceptually:

- A high score indicates strong relevance
- A low score indicates weak or negligible relevance

These scores form an **attention distribution**, expressing how the current token allocates its focus across the sequence.

At this stage, no information has yet been combined—only relevance has been evaluated.

### Values: Aggregating Information

Once relevance has been established, **values** come into play.

Each value vector represents the information content that a token can contribute. The attention scores computed from queries and keys are used to weight these value vectors, producing a weighted sum.

As a result:

- Relevant tokens contribute more strongly
- Irrelevant tokens contribute little or nothing

The output for each token is therefore a **context-aware representation**, constructed from other tokens in the sequence based on learned relevance patterns.

Putting everything together, self-attention can be summarized as:

> Each token generates a query to search for relevant keys in the sequence, and then aggregates the corresponding values according to the resulting relevance scores.
> 

This formulation bridges the intuitive idea of selective focus with a concrete, differentiable computation.

## Scaled Dot-Product Attention

With Queries, Keys, and Values in place, we can now describe how self-attention is computed in practice. The mechanism used in Transformers is known as **scaled dot-product attention**, a simple yet effective formulation that transforms relevance comparisons into differentiable operations.

### Computing Attention Scores

The first step in self-attention is to measure how well each query matches each key.

Given:

- a query matrix $Q∈\mathbb{R}^{n \times d_k}$
- a key matrix $K\in \mathbb{R}^{n \times d_k}$,
- and a value matrix $V \in \mathbb{R}^{n \times d_v}$,

we compute a matrix of raw attention scores by taking the dot product between queries and keys:

$$
Scores = QK^T
$$

Each element$\text{Scores}_{ij}$ reflects how relevant token j is to token i. A larger dot product indicates a stronger alignment between the query of token i and the key of token j.

This operation is efficient, fully parallelizable, and well-suited for matrix-based hardware accelerators.

### Scaling the Scores

As the dimensionality dkd_kdk increases, the magnitude of dot products tends to grow. This can lead to extremely large values in the score matrix, which in turn causes problems during normalization.

To address this, the dot products are scaled by the inverse square root of the key dimension:

$$
ScaledScores=QK^T/{sqrt(d_k)}
$$

This scaling stabilizes the distribution of scores, preventing them from becoming too large and ensuring smoother gradients during training.

From a practical perspective, scaling improves numerical stability and accelerates convergence without changing the expressive power of the attention mechanism.

### Normalizing with Softmax

Raw or scaled scores alone are not sufficient—they must be converted into weights that sum to one.

This is achieved using the **softmax** function applied row-wise:

$$
AttentionWeights=softmax(QK^T/sqrt(d_k))
$$

The resulting attention weights can be interpreted as a probability distribution over the sequence:

- higher weights indicate stronger attention,
- lower weights indicate weaker attention.

Softmax also introduces non-linearity, allowing the model to sharply focus on a small subset of relevant tokens when necessary.

> Why Do We Need Softmax?
> 

After computing the scaled dot-product scores, we obtain a matrix of real-valued numbers that reflect pairwise relevance between tokens. However, these raw scores alone are not yet suitable for information aggregation.

There are three fundamental reasons why an additional normalization step is required.

First, **raw attention scores are unbounded**. Dot products can take arbitrary positive or negative values depending on vector magnitudes and alignment. Without normalization, these scores cannot be directly interpreted as weights in a weighted sum.

Second, **attention should represent relative importance**, not absolute magnitude. What matters is not how large a score is in isolation, but how it compares to the scores of other tokens within the same context. A normalization function allows the model to express preference among tokens rather than relying on raw scale.

Third, **stable training requires controlled gradients**. If one or more scores dominate numerically, gradients may become unstable, slowing convergence or leading to poor optimization behavior.

The softmax function addresses all three issues simultaneously.

> Softmax equation
> 

Softmax transforms a vector of real-valued scores into a probability distribution by exponentiating and normalizing them:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

![image.png](Transformer%20-Self%20Attention/image.png)

### Aggregating Values

The final step is to combine information using the attention weights.

Each output representation is computed as a weighted sum of value vectors:

$$
Output
= \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
 * V
$$

Despite its compact form, this formula captures the entire logic of self-attention: relevance measurement, normalization, and information aggregation.

## Multi-Head Attention

So far, we have described self-attention as a single operation that produces one context-aware representation per token. While this formulation is powerful, it has an important limitation: a single attention distribution may be insufficient to capture the diverse relationships present in complex sequences.

**Multi-head attention** addresses this limitation by allowing the model to attend to the sequence from multiple perspectives simultaneously.

### Why One Attention Is Not Enough

In natural language, different types of relationships often coexist within the same sentence:

- syntactic dependencies (e.g., subject–verb agreement),
- semantic associations (e.g., coreference),
- positional or structural patterns.

If a model is restricted to a single attention mechanism, it must compress all these relationships into one relevance pattern. This can lead to overly coarse or entangled representations.

Multi-head attention solves this by asking a simple question:

> What if the model could look at the same sequence in different ways at the same time?
> 

### Splitting Attention into Multiple Heads

Instead of performing attention once with large vectors, multi-head attention performs attention **multiple times in parallel**, each time in a lower-dimensional subspace.

Formally, the input representation X is projected into multiple sets of queries, keys, and values:

$$
Q_h=XW_{Q}^{(h)},K_h=XW_{K}^{(h)},V_h=XW_{V}^{(h)}
$$

where h indexes the attention head.

Each head computes its own scaled dot-product attention:

$$
head_h=Attention(Q_h,K_h,V_h)
$$

Because each head has its own projection matrices, it can learn to focus on different types of relationships.

### Concatenation and Projection

After all heads have produced their outputs, these outputs are concatenated:

$$
Concat(head_1,…,head_H)
$$

This concatenated representation is then passed through a final linear projection:

$$
MultiHead(Q,K,V)=Concat(head1,…,headH)W_O
$$

This step serves two purposes:

- it recombines information from different attention heads,
- and it maps the result back to the original model dimension.

### What Do Different Heads Learn?

Although all heads operate on the same input sequence, they often specialize during training. Empirically, different heads have been observed to focus on:

- local versus long-range dependencies,
- positional patterns,
- syntactic roles,
- semantic similarity.

Importantly, this specialization is **emergent**. No explicit constraints force a head to learn a particular behavior; diversity arises naturally from optimization.

### **Deep Dive: The Mathematics of Multi-Head Attention**

**An anatomical dissection of the matrix operations inside the Transformer.**
The Multi-Head Attention (MHA) mechanism is the engine of modern Large Language Models. While the concept is widely understood at a high level, the specific matrix transformations often cause confusion.
This post provides a rigorous, step-by-step calculation process, specifically addressing the dimensionality changes and the interaction between Queries ($Q$) and Keys ($K$).

Let us define the hyperparameters for a standard Transformer layer (e.g., similar to BERT-base or GPT-2 small):
• **$N$**: Sequence length (e.g., number of tokens in the input).
• **$d_{model}$**: The embedding dimension of the model (e.g., 512).
• **$h$**: The number of attention heads (e.g., 8).
• **$d_k$**: The dimension of each head. Typically $d_k = d_{model} / h$ (e.g., $512 / 8 = 64$).
Input:
Let the input to the attention layer be a matrix $\mathbf{X} \in \mathbb{R}^{N \times d_{model}}$

A common misconception is that MHA slices the input vector. In reality, MHA performs distinct linear transformations for every head.
For a specific head $i$ (where $i \in \{1, \dots, h\}$), we have three unique, learnable weight matrices:
1. $\mathbf{W}_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
2. $\mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$
3. $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_k}$
We calculate the Query, Key, and Value matrices for head $i$:

$$
\mathbf{Q}_i = \mathbf{X}\mathbf{W}_i^Q
$$

$$
\mathbf{K}_i = \mathbf{X}\mathbf{W}_i^K
$$

$$
\mathbf{V}_i = \mathbf{X}\mathbf{W}_i^V
$$

Resulting Dimensions:

$$
\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i \in \mathbb{R}^{N \times d_k}
$$

**Note:** Although $\mathbf{X}$ is the source for all three, $\mathbf{Q}_i$ and $\mathbf{K}_i$ are projected into different subspaces. This orthogonality is crucial for the attention mechanism to function.

Now, we calculate the similarity scores between tokens. This is the core of the bottleneck.

**The Dot Product**

We multiply the Query matrix by the transpose of the Key matrix: $\text{Scores}_i = \mathbf{Q}_i (\mathbf{K}_i)^T$
Dimensional Analysis:

$$
[N \times d_k] \cdot [d_k \times N] \rightarrow [N \times N]
$$

The resulting matrix is an $N \times N$ grid where the entry at index $(t, j)$ represents the raw affinity between the token at position $t$ and the token at position $j$.

**The Diagonal Fallacy (Crucial Insight)**
A frequent question is: **"Is the diagonal value (where $t=j$) always the largest?"**
The entry at the diagonal $(t, t)$ is the dot product of the token's query vector and its own key vector:

$$
\text{Score}_{t,t} = \mathbf{q}_t \cdot \mathbf{k}_t = (\mathbf{x}_t \mathbf{W}_i^Q) \cdot (\mathbf{x}_t \mathbf{W}_i^K)
$$

Since $\mathbf{W}_i^Q$ and $\mathbf{W}_i^K$ are independent parameters learned via gradient descent:

- They map $\mathbf{x}_t$ to different vectors.
- $q_t$ and $\mathbf{k}_t$ are generally **not** collinear.

Therefore, $\mathbf{q}_t \cdot \mathbf{k}_t$ is **not guaranteed to be the maximum value** in the row. In fact, for a model to learn context (e.g., connecting "Apple" to "eat"), the off-diagonal score ($\mathbf{q}_{apple} \cdot \mathbf{k}_{eat}$) *must* be higher than the diagonal score.
****

To prevent vanishing gradients in the Softmax function, we scale by $\sqrt{d_k}$:

$$
\mathbf{A}_i = \text{softmax}\left( \frac{\mathbf{Q}_i (\mathbf{K}_i)^T}{\sqrt{d_k}} \right)
$$

**$\mathbf{A}_i$** is the Attention Weight Matrix ($\in \mathbb{R}^{N \times N}$), where every row sums to 1.

We now use the attention weights to compute the weighted sum of the Value vectors ($\mathbf{V}_i$).

$$
\mathbf{Head}_i = \mathbf{A}_i \mathbf{V}_i
$$

Dimensional Analysis:

$$
[N \times N] \cdot [N \times d_k] \rightarrow [N \times d_k]
$$

At this stage, the vector for a specific token is no longer just its original embedding; it is a mixture of the vectors of all related tokens in the sequence, weighted by their relevance.

We computed the above steps for all $h$ heads in parallel. Now we merge them.
****

**Concatenation**
We define the Multi-Head output by concatenating the individual heads along the feature dimension:

$$
\mathbf{H}_{concat} = \text{Concat}(\mathbf{Head}_1, \dots, \mathbf{Head}_h)
$$

Dimensional Analysis:
Since each head is $\mathbb{R}^{N \times d_k}$ and we have $h$ heads:

$$
\mathbf{H}_{concat} \in \mathbb{R}^{N \times (h \cdot d_k)} = \mathbb{R}^{N \times d_{model}}
$$

(Assuming $h \cdot d_k = d_{model}$)
****

**Final Linear Projection**
Finally, we pass this concatenated matrix through an output weight matrix 

$$
\mathbf{W}^O \in \mathbb{R}^{d_{model} \times d_{model}}
$$

$$
\text{MultiHeadOutput} = \mathbf{H}_{concat} \mathbf{W}^O
$$

Final Output Dimension:

$$
\mathbb{R}^{N \times d_{model}}
$$

| **Tensor / Variable** | **Shape** | **Description** |
| --- | --- | --- |
| **Input X** | `[10, 512]` | Raw Token Embeddings |
| **Weights** $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ | `[512, 64]` | Per-head projection matrices (x8) |
| **Q, K, V** | `[10, 64]` | Projected vectors per head |
| **Scores** ($QK^T$) | `[10, 10]` | Similarity grid (Attention Map) |
| **Head Output** ($AV$) | `[10, 64]` | Contextualized vectors per head |
| **Concat** | `[10, 512]` | Rejoined heads |
| **Final Output** | `[10, 512]` | Mixed output ready for Feed-Forward Network |

## Papers about Multi head Attention modifications(improvements)

1: Improving Vision Transformers by Overlapping Heads in Multi-Head Self-Attention

2: MoH: Multi-Head Attention as Mixture-of-Head Attention

3: Improving Transformers with Dynamically Composable Multi-Head Attention