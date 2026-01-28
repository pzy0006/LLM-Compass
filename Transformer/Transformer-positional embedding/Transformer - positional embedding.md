# Transformer - positional embedding

When users give an input to transformer, the transformer need to understand two things: 1) the meaning of the inputs and the ordering of the input. When dealing with sequence data, word order and position are huge for understanding meaning. The problem is, traditional word embeddings only capture what a word means, not where it is. On top of that, attention mechanism suffer from ‘permutation invariance’ - basically, they don’t naturally care about order.

That’s where positional encoding comes in. It assigns a unique position vector to each word. This bakes location information right into the model’s representation, fixing the order-blindness issue and helping the model truly understand the context and relationships between words.

> What is permutation invariance?
> 

Self=attention is the backbone of the Transformer. Unlike RNNs, which read on word at a time, self-attention looks at the input as a whole.

Because of this global view, it’s inherently permutation invariant. Meaning, the model doesn’t naturally care about order. You can shuffle the words around, and the attention weights - how strongly words relate to each other - won’t change at all. 

The calculation result for each word remains exactly the same. The only thing that changes is the row order in the output matrix to match the new shuffled input.

## **Code:**

```jsx
import torch.nn.functional as F
import torch
d = 8
l = 3
q = torch.randn(1,d)
k = torch.randn(l,d)
v = torch.randn(l, d)

attn = F.softmax(q @ k.transpose(1,0), dim=1) @ v

k_shift = k[[2,1,0],:]
v_shift = v[[2,1,0],:]
shift_attn = F.softmax(q @ k_shift.transpose(1,0),dim= 1) @ v_shift
print('before shifting: ', attn)
print('after shifting: ', shift_attn)
```

```jsx
before shifting:  tensor([[-1.2237,  0.8863, -0.4801,  0.9779, -1.6015,  0.1957,  0.3362,  0.2132]])
after shifting:  tensor([[-1.2237,  0.8863, -0.4801,  0.9779, -1.6015,  0.1957,  0.3362,  0.2132]])
```

## **Proof:**

The original matrices: 

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{X}\mathbf{W}^{(q)} \\
\mathbf{K} &= \mathbf{X}\mathbf{W}^{(k)}
\end{aligned}
$$

after permutation:

$$
\begin{aligned}
Q' &= \mathbf{P}_{\pi}\mathbf{X}\mathbf{W}^{(q)} = \mathbf{P}_{\pi}\mathbf{Q} \\
K' &= \mathbf{P}_{\pi}\mathbf{X}\mathbf{W}^{(k)} = \mathbf{P}_{\pi}\mathbf{K}
\end{aligned}
$$

Where ${P}_{\pi}$ is permutation matrix and only work on X matrix.

The original attention:

$$
\mathbf{A} = \frac{1}{\sqrt{d}}\mathbf{Q}\mathbf{K}^\top
$$

After permutation:

$$
\begin{aligned}
\mathbf{A}' &= \frac{1}{\sqrt{d}}\mathbf{Q}'(\mathbf{K}')^\top \\
&= \frac{1}{\sqrt{d}}(\mathbf{P}_{\pi}\mathbf{Q})(\mathbf{P}_{\pi}\mathbf{K})^\top \\
&= \frac{1}{\sqrt{d}}\mathbf{P}_{\pi}\mathbf{Q}\mathbf{K}^\top \mathbf{P}_{\pi}^\top \\
&= \mathbf{P}_{\pi}\mathbf{A}\mathbf{P}_{\pi}^\top
\end{aligned}
$$

The original softmax:

$$
SoftMax(A)_i = \frac{e^{A_i}}{\sum_j e^{A_j}}
$$

after permutation:

$$
SoftMax(A')_i = \frac{e^{(\mathbf{P}_{\pi}\mathbf{A}\mathbf{P}_{\pi}^\top)_i}}{\sum_j e^{(\mathbf{P}_{\pi}\mathbf{A}\mathbf{P}_{\pi}^\top)_j}}
$$

$$
SoftMax(\mathbf{P}_{\pi}\mathbf{A}\mathbf{P}_{\pi}^\top) = \mathbf{P}_{\pi}SoftMax(\mathbf{A})\mathbf{P}_{\pi}^\top
$$

$$
P P^T = P^T P = I 
$$

where $I$  is Identity Matrix.

This formula expresses the Permutation Equivariance of the Softmax function.
In simple terms, if you apply a permutation to the rows and columns of the input matrix $A$ (i.e., by multiplying by $P_\pi$ and $P_\pi^\top$) *before* applying Softmax, the result is equivalent to applying Softmax to $A$ *first* and then applying the same permutation to the output.
This implies that shuffling the graph structure or sequence order does not disrupt the relative relationships of the internal values within the Softmax; it merely changes their positions.

## **Solution:**

To solve this problem, the Transformer’s author choose one algorithm: sin position embedding. We need add position embedding (PE) to every word embedding:

$$
Input = \text{WordEmbedding} + \text{PositionalEmbedding}
$$

Even Dimension ($2i$)：
$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

Odd Dimension ($2i+1$)：
$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

### **Formula Interpretation**

The following points clarify the components and logic of the positional encoding formula:

**$d_{model}$ (Embedding Dimension):** This represents the dimension of the word vectors. In the paper, the positional encoding is added directly to the word embedding. Therefore, the dimension of the positional encoding must equal the dimension of the word embedding ($d_{model}$).

**$pos$ (Position Index):** This indicates the position of a token within the sequence. If the sentence length is $L$, the range of $pos$ is $0$ to $L-1$ (e.g., the first token is at position 0).

**$i$ (Dimension Index):** This represents the index of the dimension within the vector.
$2i$ represents even dimensions.
$2i+1$ represents odd dimensions.
*Example:* If $d_{model}$ is 512, the range of $i$ is $0 \sim 255$.

**$10000$ (Hyperparameter):** This is a scalar hyperparameter defined and used by the authors of the Transformer.

**Why Use Both Cos and Sin?**

This section explains the three main reasons for using alternating trigonometric functions.
**Distinguish Neighboring Words**: It makes it easier to distinguish words at close positions. Words that appear together often (like "I" and "Love") have similar semantic information and vector distances. To ensure these neighbors are distinguishable, the Transformer uses Sin for even dimensions and Cos for odd dimensions to differentiate their positions.

**Extend Periodicity**: Alternating Sin/Cos helps make the period length longer.

> Why longer period length is better?
> 

**Ensuring Uniqueness (Avoiding Repetition)**
The most critical reason for a long period is to ensure that every position in a sequence has a **unique** encoding.
**The Problem with Short Periods:** Trigonometric functions are cyclic ($\sin(x) = \sin(x + 2\pi)$). If the period is short (e.g., the wave repeats every 5 tokens), then Token #1 and Token #6 would have the exact same value in that dimension. The model would get "confused" and unable to distinguish which one came first.
**The Solution:** By extending the period length (up to $10000 \cdot 2\pi$), the Transformer ensures that the wave does not repeat within the typical length of a sentence or paragraph.
**Result:** Even for a very long sequence (e.g., 500 or 1000 words), the combination of these waves creates a unique "fingerprint" for every single position $0$ through $L-1$.

**Calculate Relative Positions (Main Reason)**: It allows the model to easily calculate relative positions. For a fixed distance $k$, $PE(pos+k)$ can be calculated linearly from $PE(pos)$.

This is based on the trigonometric identity: $\sin(A+B) = \sin(A)\cos(B) + \cos(A)\sin(B)$ 
This means both $\sin(x+k)$ and $\cos(x+k)$ can be expressed using the values of $\sin(x)$ and $\cos(x)$.
****

## A Better solution: RoPE - relative positional embedding

Relative positioning is superior because semantic meaning in natural language relies on the distance between words rather than their fixed coordinates, allowing models to generalize better to sequence lengths longer than those seen during training.

**RoPE** (Rotary Positional Embedding) is a modern positional encoding method that has become the standard for current Large Language Models (like LLaMA, PaLM, Qwen, and Mistral).

It was introduced to solve the limitations of absolute position encoding (like the Sinusoidal method in the original Transformer) by mathematically enforcing **relative position** awareness directly into the attention mechanism.

In traditional Transformers (like BERT or GPT-2), position information is added to the word vector:

$Vector_{final} = Vector_{word} + Vector_{position}$

In RoPE, position information is encoded by rotating the word vector in geometric space:
$Vector_{final} = Rotate(Vector_{word}, Angle_{position})$

Imagine every word vector as an arrow on a 2D plane.

- If a word is at position 1, RoPE rotates it by 10 degrees
- If a word is at position 2, RoPE rotates it by 20 degrees
- If a word is at position 3, RoPE rotates it by 30 degrees

**Formula**

Let’s say we have 2D vector:

$$
x = (x_1,x_2)
$$

$$
f(\boldsymbol{x}, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

**$m$**: The absolute position of the current token (e.g., the 5th word).

**$\theta$**: A preset base angle (frequency parameter, similar to $10000^{-2i/d}$ in sinusoidal encoding).

**Result**: The vector $\boldsymbol{x}$ is rotated counter-clockwise by an angle of $m\theta$ in the 2D plane.

**How do we have this?**

$\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$

**Proof**

**Step 1: Simplification and Definitions**
To simplify, let's represent **$pos$** as $t$.
For a specific frequency pair (dimension index $i$), the positional encoding vector at position $t$ is:

$\begin{aligned}
PE_{t, 2i} &= \sin\left(\frac{t}{1000^{2i/d}}\right) \\
PE_{t, 2i+1} &= \cos\left(\frac{t}{1000^{2i/d}}\right)
\end{aligned}$

**Step 2: Target Position ($t+k$)**

Now, consider the position $t+k$. We want to find $PE_{t+k}$.

The formulas for the new position are:

$\begin{aligned}
PE_{t+k, 0} &= \sin\left(\frac{t+k}{1000^{2i/d}}\right) \\
PE_{t+k, 1} &= \cos\left(\frac{t+k}{1000^{2i/d}}\right)
\end{aligned}$

**Step 3: Define the Relative Shift $\alpha_k$**
Let's define a term `$\alpha_k$` that represents the frequency shift caused by distance

 $\alpha_k = \frac{k}{1000^{2i/d}}$
****

**Step 4: Expansion using Trigonometric Identities**
Now, expand the formulas for position $t+k$ using the sum-to-product identities.
**For the Sine component ($PE_{t+k, 0}$):**

$\begin{aligned}
PE_{t+k, 0} &= \sin\left(\frac{t}{1000^{2i/d}} + \alpha_k\right) \\
&= \sin\left(\frac{t}{1000^{2i/d}}\right)\cos(\alpha_k) + \cos\left(\frac{t}{1000^{2i/d}}\right)\sin(\alpha_k) \\
&= PE_{t, 0} \cdot \cos(\alpha_k) + PE_{t, 1} \cdot \sin(\alpha_k)
\end{aligned}$

**For the Cosine component ($PE_{t+k, 1}$):**

$\begin{aligned}
PE_{t+k, 1} &= \cos\left(\frac{t}{1000^{2i/d}} + \alpha_k\right) \\
&= \cos\left(\frac{t}{1000^{2i/d}}\right)\cos(\alpha_k) - \sin\left(\frac{t}{1000^{2i/d}}\right)\sin(\alpha_k) \\
&= PE_{t, 1} \cdot \cos(\alpha_k) - PE_{t, 0} \cdot \sin(\alpha_k)
\end{aligned}$

**Step 5: Matrix Representation (Linear Transformation)**

We can rewrite the two equations above as a matrix multiplication. This shows that $PE_{t+k}$ is simply $PE_t$ multiplied by a rotation matrix that depends only on $k$.

$\begin{bmatrix} PE_{t+k, 0} \\ PE_{t+k, 1} \end{bmatrix} = \begin{bmatrix} \cos \alpha_k & \sin \alpha_k \\ -\sin \alpha_k & \cos \alpha_k \end{bmatrix} \begin{bmatrix} PE_{t, 0} \\ PE_{t, 1} \end{bmatrix}$

## Mathematical Derivation of RoPE: Why Rotation Equals Relative Position

**1. Setup and Definitions**
First, we define the standard Query and Key vectors at positions m and n before position encoding is applied:
• Let $\mathbf{q}_m$ be the query vector at position m.
• Let $\mathbf{k}_n$ be the key vector at position n.
The RoPE Operation:
RoPE treats the position embedding as a rotation in a 2D plane.
• The vector at position m is rotated by angle $m\theta$.
• The vector at position n is rotated by angle $n\theta$.
The 2D rotation matrix is defined as:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

So, the encoded vectors become:

$$
\mathbf{q}_m^{RoPE} = R(m\theta)\mathbf{q}_m \\
\mathbf{k}_n^{RoPE} = R(n\theta)\mathbf{k}_n
$$

The dot product of two column vectors $\mathbf{a} \cdot \mathbf{b}$ can be written as matrix multiplication $\mathbf{a}^T \mathbf{b}$.
Substitute the RoPE vectors into the equation:

$$
\langle \mathbf{q}_m^{RoPE}, \mathbf{k}_n^{RoPE} \rangle = (\mathbf{q}_m^{RoPE})^T \cdot \mathbf{k}_n^{RoPE}
$$

Substitute the rotation definitions:

$$
= (R(m\theta)\mathbf{q}_m)^T \cdot (R(n\theta)\mathbf{k}_n)
$$

In linear algebra, the transpose of a product is the product of the transposes in reverse order: $(AB)^T = B^T A^T$.Applying this to the first term:

$$
= \mathbf{q}_m^T \cdot R(m\theta)^T \cdot R(n\theta) \cdot \mathbf{k}_n
$$

This is the crucial step mentioned in the image (`where R(θ)^T = R(-θ)`).
A rotation matrix is orthogonal. Its transpose is its inverse, which geometrically corresponds to rotating by the **negative angle**:

$$
R(m\theta)^T = R(-m\theta)
$$

Substitute this back into the equation:

$$
= \mathbf{q}m^T \cdot \underbrace{R(-m\theta) \cdot R(n\theta)}{\text{Combine Rotations}} \cdot \mathbf{k}_n
$$

When you perform two 2D rotations sequentially, you simply sum the angles.
Rotating by $-m\theta$ and then by $n\theta$ is equivalent to a single rotation of $(n\theta - m\theta)$:

$$
R(-m\theta) \cdot R(n\theta) = R(n\theta - m\theta)
$$

Substituting the combined rotation back into the equation, we get the final formula highlighted in the image:

$$
\langle \mathbf{q}_m^{RoPE}, \mathbf{k}_n^{RoPE} \rangle = \mathbf{q}_m^T \cdot R((n-m)\theta) \cdot \mathbf{k}_n
$$