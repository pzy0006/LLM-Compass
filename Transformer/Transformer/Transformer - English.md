# Transformer - English

![image.png](Transformer%20-%20English/image.png)

The traditional Transformer architecture consists of two main components: the Encoder on the left and the Decoder on the right. In practical implementations, both the encoder and decoder are usually stacked multiple times. For example, a typical Transformer may contain six encoder layers and six decoder layers , each building upon the previous one.

# Encoder Inputs

Encoder is mainly responsible for mapping input sequences (words / tokens) into semantically rich contextual representations.

In fact, Encoder inputs are composed of two parts: embedding + positional Encoding.

## Embedding

The whole process of embedding is to transform some human-readable language into computer-readable numbers. It is similar to the Java code encoding process, from code to byte code.

> Before entering the embedding stage, we first need to perform tokenization, which means breaking the raw text into smaller units called tokens.
> 
> 
> In English, the simplest approach is to split words based on spaces and treat punctuation marks as separate tokens.
> 
> However, this naive method is not ideal in practice, because:
> 
> - the vocabulary can become extremely large, and
> - it struggles to handle rare or unseen words (OOV, out-of-vocabulary).
> 
> To address these issues, modern language models use subword segmentation methods such as BPE (Byte Pair Encoding) and WordPiece.
> 
> These techniques break words into smaller subword units, reducing the OOV problem while keeping the vocabulary size manageable.Andrej Karpathy's video is highly recommended: https: https://www.youtube.com/watch?v=zduSFxRajkE&t=5200s
> 

> After tokenization, we obtain a sequence of discrete tokens.However, neural networks cannot directly understand symbolic data, so each token must be converted into a corresponding integer index.
> 
> 
> To achieve this, we construct a vocabulary that contains all tokens allowed in the model, assigning each token a unique integer ID.
> 
> This process maps the raw text into a sequence of numbers that the model can process.
> 
>  For example, the sentence "I love cream crackers" may become: ["I": 101, "love": 102, "cream": 103, "crackers": 104] after the vocabulary mapping.
> 

### Step 1:

 After getting a string of integer IDs like [101, 102 ,103, 104]. It is time to convert them into dense vector representation through embedding layer. Each token id is the index of one row of the matrix. For example, if the embedding matrix is 100,000 * 512, and ID = 101, then we take the 101st row of the matrix, and get a vector with 512 dimensions.

![image.png](Transformer%20-%20English/image%201.png)

 The above steps are called embedding lookup, and the whole process takes place in the "input embedding" of the transformer's original paper.

> Quick Review: The larger the embedding dimension, the more information the model has the potential to capture.However, bigger is not always better.As the dimensionality increases, the number of parameters in both the embedding matrix and subsequent Transformer layers grows dramatically, leading to higher training and inference costs.Moreover, if the training dataset is not large enough, a high-dimensional embedding may cause the model to learn noise instead of generalizable features
> 

 We get these vectors looking like this, which I'll show you here in 2D

![image.png](Transformer%20-%20English/image%202.png)

After the embedding step, words with similar meanings are represented by similar vectors.

For example:

- The vectors for “Apple” and “Banana” are close to each other because both are fruits.
- The vectors for “BMW” and “Toyota” are also similar because they represent car brands.

In other words, semantically related words are located near each other in the embedding space.

The final output of this step is a set of continuous floating-point vectors, typically with hundreds or thousands of dimensions.

For instance, if we use a simple tokenization algorithm, the sentence: “I love eating apples and bananas.”
might be transformed into something like this (simplified illustration):

| Token | Embedding (simplified example) |
| --- | --- |
| I | [0.12, -0.05, 0.33, ...] |
| love | [0.51, 0.22, -0.11, ...] |
| eating | [0.09, 0.47, -0.02, ...] |
| apples | [0.62, 0.13, 0.77, ...] |
| and | [-0.03, 0.08, 0.04, ...] |
| bananas | [0.60, 0.15, 0.75, ...] |

As you can see, the vectors for “apples” and “bananas” are numerically close, reflecting their semantic similarity.

![image.png](Transformer%20-%20English/image%203.png)

The final output of this step is a simple matrxi containing the vectors for each word, assuming on the graph that each vector has a dimension of 512. (The vector dimensions can be other numbers.)

So can we get the vectors for each word and send them directly to the transformer for training? The answer is no!

Transformer does not know the location information of each word. For example, if you type: I love cream crackers and you type: love cream I crackers, it's the same for the transformer. We have to figure out a way to make the Transformer understand that there is a positional relationship between each word. See Step2

> If you forget RNN, you can review it.
> 

### Step2 : Positional Embedding

 In Transformer paper, the author used Absolute Positional Encoding.

![Screenshot 2025-09-22 at 1.34.29 PM.png](Transformer%20-%20English/Screenshot_2025-09-22_at_1.34.29_PM.png)

**The meaning of each symbol:**

- **`d`**: dimension of embedding.
    - For example, the input embedding of BERT-base is 768 dimensions; if you set `d = 512`, it is a 512 dimensional vector.
- **`i`**: the index of the dimension.
    - Because the formula uses sin and cos for even and odd dimensions, `i` usually takes the range `0,1,2,...,d/2-1`.
    - As an example:
        - `i = 0` → compute the 0th and 1st dimensions (sin/cos pair).
        - `i = 210` → compute the 420th and 421st dimensions.
- **`pos`**: index of the position of the token (the first token), e.g. word 0, word 1 ......
- **Result**: each position `pos` ends up with a position vector of length `d`, alternately filled with sin and cos of different frequencies.

> In addition to absolute position coding, we also have learnable position embedding and relative position embedding. I will compare these three algorithms later, but I want to focus on the RoPE relative position embedding algorithm.
> 

 When we calculate the absolute position information of each vector after applying above formula:

$$
P=[p_1,p_2,...,p_n],p_i∈R^d
$$

 It will be compared to the input embedding:

$$
X=[x_1,x_2,...,x_n],x_i∈R^d
$$

 We will add it to the input embedding:

$$
Z=X+P
$$

 We get the final representation of the sequence

$$
 Z=[z_1,z_2,...,z_n]
$$

 Each of these

$$
z_i=x_i+p_i
$$

Calculated by various position encoding algorithms, our real input is actually the vector Z. The size of the vector Z is:

$$
Z = (B×N×D)
$$

 where:

- **batch size** = B: the **number of samples** fed into the model for training/inference at one time.
- **sequence length** = N: **number of tokens** in each sample (i.e. sentence length)
- **embedding dimension** = D: the length of the vector representation of each token (i.e., the feature dimension of each token).

![image.png](Transformer%20-%20English/image%204.png)

> The steps above demonstrate how positional information vectors are added to the original word embeddings — essentially, it’s a process of weighted summation between the two types of vectors.
> 
> 
> However, this direct addition has a potential drawback: the positional information may interfere with or even distort the semantic meaning carried by the original embeddings.
> 
> Therefore, we need a more sophisticated method that can preserve the semantic integrity of the original vectors while effectively incorporating positional information.
> 
> I will discuss this in detail in a later article about the**RoPE (Rotary Position Embedding)**
> 
> algorithm, which was specifically designed to address this issue.
> 

### Step 3

This step is the main focus. In the attention mechanism we will divide Z into three parts Q, K and V (note that the size of their three matrices is the same). These three matrices contain different information and serve different purposes.

![image.png](Transformer%20-%20English/image%205.png)

 Inside the multi head attention we first weight Z so that we get the corresponding Q, K and V. In this case, the weights of the three matrices are the same as in the previous one.

$$
Q=ZW_Q,K=ZW_K,V=ZW_V
$$

 Among them:

$$
W_q, W_v, W_k
$$

 is learnable (i.e., the three matrices keep changing during the training process to find an optimal matrix.) Weight Matrix.

$$
Q,K,V
$$

 is the **Query**, **Key**, and **Value** matrix.

 

**Q** represents the vector that is currently making the “query” — what the model wants to know.

**K** represents the “feature tag” or identifier of the current token.

**V** represents the actual information carried by that token.

> The transformation of the matrix from Z to Q, K and V is a linear transformation.
> 

### Step 3.1: Attention Score

 In this step, we focus on how Q, K and V interact with each other and what information we can get from them. The original paper uses multi head attention, so I will use single head attention as an example to see how the whole calculation process looks like.

$$
\begin{equation}\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V\end{equation}
$$

 The above equation shows how we calculate attention.

![image.png](Transformer%20-%20English/image%206.png)

 Example: ["I", "love", "cream", "crackers"]. For each word, there are q,k and v, their size is 1*5, where 5 is the vector dimension.

![image.png](Transformer%20-%20English/image%207.png)

![image.png](Transformer%20-%20English/image%208.png)

 The equation above corresponds to

$$
Q*K^T
$$

 This means that the $q$ of any word is multiplied by the $k$ of all words (including your own). For example:

 The $q$ of the word "I" is multiplied by the $q$ of every word ["I", "love", "cream", "crackers"]. "crackers"].

$$
[q_1*k_1, q_1*k_2,q_1*k_3,q_1*k_4]
$$

 The $q$ of the word "love" is multiplied by the $q$ of each word ["I", "love", "cream", "crackers"]. ", "crackers"].

$$
[q_2*k_1, q_2*k_2,q_2*k_3,q_2*k_4]
$$

 And so on.

 So why are we doing this math? It actually makes mathematical sense here:

 When we calculate

$$
Q*K^T
$$

 the inner product of each Query and all the Keys.

- The larger the inner product, the better the match between the Query and the Key, and the stronger the correlation.
- The smaller the inner product is, or even negative, it means that the Query is not interested in that Key.

 In other words, the degree that the $i_{th}$ token wants to "pay attention" to the $j_{th}$ token, which is what we call **Attention Score.**

### Step 3.2: Value Scaling

 In the attention mechanism, we divide $Q*K^T$ by $\sqrt{d}$ where d is the dimensinality of the vectors( in the multi-attention mechanism, it should be $d_k$ here, we will talk about it later.)

 In the process of multiplying Q and K, some of the values in the matrix will become very large. In order to keep the numbers within a certain range and maintain numerical stability, we scale the dot product by dividing it by $\sqrt{d}$ . This  prevents the inner product values from growing excessively large, which helps stabilize the softmax operation that follows.

### Step 3.3: Softmax

 Let's look at the formula first:

$$
\begin{equation}\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{j'} \exp(s_{ij'})}\end{equation}
$$

 Where:

- $s_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d}}$ the correlation score of token i to token j
- $\alpha_{ij}$ is the attention weight.

 Once you have calculated the attention score of a word for all words and scaled it down (divided by sqrt(d)), these values are only relative similarities and do not directly represent the attention distribution. softmax converts these attention scores into a set of attention weights that add up to 1, thus making it clear that each word should have an attention score of 1. This process makes it explicit how much each token should *attend to* (or focus on) every other token in the sequence.

Softmax does this by **converting a set of scores into a probability distribution**:

- Each $α_{ij}∈(0,1)$ a normal distribution, where each attention weight is like a "probability", with no negative weights or weights greater than one.
- And  $∑α_{ij}=1$: guarantees that all the weights together are 100%.

 For example, the Query for "love" gives [3.2, 0.8, -5.7]:

- Softmax result = [0.71, 0.28, 0.01].
- Meaning: love 70% focuses on "I", 28% focuses on itself, 1% focuses on cream, and hardly looks at crackers.

![image.png](Transformer%20-%20English/image%209.png)

### Step 3.3

With Step 3.2, we obtain a weight matrix that represents the attention of each word to all words.

Assuming that the final output is $Z$, the weight matrix is $A$, and there is a matrix $V$. (Here the size of the V matrix is 4*2 for ease of calculation)

$$
Z =\begin{bmatrix}0.7 & 0.1 & 0.1 & 0.1 \\0.2 & 0.5 & 0.2 & 0.1 \\0.25 & 0.25 & 0.25 & 0.25 \\0.4 & 0.1 & 0.4 & 0.1\end{bmatrix}\cdot\begin{bmatrix}1 & 0 \\0 & 1 \\1 & 1 \\0 & 2\end{bmatrix}
$$

 Step-by-step calculation

 First row

$$
z_1=0.7[1,0]+0.1[0,1]+0.1[1,1]+0.1[0,2]=[0.8,0.3]
$$

 Second row

$$
z_2=0.2[1,0]+0.5[0,1]+0.2[1,1]+0.1[0,2]=[0.4,0.9]

$$

 Third row

$$
z_3=0.25[1,0]+0.25[0,1]+0.25[1,1]+0.25[0,2]=[0.5,1.0]
$$

 Fourth row

$$
z_4=0.4[1,0]+0.1[0,1]+0.4[1,1]+0.1[0,2]=[0.8,0.6]
$$

 The final output matrix:

$$
Z =\begin{bmatrix}0.8 & 0.3 \\0.4 & 0.9 \\0.5 & 1.0 \\0.8 & 0.6\end{bmatrix}\quad \in \mathbb{R}^{4 \times 2}
$$

The meaning of this matrix is: the output vector of each token = its attention weight to all other tokens × the Value vector of each token, and then weighted and summed. In other words, the ith row of the final output matrix (i.e., the output vector of the ith token) represents the contextual representation of the token after combining the value information of all the other tokens that the token has "paid attention" to.

### Multi head attention

Single-head attention can only learn associations from one "projection space", while multi-head attention enhances the model's representational capability by computing attention in different subspaces in parallel through multiple linear transformations (i.e., multiple "heads").

![image.png](Transformer%20-%20English/image%2010.png)

$$
head_i = Attention(Q_i, K_i, V_i) = softmax\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i, \quad i \in [1, H]
$$

 Where:

$i$: the ith head index, there are $H$ heads in total.

$Q_i$: the Query matrix of the ith head, shape $[n * d_k]$, $W_i^Q * Q = Q_i$

$K_i$: Key matrix of the i-th head, shape $[n * d_k]$, $W_i^K * K = K_i$

$V_i$: Value matrix of the ith header, shape $[n * d_k]$, $W_i^V * V = V_i$

$d_k$: Key/Query vector dimension $d_k = d_{model} / H$

> How to calculate $d_k$? Suppose your input Q has dimension $d_{model} = 518$ and number of headers is $h = 4$, then $d_k = d_{model} / h = 128$
> 

$$
MultiHeadAttention(Q, K, V) = Concat(head_0, head_1, \ldots, head_H)W^O
$$

Where $W^O = [d_{model} * d_{model}]$

This formula is what stitches together the results of all the $head$ to get the overall output of multi-head attention.

![image.png](Transformer%20-%20English/image%2011.png)

**Model relationships in parallel in different subspaces (cut matrices)**

 Divide the total dimension

$$
d_{model} 
$$

 Divided into h heads, each head learns its own set of

$$
W^Q,W^K,W^V
$$

Equivalent to computing attention in different representation subspaces; this captures different types of dependencies such as short-range/long-range, syntactic/semantic, etc. at the same time (e.g., head 1 captures neighboring word relations, head 2 captures subject-predicate-object relations, head 3 captures syntactic structures, etc.), and then collapses the outputs of each head to form a richer representation. This is the core motivation of the original paper and textbook on multiple heads.

> In a word, it summarizes that the attention mechanism of multiple heads is to capture the relationship between words and improve the model's ability to understand the context.
> 

### Step 4: Normalization and residual network

 Now we explain this part, the figure Add represents residual network, norm represents normalization.

![image.png](Transformer%20-%20English/image%2012.png)

> What is a deep network? Why do we need deep networks? Deep Network Problem.
> 

 Deep Neural Network is a neural network consisting of multiple layers of nonlinear transformations stacked on top of each other. In modern large models, the number of network layers often reaches tens or even hundreds of layers. Theoretically, the deeper the network is, the richer the information it can learn: the bottom layer focuses on capturing simple patterns (e.g., edges, colors), the middle layer gradually combines these patterns to form more complex features (e.g., textures, local structures), and the top layer is further abstracted into semantic-level information (e.g., object categories), just like human beings' understanding of the world, which is a gradual transition from perception to concept.

 However, deep networks are not simply "stacked deeper and deeper" to improve performance indefinitely. As the number of layers increases, we encounter two major problems: Gradient Exploding/Vanishing and degradation, the former refers to when the gradient is too large or too small during backpropagation, resulting in unstable training or even failure to converge; these problems can be mitigated by Normalization or suitable activation functions. These problems can usually be mitigated by Normalization or a suitable activation function. The real problem is degradation: as the layers get deeper, the nonlinear activation function introduces irreversible losses in the information propagation process, resulting in a deeper network with more loss of information and degradation of performance.

> Degradation solution: residual network
> 

![image.png](Transformer%20-%20English/image%2013.png)

 The residual network layer is:

$$
y = f(x) + x
$$

 Where: x is the input (original information), f(x): information learned by this layer, y: original information + learned information.

 Regardless of the internal variation of f(x), the input raw information x can be passed directly to the next layer. If f(x) in this layer doesn't learn any useful features, at least x will be preserved and not degrade the performance of the network. This is the core idea of Residual Networks: to let the model only learn to improve upon the inputs, rather than replace them entirely.

> Highly recommended to learn: forward and backpropagation, ResNet papers, what is the role of residual network in forward and backpropagation and what is the activation function.
> 

> What is Normalization?
> 

**Normalization** is the numerical transformation of data or intermediate features so that the data falls within a more "normal" range or distribution. For example, normalizing the data range between 0 and 1.

 In today's large language model training, the models are usually tens or more than 100 layers, which makes it extremely easy for the numerical distributions of the input and intermediate features to be very different or very small, which leads to unstable training.

> Large values: gradient explosion. Small values: the gradient disappears.
> 

**The role of Normalization**:

1. **Stabilize training**: control the range of values to avoid gradient explosion or disappearance.
2. **Accelerate convergence**: reduce the internal covariate offset, so that the training faster convergence.
3. **Improve generalization**: reduces overfitting, allowing the model to perform better on new data.
4. **Support for deeper networks**: helps residual structures maintain efficient gradient flow in deeper models.

> In a word: Normalization is to make network training more stable, faster and more accurate by unifying the feature distribution.
> 

 We understand that Normalization is to normalize the values, i.e. to control the values within a certain range. So the question is, what kind of values do we normalize? In the original Transformer, the authors used Layer normalization, which normalizes all the feature dimensions of a single sample.

 For example, we have an input token with 512 dimensions.

$$
x=[x_1,x_2,x_3...x_{512}]
$$

 Layer Normalization normalizes this token alone. Here is the LN formula and the general steps.

 First calculate the mean of the 512 dimensional vector:

$$
\mu = \frac{1}{512} \sum_{i=1}^{512} x_i
$$

 Calculate the variance:

$$
\sigma^2 = \frac{1}{512} \sum_{i=1}^{512} (x_i - \mu)^2
$$

 Normalize each element:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}   \text{where i = 1,2...512}
$$

 Finally do the linear transformation (learnable scaling and translation)

$$
y_i = \gamma_i \cdot \hat{x}_i + \beta_i\quad \text{where } \gamma, \beta \in \mathbb{R}^{512}
$$

 The above Layer normalization step, put in Transformer its expression is:

$$
\text{Output} = \mathrm{LayerNorm}\!\big(x + \mathrm{Sublayer}(x)\big)
$$

 Where sublayer is f(x).

> In the original Transformer article, the author used post-norm, actually we use pre-norm more now.
> 

### Step 5: FFN - Feed forward network

 Up to now, most of the formulas we have come across are linear. People always want to find an intuitive and linear expression, but in the real world, many phenomena are essentially non-linear, such as language. In the training process of large models, in order to allow the model to acquire more knowledge and have a richer expressive ability, we usually try to introduce non-linear formulas into the model.

> An FFN (feed-forward network) is a position-wise, two-layer, fully connected neural network used in transformer architectures to perform nonlinear transformations and refine features for each input vector. If you are not familiar with FFN concepts, it is recommended that you study them separately — and don’t forget to learn about **R**NN (recurrent neural network) concepts as well.
> 

 Next, let's see how to use FFN in Transformer.

$$
\text{FFN}(x) = W_2 \, \sigma \!\left( W_1 x + b_1 \right) + b_2
$$

 Step1.

$$
W_1x+b_1
$$

 Suppose you have a vector

$$
x∈R^{d_{model}},d_{model}=4
$$

 This vector currently contains only basic semantic information, but the dimension is too small for the model to capture complex and rich patterns from it.

 In order to make the model more expressive, we need a larger space to temporarily hold and process the information. So we extend the dimensionality of the vector from $d_{model}$ to $d_{ff} = 4 * d_{model} = 16$ (usually a factor of 4)

 This is done by introducing a weight matrix.

$$
W_1∈R^{16×4},b_1∈R^{16}
$$

 by a single linear transformation:

$$
h_1=W_1x+b_1,h_1∈R^{16}
$$

 In this way, the vector, which originally had only 4 numbers, is mapped into a 16-dimensional vector. The expanded vector can carry more combinations of features and potential patterns in higher dimensional space.

 Step 2.

 Then we apply a nonlinear activation function (e.g. ReLU or GELU) to it:

$$
h_2=σ(h_1)
$$

> An activation function is a nonlinear function acting on the linearly transformed output of a neural network, which both breaks the limitation that a multi-layer linear superposition is still equivalent to a single layer, and introduces the nonlinear ability to allow the model to sift through and transform features in higher dimensional spaces, so that it can learn and express more complex and abstract patterns. Common activation function formulas are sigmoid, ReLU, GELU, etc., which I will explain separately later.
> 

 This step gives the model a stronger nonlinear representation. Finally, to get back to the original dimensions, we use another compression matrix:

$$
W_2∈R^{4×16},b_2∈R^4
$$

 to get the output:

$$
y=W_2h_2+b_2,y∈R^4
$$

 In this way, a 4-dimensional vector is expanded to 16 dimensions (more capacity) in an intermediate process, and then compressed back to 4 dimensions after a nonlinear transformation, and the final representation y is much richer and more abstract than the original input x.

> Why should FFNs be upscaled and then downscaled?
> 

 With the above steps in the FFN layer, let's now discuss the core role of FFN:

- **Enhanced nonlinear representation**
    
     The attention layer itself is linear (dot product + Softmax), the nonlinear part of FFN enables the model to learn complex patterns and feature combinations.
    
- **Extension and Compression**
    - Scaling to higher dimensions ( $d_{ff}$ ) gives the model "more room" to mine and combine features.
    - Compressing back to the original dimension ensures that the output and input have the same shape, making it easier to stack layers.
- **Position-wise Feature Extraction**
    
     After the attention layer has integrated the contextual information, the FFN acts as an "independent processing plant" for each token, helping it to further refine and abstract the information. Note that this does not mean that we can't process tokens in parallel using FFNs.
    
- **Increasing Model Depth**
    
     Self-attention is responsible for "information interaction", while FFN is responsible for "local processing", the two are alternately stacked to give the Transformer enough computational depth and expressive power.
    

 I would like to add that the two weight matrices $W_1$ and $W_2$ are learnable parameters. They are backpropagated to compute the gradient during training and are updated by an optimizer (e.g., Adam, here's the terrier who is Adam), which gradually learns how to transform the input vectors into more useful representations.

> What is backpropagation? Simply put, it means that **after the neural network gets the prediction error, it sends the error back one layer at a time, figures out how to adjust each parameter, and then updates the parameters to make the next prediction more accurate. Here we need to use the derivation. If you forget how to derive, you can review it.**
> 

### Step6

 After the FFN layer is finished, it is the normal residual network and normalization, the role of this layer is the same as before. Again, post norm is used here:

$$
z=LayerNorm(x+FFN(x))
$$

### Encoder Final output

 Up to now, we have only completed one Encoder Block, in fact, Encoder is stacked by multiple Encoder Blocks, for example, the output of the first Block will be used as the input of the second Block, and they will be stacked one on top of the other until we get the final output. The final Encoder output is a matrix of the same length as the input sequence, where each row corresponds to the contextual representation of a token. This representation not only preserves the original semantic information, but also fuses the global context through a self-attention mechanism, thus becoming a richer semantic feature. In the original Transformer architecture, part of the Decoder's input (Key and Value in the cross-attention) is derived from the Encoder's final output.

## Decoder

 As we can see from the Transformer framework, the overall structure of Decoder is basically similar to that of Encoder, which also consists of multiple stacked sub-layers, each of which contains modules such as Residual Connection, Layer Normalization, and Feed-Forward Network. Each layer contains modules such as Residual Connection, Layer Normalization, and Feed-Forward Network. These are the same as Encoder and will not be repeated here.

The differences are mainly in the following aspects:

1. **Masked Multi-Head Self-Attention**
    - The Decoder first passes through a Multi-Head Self-Attention layer with a **Mask mechanism** at the input.
    - The role of the mask is to ensure that each position can only see the current and previous words but not the future words in the prediction, thus ensuring the correctness of the auto-regressive generation (ARG).
2. **Encoder-Decoder Attention (Cross Attention Layer)**
    - Unlike Encoder, Decoder introduces an additional layer of multi-attention after the self-attention.
    - This layer uses the output of the Encoder as Key and Value, and the Decoder's own representation as Query, so that the Decoder can effectively "query" the information of the source sequence and realize conditional modeling of the input sequence.
3. **Output Layer (Prediction Layer)**
    - After stacking several layers, the Decoder will eventually output a series of hidden representations.
    - These representations are fed into a linear layer (Projection Layer), which then undergoes Softmax to obtain a probability distribution for each token in the word list, which is used to generate the next word.

 Before we get into the components of a Decoder, I'd like to introduce two more concepts: Teacher Forcing and autoregressive structures.

**What is Teacher Forcing?**

 Teacher Forcing is a strategy for training a model that guides and accelerates model training. When training a decoder, the input is not the token predicted by the model in the previous step, but the real target token in the previous step, which is called ground truth.

**What is autoregressive model?**

 This pattern is the opposite of teacher forcing, when the model predicts the next token, the output of each step can only rely on the output of the previous step, and can only predict the "future" based on the "past".

### Step 1: Input

 Input of Decoder is different from Encoder, which is the input of Decoder:

$$
<BOS> I \ love \ cream \ crackers 
$$

 BOS stands for beginning of sequence. This form of input is actually what is often called **shifted right**: shifting the target sequence one place to the right.

 Ideally, we would like to get predictions for all positions in the sequence in a single forward propagation to fully utilize GPU parallel computing power. However, if the Decoder is generated in a strictly autoregressive step-by-step fashion, then it has to be decoded word by word during training, which is extremely inefficient.

 To improve training efficiency, we want to leverage parallelism so that the model can predict all tokens in a sequence within a single training step. achieve this, researchers introduced **Teacher Forcing**, a technique that allows the decoder to take the entire target sequence as input during training, enabling parallel decoding of all outputs. enabling parallel decoding of all outputs at once.

 More specifically, Teacher Forcing means that at each step, instead of feeding the decoder with its own previous prediction, we feed it with the ground truth token(true value) from the training data. This mechanism ensures that the Transformer can produce predictions for all positions in parallel during training, without relying on sequential decoding. This mechanism ensures that the Transformer can produce predictions for all positions in parallel during training, without relying on sequential decoding.

 In practice, the decoder input is not just a single label at each step but a concatenated shifted sequence, as illustrated in the figure.

![image.png](Transformer%20-%20English/image%2014.png)

![image.png](Transformer%20-%20English/image%2015.png)

**Masked Self-Attention**

 In the Transformer, we can see two different masking mechanisms. One masking mechanism is when preparing the input and one masking mechanism is in masked self attention. These two masking mechanisms are called Padding masking and sequence masking.

- Padding masking is very common in natural language processing models. Since the lengths of different sentences are usually different, in order to batch input and keep the tensor shape consistent, we need to patch the shorter sentences to a uniform length. This is done by adding a special symbol `<PAD>` as a placeholder after the sentence. For example, if we want the input matrix to be of uniform length 4, then the sentence "I love you" would be processed as "I love you `<PAD>",` while the sentence "I love cream crackers" would occupy exactly 4 positions and would not need to be filled. Note that `<PAD>` has no real semantics, it is just a placeholder. If it is not handled when computing the attention distribution (e.g., Transformer's self-attention), the model may incorrectly assign attention to `<PAD>`, which may interfere with the normal prediction. For this reason, the Padding Mask mechanism is introduced. When calculating the attention score, the model will replace the values corresponding to the `<PAD>` positions with a very small value (e.g. -1e9), so that after softmax, the weight of these positions is almost zero, which is equivalent to being ignored, thus ensuring that the model only pays attention to real and valid tokens.
    
    For example, we have input “I love cream crackers” , length of this sentence is 4, and sequence length is 6, so we need to mask to more rows.
    

![image.png](Transformer%20-%20English/image%2016.png)

- Sequence Masking: Sequence Masking is also a common masking mechanism in Transformer. Different from Padding Mask, Sequence Mask is mainly used to ensure that the decoder can only utilize the generated words during the training and inference process, and cannot "peek" at the future information. Because in the autoregressive generation model, each step of the prediction must rely on the previous token, not the subsequent token, for example, suppose the target sequence is "I love you". In predicting the first word, the model can only see `<BOS>;` in predicting the second word, the model can only utilize `<BOS> I`; and in predicting the third word, the model can only utilize `<BOS> I love`. without Sequence Mask, the model can compute the self-attention while see "you", which would break the autoregressive generation setting. To solve this problem, we use an upper triangular mask matrix (causal mask) to mask out the attention scores at future moments, which is usually done by setting the value of these positions to a very small negative value (e.g. -1e9). In this way, the Sequence Mask ensures the autoregressive nature of the Transformer decoder, preventing information leakage while keeping training and inference consistent.

![image.png](Transformer%20-%20English/image%2017.png)

> We have explained two different forms of masking above, why does the decoder need sequence masking? This is because the Transformer adopts Autoregressive mode (generating sequences token by token, and predicting the next token based on the previous token at each step). This mode guarantees three benefits: 1: to ensure that the generation logic is correct, 2: to maintain the consistency of the training and inference phases, 3: to prevent information leakage. note that sequence masking is only for the autoregressive mode of the training and inference of the pre-fill phase, inference of the decoder does not need masking mechanism.
> 

 Let's take a look at its computational steps:

1. **Input Embedding**
    
     The input sequence is embedded and positionally encoded, resulting in a vector representation $X$. The input sequence is embedded into a vector representation $X$.
    
2. **Linear transformation**
    
    $X$ is multiplied by three weight matrices $W^Q,W^K,W^V$ to get $Q,K,V$ matrices.
    
3. **Attention Score Calculation**
    
    $Q$ is multiplied by $K^T$ to obtain the attention scoring matrix $QK^T$.
    
4. **Apply Mask**
    
     Add $QK^T$ to a Mask matrix element by element.
    
    - The lower triangular portion of the Mask matrix is left unchanged, indicating that the current position and previous tokens can be seen.
    - The upper triangle is set to -∞, i.e. "masking future information".
    - After softmax, the weight of the masked position becomes 0.
5. **Weighted Summation**
    
     Multiply the Masked attention weight matrix with $V$ to get the new context representation matrix $Z$. In other words, each row of Z has a weight of 0, and each row of Z has a weight of 0, and each row has a weight of 0.
    
     In other words, each row of $Z$ is a weighted average of all word vectors before the current position.
    

### Step 2: Add & Norm

 This step is the same as Encoder. We won't elaborate too much.

### Step3: **Encoder-Decoder Attention**

![image.png](Transformer%20-%20English/image%2018.png)

 Earlier we talked about Self-Attention, which is essentially the exchange of information within the sequence "between itself and itself". The purpose of using the model is prediction, which must be combined with historical information. In the Encoder-Decoder framework, the history information is stored in the output of Encoder, so how to ensure that the generated information can be connected with the history information? This is where Cross-Attention is introduced.

 The core idea of Cross-Attention is to build a bridge between Encoder and Decoder, so that the source sequence can interact with the target sequence. It calculates the correlation or similarity between each position in the target sequence and all the positions in the source sequence to decide which parts of the source sentence should be given more attention when generating a target token. In other words, the source sequence provides the contextual semantics, and the target sequence is responsible for generating predictions, and the two are tightly coupled through cross-attention.

**Source of Q / K / V**

 In Cross Attention:

- **Q (Query)** comes from the Decoder's output (Masked Self-Attention, already contains historical translation information).
- **K (Key)** and **V (Value)** come from the output of the Encoder (which contains the contextual representation of the source text).

 In this way, cross-attention is like an information retrieval process:

- The output of the Encoder can be seen as a "database" (V, containing all the information of the source sequence).
- Each position of the Decoder issues a "query" (Q) to find the most relevant part of the database (calculated by the similarity with K).
- The resulting weighted result tells the Decoder which words from the source sentence to focus on when generating the current token.

**Rough steps:**

- **Source sequence (Encoder input)**: `I love cream crackers`
- **Target sequence (Decoder output)**: `I love cream crackers`

 When the Decoder wants to predict `"crackers"`:

1.  Its **Query** comes from the hidden representation of the previous `"I love cream"`.
2.  It goes to the Encoder's output (the representation of `I love cream crackers` ) and looks for the most relevant position:
    - It finds that the Key of `"crackers"` is the most relevant.
3.  So it takes the Value of `"crackers"` and fuses it into the current hidden state of the Decoder.
4.  Finally, it helps Decoder to generate the target word `"crackers".`

**Calculation process:**

 Same as encoder

### Step 4 FFN + Add & Norm

 Same as encoder

### Step 5: Linear layer and softmax = generator

 By the time we get to this layer, we've figured out what the next token will be, but at this point the token is still represented as a number, i.e. a vector. We have to figure out a way to convert this vector into text that people can read.

![image.png](Transformer%20-%20English/image%2019.png)

**Linear**

 In the linear layer (often called Language Model Head or LM Head), the main goal is to map the hidden vectors generated by the decoder to the dimensions of the vocabulary so that they are ready for the prediction of the next word. In other words, the essential function of the Linear Layer is Dimension Transformation: transforming the vectors output by the decoder into vectors (logits) of the same size as the word list.

 You can also understand the Linear layer as taking the predicted vectors and making INNER products with each word in the vocabulary, waiting until the similarity with each word, i.e., logits. logits is a vector of scores for candidate words, indicating the likelihood that each word will be used as the next token in its current position. Each dimension of the vector corresponds to a word in the word list, and the larger the value, the higher the probability that the word is the correct output.

> Once the logits are obtained, the model does not simply always choose the word with the highest probability, otherwise the generated sentences would be very mechanical and lack variety (e.g. "I love you" → always answer "I love you too"). To solve this problem, a sampling strategy such as Top-k or Top-p (Nucleus Sampling) is usually introduced to sample from a fraction of words with higher scores. This ensures reasonableness and increases the diversity of the generation.
> 

**Softmax.**

 The outputs of the linear layer are logits, which by themselves are not directly interpretable as probabilities. To solve this problem, we apply a softmax function to the logits, scaling the last dimension of the vector to the interval [0,1] and guaranteeing that the sum of all elements is 1. After this process, the output of the model can be interpreted as a probability distribution representing the likelihood of each word being the next token at the current position. This step is especially crucial in multicategorization tasks, as it gives probabilistic meaning to the predictions. The model then samples according to this probability distribution to select the final generated word.

**Sampling**

 In natural language generation tasks, **sampling** is the process of randomly selecting the next word based on the probability distribution output by the model, rather than just picking the word with the highest probability. The core advantage of sampling over greedy search is that it introduces a certain amount of randomness, thus avoiding the model from falling into mechanical and repetitive patterns in its generation.

 Sampling is necessary because if the model always picks the word with the highest probability, the generated sentences tend to be very rigid and lack variation. For example, when you type "I love you", the model may only ever output "I love you too". By sampling, the model can randomly choose between multiple high-probability candidates to generate richer expressions, such as "I love you too" or "I feel the same way about you".

 Therefore, sampling not only improves sentence diversity, but also makes the generated results more natural and closer to human language habits. Different sampling strategies (e.g., Top-k or Top-p) can also strike a flexible balance between "reasonableness" and "variety", ensuring that the output is not too outrageous, but also avoids being static.

> There are many different sampling algorithms that can be used in this step, so I won't introduce them, but you can learn them yourself if you are interested.
> 

### Final output

 After sampling, the resulting token is added to the generation sequence and used as input for the next step, and the process continues until the sentence is complete.