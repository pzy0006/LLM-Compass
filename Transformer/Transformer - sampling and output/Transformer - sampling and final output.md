# Transformer - sampling and final output

![82982a36-e63d-4d7b-9a28-9ca550a465cc.png](Transformer%20-%20sampling%20and%20final%20output/82982a36-e63d-4d7b-9a28-9ca550a465cc.png)

The last two layers can be called Generator:

## 1. The Linear Layer

after the decoder processes the input, it produces a vector of continuous numbers(the hidden state). While this vector contains the “meaning” of the next word, it isn’t a word yet.

- Role: it maps the hidden state to much larger space: the Vocabulary size, like from 512(dimension) to 50,000(Vocabulary size)
- The math: it multiplies the hidden state by a massive weight matrix.
    - if your vocabulary has 50,000 words, the Linear layer outputs 50,000 raw numbers.
- Output(logits): These raw numbers are called Logits. They are unnormalized scores - higher values indicate a higher likelihood that a specific word is the correct next token.

$$
Z = \text{Decoer's Output} \times W 
$$

> The final vector produced by the decoder is multiplied by this Token Embedding matrix—essentially calculating the **inner product** (dot product) between the decoder's output and each token's embedding vector. This process yields **similarity scores (logits)** for every token in the vocabulary, generating a list of numerical values that represent the likelihood of each word appearing as the next token.
> 

## 2. The Softmax Layer(The probability)

Logits are hard to work with because they can be any real number (e.g., -5.2, 10.1, 0.03). To make a decision, we need a clean probability distribution.

- Role: it squashes the logits into a range between 0 and 1, ensuring that sum of all values equals 1.0 (100%)
- The math: it uses the exponent function ( $e^x$) to amplify the differences between scores. Even a small lead in a logit score can become a massive lead in probability after SoftMax.
- Output: A probability distribution where each index corresponds to a words in your dictionary.

## Linear layer in Training vs Inferencing

### Inferencing

During inference, we only take the tensor corresponding to the last token of the Decoder's output and pass it to the Generator to obtain the probability distribution for the next single word.

### Training

During training, however, the entire sequence of hidden states from the Decoder is passed to the Generator. This generates a probability distribution for every token position in the sequence simultaneously. At each position, we identify the index of the word with the highest probability (via greedy search) and map that index to the corresponding word in the vocabulary. These predicted words collectively form the output sequence of the Transformer.

## Sampling Strategy

Once we have all next words probability, how can we pick? the one has the highest probability? Or randomly pick one? 

### 1. Greedy Search

Greedy search is the simplest decoding strategy. It follows the "highest probability at every step" rule.

- Algorithm:
    - Obtain the probability distribution P for the entire vocabulary.
    - Select the token W with the maximum probability $w = argmax(P)$
- Pros: Very fast and computationally cheap.
- Cons: Often leads to repetitive loops like “I am, I am….” and lacks creativity

### 2. Top-K Sampling

This method limits the selection pool to a fixed number of candidates, filtering out the "long tail" of low-probability words.

- **Algorithm:**
    - Sort the vocabulary tokens by their probabilities in descending order.
    - Keep only the top K tokens.
    - Set the probability of all other tokens (K+1 to V) to zero.
    - Renormalize the probabilities of the top K tokens so they sum to 1.
    - Randomly sample from this new distribution.

### 3. Top-P Sampling (Nucleus Sampling)

Unlike Top-K, which uses a fixed number of tokens, Top-P uses a dynamic number based on the **cumulative probability**.

- **Algorithm:**
    - Sort tokens by probability in descending order.
    - Calculate the cumulative sum of probabilities.
    - Find the smallest set of tokens $\{w_1, w_2, ... w_n\}$ such that their combined probability exceeds P (e.g., P = 0.90).
    - Remove all tokens outside this set.
    - Renormalize and sample from the "nucleus" of tokens.

### **4. Beam Search**

Commonly used in translation and summarization, Beam Search looks multiple steps ahead rather than just at the next word. To find the most likely *sentence* overall, even if the first word wasn't the absolute highest probability.

- **Algorithm:**
    - Keep the top N (Beam Width) most likely sequences at each step.
    - For each of those N sequences, predict the next word candidates.
    - Calculate the **total log-probability** for all new combinations.
    - Select the top N highest-scoring total sequences to move to the next step.
    - Repeat until an `<EOS>` (End of Sentence) token is reached.

### 5. Tempeture Sampling

In Large Language Models (LLMs), there is a hyperparameter used to control the 'randomness' or 'creativity' of the generated text, known as **Temperature**. The temperature hyperparameter $T$ is typically applied within the Softmax function. Assuming the model generates a raw score (logit) $z_i$ for the $i$th word in the vocabulary $V$, the formula for the probability $P_i$ adjusted by $T$ is:

$$
P_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}
$$

**Parameter Descriptions:**
• $P_i$: The adjusted probability of selecting the $i$-th word.
• $z_i$: The raw Logit predicted by the model (the unnormalized score).
• $V$: The size of the vocabulary.
• $T$: The Temperature parameter (typically ranging between 0 and 2).

### Calculation Process

Assume the vocabulary is very small. When the input is "I love", the model's goal is to predict the next word.

### Step 1: Calculate the Exponential Scores $\exp(z_i / T)$

| **Candidate** | **Raw Logit (zi)** | **T=0.7 (Conservative)** | **T=1.0 (Standard)** | **T=1.5 (Random)** |
| --- | --- | --- | --- | --- |
| **cats** | 10 | $e^{14.28} \approx 1,591,200$ | $e^{10} \approx 22,026$ | $e^{6.66} \approx 780.5$ |
| **dogs** | 8 | $e^{11.42} \approx 91,125$ | $e^8 \approx 2,981$ | $e^{5.33} \approx 206.4$ |
| **birds** | 5 | $e^{7.14} \approx 1,261$ | $e^5 \approx 148.4$ | $e^{3.33} \approx 27.9$ |
| **Sum ($\sum$)** | - | **1,683,586** | **25,155.4** | **1,014.8** |

Step 2: Calculate the Final Probabilities $P_i$

| **Predicted Word** | **T=0.7 (Low Temp)** | **T=1.0 (Original)** | **T=1.5 (High Temp)** |
| --- | --- | --- | --- |
| **cats** | **94.5%** | **87.5%** | **76.9%** |
| **dogs** | 5.4% | 11.9% | 20.3% |
| **birds** | 0.1% | 0.6% | 2.8% |

**1. $T < 1$: Distribution Convergence and Hardening**
When the temperature parameter is less than 1, the term $z_i / T$ multiplies the numerical differences between the original logits. After processing through the exponential function $\exp$, these differences evolve into massive multiplicative gaps. From a mathematical perspective, the probability distribution converges toward the argmax function (the maximum value function). In this state, the system exists in a low-entropy state, and the model exhibits extreme determinism. The status of high-probability candidates becomes absolute, while low-probability tokens are effectively stripped of any chance of being selected. This setting is ideal for tasks requiring high logical rigor but can easily lead the model into repetitive loops during text generation because its word-selection path becomes too narrow.

**2. $T = 1$: Natural Statistical State**
When the temperature equals 1, the formula reduces to the standard Softmax operation. At this point, the model strictly follows the statistical patterns learned from large-scale corpora during the pre-training phase, without any manual intervention in its prediction confidence. The output reflects the model’s most authentic semantic association strengths, reaching a natural equilibrium—determined by the training data—between text coherence, logic, and lexical richness.

**3. $T > 1$: Distribution Flattening and Chaos**
When the temperature parameter is greater than 1, the increase in the denominator $T$ reduces the relative gap between logits. This means the probability of the originally dominant candidate (Top-1) is forcibly lowered, while the probability of "long-tail" tokens (low-probability words) in the vocabulary is significantly raised. As $T$ increases further, the probability distribution gradually converges toward a Uniform Distribution. In this state, the model demonstrates intense randomness and divergent thinking, potentially leading to "hallucinations" or a collapse in grammatical logic as it begins to perform unordered sampling across a vast array of semantically irrelevant words.

### **Other Properties of Temperature**

**1. $T = 0$: Deterministic Mode and Understanding Tests**
When $T = 0$, the model's output enters Deterministic Mode. In this state, the Softmax probability distribution collapses into a single "spike" (a delta distribution). The model loses its ability for "random selection," and for any given prompt, it will return identical results every time. This mode is particularly useful for testing whether the model has truly "understood" the patterns or logic provided in your few-shot samples.

**2. $T \gg 1$: Self-Consistency and Ensemble Logic**
When $T$ is significantly greater than 1, the strategy aligns with concepts in machine learning known as Ensemble Learning or Multi-path Reasoning Sampling. In academia, the most famous application of this is called Self-Consistency.

- Although setting $T$ excessively high (e.g., above 2.0) is generally discouraged, generating multiple answers within the $T = 0.7 \sim 1.0$ range and performing a "majority vote" is an excellent way to improve accuracy.
- While a single attempt might fail, if the model possesses a foundational understanding of a topic, the correct logical path will typically appear more frequently across multiple samples than any single specific error path.

**3. Temperature and Generation Length (The EOS Factor)**
Temperature also significantly influences the length of the generated text by affecting the `<EOS>` (End of Sentence) token probability:

- **Low Temperature ($T < 1$):** If the model is confident the sentence is complete, the logit score for `<EOS>` will be high. A low temperature amplifies this advantage, allowing the model to "wrap up" decisively. The resulting output is usually concise and punchy.
- **High Temperature ($T > 1$):** High temperature smoothes the distribution, lowering the relative probability of selecting `<EOS>`. The model may miss the optimal moment to stop and instead select a lower-probability but viable transition word (such as "moreover" or "additionally"), leading to a more "talkative" or wordy response.