# Transformer - FFN

![image.png](Transformer%20-%20FFN/image.png)

While Self-Attention handles the “routing” of information ( like figuring out which words should talk to which), the FFN handles the “Processing” of that information (interpreting what those relationships mean).

The authors of the Transformer recognized that attention alone might not be suffcient to fit complex data processes. To address this, they introduced the FFN to every block to significantly enhance the model’s capacity and non-linearity.

## The formula

$$
FFN(x) = max(0,xW_1+b_1)W_2 +b_2
$$

The FFN is a two-layer fully connected network applied to each position separately. It consists of two linear transformations with a non-linear activation function (RELU) in between.

where:

- $x$ is the input vector(the output from the multi-head attention layer) for a specific token position
- $W_1$ and $b_1$ are the weights and bias of the first layer (expansion)
- $W_2$ and $b_2$ are the weights and bias of the second layer (contraction)
- $max(0,...)$ represents the ReLU activation function.

### **detailed Component Breakdown**

- First Linear Layer (Expansion):
    - The input $x$ (dimension $d_{model}$) is multiplied by $W_1$ to project it into a higher-dimensional space, $d_{ff}$.
    - **Why?** In the standard Transformer, the dimension expands from **512 to 2048** (a $4\times$ expansion). This wider layer allows the model to learn more complex features and "unfold" the information compressed by the attention mechanism.
- Non-Linear Activation (ReLU):
    - Mathematically: $ReLU(z) = \max(0, z)$.
    - This step introduces non-linearity. Without this, the two linear layers would mathematically collapse into a single matrix multiplication, preventing the model from learning complex patterns.
    - *Note:* While the original paper used ReLU, modern variants (like BERT or GPT) often use **GELU** (Gaussian Error Linear Unit) for smoother gradients.
- Dropout
    - Although now always written in the simplified formula, a Dropout layer is applied to the output of the ReLU activation before it is multipliied by the second weight matrix. This prevents overfitting.
- Second linear layer
    - The vector is multiplied by $W_2$ to project it back down from $d_{ff}$ to the original dimension $d_{model}$ (e.g., $2048 \to 512$). This ensures the output can be added to the residual connection and passed to the next layer.

![image.png](Transformer%20-%20FFN/image%201.png)

The paper “Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth” shows a strong failure mode of pure stacked self-attention: without residual/skip connections and without MLP/FFN blocks, deeper self-attention layers drive the representation toward token uniformity and the network output rapidly approaches rank-1 (i.e., token representations become nearly identical), making the model effectively unusable. The same work argues that skip connections and MLP/FFN components stop/slow this degeneration, which is why Attention + Residual + FFN are structurally complementary.

## FFN extracts richer semantic information

What attention gives you: Self-attention primarily computes content-addressed mixtures of value vectors—i.e., each token becomes a weighted average of other tokens’ representations. This is great for *gathering* relevant context, but it is structurally biased toward smoothing: repeated “mixing/averaging” tends to wash out distinctions, pushing tokens toward similar representations (the “token uniformity” effect).

Why FFN helps: The FFN is a position-wise nonlinear feature transformer. After attention collects a contextual blend, the FFN:

- **Mixes channels (features) within each token**, rather than mixing tokens.
- Applies **nonlinearity** (e.g., GELU/ReLU), which enables **feature detection and feature construction**: interactions like “this token is a PERSON *and* appears in an appositive clause *and* is linked to a date mention” are not representable as repeated linear averaging.
- Often uses an **expand → nonlinearity → project** pattern (e.g., $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$), which gives the model a wider intermediate space to carve out **semantic sub-features** before compressing back.

**Deep intuition:** attention is a *router*; FFN is the *local compute*. Attention answers “*which other tokens matter for me right now?*” FFN answers “*given that contextual evidence, what semantic features should I represent?*”

Without FFNs, you mostly have progressively refined averaging. With FFNs, each layer can re-interpret the gathered context into higher-level semantic features rather than just passing along mixtures—countering the trend toward uniformity.

## FFN increases the model’s expressive power

Why attention alone is constrained: even though attention uses a softmax, the update on values is still a weighted sum—a structure that can become progressively compressive under depth, culminating in rank collapse in the “pure attention” setting analyzed by the paper.

What property gives FFN expressivity:

- Nonlinearity breaks the “everything is just a mixture” regime. It allows *conditional computation*: small contextual differences can trigger different activations and therefore different transformations.
- Width expansion (large $d_\text{ff}$) provides a big intermediate basis to represent many latent features simultaneously before projecting back, increasing the function class the network can implement.
- In the paper’s framing, FFNs (along with residual paths) are precisely the components that prevent deep stacking from degenerating into near rank-1 behavior—i.e., they keep depth useful by allowing representations to be reshaped, not merely averaged.

## FFN stores knowledge (parameterized memory)

There’s a complementary empirical view: FFNs aren’t only generic compute blocks; they often behave like **memories**.

**Why the FFN naturally behaves like memory:**

- A standard FFN is $\mathrm{FFN}(x)=W_2 \sigma(W_1 x)$. This resembles a **key–value** mechanism:
    - W1x produces activations that act like pattern matches (keys).
    - Those activations gate contributions of W2, which act like stored outputs (values).
- This is not just metaphorical: “Transformer Feed-Forward Layers Are Key-Value Memories” argues FFNs operate as key–value memories where keys correlate with textual patterns and values influence output distributions.
- Work on knowledge neurons further supports that factual knowledge can be localized to subsets of FFN neurons and can be manipulated by suppressing/activating them, consistent with “knowledge stored in FFN parameters.”

**Deep intuition:** attention helps retrieve *from the prompt* (dynamic context), while FFN helps retrieve *from the model’s parameters* (implicit stored associations).

![Screenshot 2026-01-28 at 3.52.39 PM.png](Transformer%20-%20FFN/Screenshot_2026-01-28_at_3.52.39_PM.png)

## Why the activation function in the FFN is so critical

In a Transformer block, self-attention is mostly a routing operator: it mixes information *across tokens* by taking weighted averages of value vectors. On its own, this kind of token mixing tends to be smoothing—it can blur distinctions when stacked deeply. The FFN is the main per-token compute module, and the activation function is the part that turns it from “just another linear map” into a real nonlinear program.

Here’s what the activation is really doing inside the FFN:

### 1) It creates nonlinearity, which is where “new features” come from

Without an activation, the FFN becomes linear:

$$
W2(W1x)=(W2W1)x
$$

So stacking layers would largely reduce to repeated linear transforms plus token mixing—limited in the kinds of abstractions it can form. The activation breaks this linearity, enabling the model to build **compositional features** (e.g., interactions like “A and B but not C”) that cannot be represented by repeated averaging and linear projection alone.

### 2) It implements gating / feature selection (conditional computation)

Most activations behave like **gates**: they suppress some directions and amplify others based on the input. That means the FFN can act like a set of “if-then” rules:

- if a pattern is present → activate a feature strongly
- if not → keep it small or zero
    
    This selective behavior is essential after attention has aggregated context: the model needs a mechanism to **d**ecide what to keep, what to discard, and what to sharpen into higher-level semantics.
    

### 3) It controls gradient flow and training stability in deep networks

The activation determines:

- where gradients are **zero vs non-zero** (risk of dead units vs smooth learning)
- how **smooth** the function is (sharp kinks vs continuous derivatives)
- how activations distribute (sparse vs dense), which affects optimization noise and generalization
    
    In large Transformers, these properties directly impact whether training is stable and efficient at scale.
    

**ReLU: hard gating, sparsity, and cheap compute**

$$
ReLU(x)=max(0,x)
$$

### Hard gating: a literal on/off switch

ReLU imposes a **sharp threshold at 0**. Anything below 0 is *deleted* (set to exactly zero), anything above 0 passes through unchanged. In an FFN, this acts like binary feature selection:

- The first linear layer $W_1x$ creates many candidate features (directions in feature space).
- ReLU then decides, per feature channel, whether it is active (positive) or inactive (non-positive).
- The second linear layer $W_2$ combines only the active channels into the residual stream.

Because the gate is “hard,” the FFN can implement crisp, rule-like behavior: *if this projection exceeds a threshold, enable this feature; otherwise silence it*. This makes it easy for the network to carve the input space into piecewise-linear regions, where different sets of features fire in different regions.

**Sparsity: selective representations, implicit regularization**

The hard cutoff produces **exact zeros** in the hidden activation. In practice, a substantial fraction of channels will be zero on a given token.

That sparsity matters because it changes how the FFN behaves:

- **Selective computation:** only a subset of features contributes to the output for each token, which encourages specialization (different neurons respond to different patterns).
- **Reduced interference:** zeroed channels cannot “leak” weak, noisy activations into later layers, helping preserve sharp distinctions after attention mixing.
- **Implicit regularization:** sparse activations limit the effective degrees of freedom used per example, which can improve generalization in some regimes.

In Transformer FFNs, this sparsity is also a form of conditional computation: each token “chooses” a small active subnetwork (the neurons with positive pre-activations).

### Cheap compute: simple and hardware-friendly

ReLU is computationally minimal: a compare + max. It’s:

- **Fast on modern accelerators**
- **Numerically stable**
- **Easy to fuse** with other ops (common in optimized kernels)

This is one reason ReLU historically dominated early deep learning, and it remains a strong baseline when throughput and simplicity matter.

## **GELU: smooth “probabilistic gating” and finer-grained feature selection**

ELU (Gaussian Error Linear Unit) can be viewed as scaling the input by a smooth “keep probability”:

$$
GELU(x)=xΦ(x)
$$

where $\Phi(x)$ is the standard normal CDF (implementations typically use a fast approximation).

This makes GELU feel like: **“**keep large positive evidence, softly suppress weak/negative evidence”, rather than the hard on/off behavior of ReLU.

### Probabilistic gating: a soft, input-dependent mask

In an FFN, the first linear layer $W_1x$ produces many candidate features. GELU applies a **continuous gate** to each feature:

- If a feature pre-activation xxx is strongly positive, $Φ(x)≈1$⇒ $output ≈x$(mostly preserved).
- If xxx is near 0, Φ(x) is around 0.5 ⇒ output ≈0.5 (partially preserved).
- If xxx is negative, Φ(x) shrinks toward 0 ⇒ output smoothly fades (not abruptly killed).

So the “gate” is not binary; it’s a graded retention factor that behaves like a probability weight. In practice, that means the model can express *degrees of belief* in a feature rather than making an immediate cut.

### Why this yields more nuanced feature selection

**ReLU makes a discrete decision**: either a feature participates fully or it is zeroed out. That encourages crisp partitioning but can be too coarse when evidence is ambiguous.

**GELU enables soft participation**:

- Features with weak but meaningful evidence can still contribute a little.
- Competing features can be blended smoothly rather than winner-take-all.
- Subtle context shifts can modulate feature strength continuously.

In language modeling, this is particularly useful because many linguistic cues are **graded** (syntax/semantics often provide partial signals rather than hard triggers). GELU lets the FFN encode that gradedness directly in activation magnitudes.

### Smoothness: better-behaved optimization geometry

GELU is **smooth** (no sharp kink like ReLU at zero). That matters because:

- Gradients change continuously as activations cross zero.
- Small input changes produce small output changes (locally stable behavior).
- The model is less prone to “flip” features abruptly due to noise or minor perturbations.

For large Transformers, this smoothness often translates into more stable training dynamics, especially when many layers repeatedly transform the same residual stream.

## SiLU / Swish: self-gating that balances smoothness and gradient flow

**Definition**

SiLU (a.k.a. Swish-1) is:

$$
SiLU(x)=x⋅σ(x)
$$

where $\sigma(x)=\frac{1}{1+e^{-x}}$ is sigmoid

The key idea is explicit in the formula:**the input gates itself.** The same scalar x both (a) provides the signal and (b) determines how much of that signal passes through.

### Self-gating: the feature controls its own “open rate”

In an FFN, the first linear layer W1xW_1xW1x generates candidate feature pre-activations. With SiLU, each candidate feature is transformed as:

- **Gate** = $σ(x)$ (between 0 and 1)
- **Output** = $x×Gate$

So the neuron behaves like: *“if this feature is confidently positive, pass it almost fully; if it’s uncertain or negative, suppress it smoothly.”*

Compared with GELU’s “probabilistic keep” interpretation, SiLU is a very direct engineering version of the same principle: **continuous gating**, but implemented via sigmoid.

### Smoothness: no hard threshold, no kink at zero

ReLU has a sharp decision boundary at 0. SiLU is smooth everywhere:

- No discontinuity in gradient
- No abrupt switching behavior around 0

That smooth transition is useful when many layers repeatedly refine representations: small changes in the residual stream won’t suddenly flip a feature from fully-off to fully-on, which reduces training brittleness.

### Gradient flow: fewer dead units than ReLU

A practical benefit is that SiLU **does not zero out the entire negative half-line**. ReLU sets all x<0x<0x<0 to 0, so those units have **exactly zero gradient** there—leading to “dead ReLUs.”

SiLU, by contrast, produces small negative outputs for negative inputs and keeps a nonzero (though sometimes small) gradient over a much wider region. This means:

- neurons are less likely to stop learning permanently
- optimization is often smoother in deep / large models

You can think of it as **soft suppression instead of hard deletion**, which preserves a training signal even when a feature is currently “wrong-signed.”

![image.png](Transformer%20-%20FFN/image%202.png)