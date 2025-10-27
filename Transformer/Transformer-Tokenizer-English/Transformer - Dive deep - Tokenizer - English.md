# Transformer - Dive deep - Tokenizer - English

![image.png](Transformer%20-%20Dive%20deep%20-%20Tokenizer%20-%20English/image.png)

We already know that the variable accepted by Transformer can only be high dimensional vectors, and there is a very important stage to go through before a paragraph/text can be transformed into an embedding matrix: Tokenization . A good tokenizer can determine the accuracy and performance of your model.

### The main steps of the lexicon:

1. **Vocabulary Building**
    - Based on the corpus, all text units (such as characters, words, or subwords) that have appeared are collected and deduplicated to form the vocabulary.
    - Different tokenization algorithms — such as WordPiece, BPE (Byte Pair Encoding), and Unigram — dynamically construct their vocabularies according to specific criteria such as token frequency, merge rules, or probabilistic modeling.
    - Each token in the vocabulary is assigned a unique integer ID, which serves as its index during model training and inference.
2. **Pre-tokenization**
    - Includes normalization (case folding, punctuation standardization) and removal or replacement of special symbols.
    - Long texts are often split into sentences or fixed-length segments to fit the model’s input constraints.
3. **Tokenization**
    - Text is segmented into smaller units (tokens) according to the vocabulary and algorithm rules.
    - The granularity of tokens depends on the algorithm type:
        - **Word-level**: split by words (may cause OOV issues).
        - **Character-level**: split by characters (fine-grained but inefficient).
        - **Subword-level**: balance between words and characters; effectively handles new words.
4. **Indexing**
    - Each token is mapped to its integer ID in the vocabulary.
    - The resulting ID sequence serves as model input.
    - Special tokens such as `[CLS]`, `[SEP]`, and `[PAD]` are often added for specific purposes.
5. **Embedding**
    - The model retrieves the vector representation of each token from the embedding matrix.
    - These vectors encode semantic and contextual information and serve as the foundation for downstream language understanding and generation.

# Tokenizer Algorithm

There are many Tokenizer algorithms. There are three kinds of algorithms according to the granularity of words: word base, char base and subword base.

- Word base
    - Word base. For example, I love cream crackers is categorized as ["I", "love", "cream", " crackers"]
    - Pros: maintains semantic integrity well, simple algorithm implementation
    - Disadvantages: need to list all word combinations, which seriously affects computational efficiency and consumes memory. Can lead to OOV (out of vocabulary) problem: words that are not in the model word list.
- char base
    - Character-level categorization. For example [I, l, o, v, e, c, r....]
    - Advantages: fewer characters, no OOV problem
    - Disadvantages: no access to rich information such as semantics. It is difficult for the model to learn word-to-word and sentence-to-sentence relationships.
- subword base
    - Between word and char, e.g. [I, lo, ve, cr, ea, m, cra, c, kers].
    - Advantage: effectively alleviate the OOV problem. Higher generalization ability compared to char base. Particularly effective for morphologically rich languages (e.g. German, Finnish, Arabic)
    - Cons: increased complexity of the generation task: for generative models (e.g., GPT), multiple tokens are needed for a single word, leading to slower and longer generation.

| **Algorithm name** | **Typical Usage Model** | **Core Idea / Features** | **Pros** | **Disadvantages** |
| --- | --- | --- | --- | --- |
| **Word-level Tokenizer** |  Word2Vec, GloVe, early RNNs, ELMo |  Split by full word. Each word is a token. |  Simple to implement, clear semantics |  Easy to generate OOV, huge word list |
| **Character-level Tokenizer** |  CharCNN, DeepMoji, some speech/spelling models |  Each character as a token |  No OOV, cross-language |  Too long sequence, weak semantics |
| **BPE (Byte Pair Encoding)** |  GPT-2, GPT-3, RoBERTa, MarianMT |  Merge most common character pairs to form subword units based on statistical frequency |  Balancing OOV with word list size |  Subwords are not always linguistically natural |
| **WordPiece** |  BERT, ALBERT, DistilBERT, ELECTRA |  Selection of subword combinations by probability maximization (similar to BPE, but more precise) |  Can handle rare words better |  Complex to build word lists, slightly slower to compute |
| **Unigram Language Model** |  T5, XLM-R, mT5, ByT5 |  Use language model to select optimal set of subwords, each token is a probability unit |  More flexible sub-word partitioning |  Complex training process |
| **SentencePiece (based on BPE or Unigram)** |  T5, mT5, XLNet, ALBERT (Google version) |  Acts directly on raw bytes, does not rely on spaces, supports multiple languages |  Cross-language generalization, stable training |  Output token is not intuitive (contains prefix symbols, e.g. "▁") |
| **Byte-Level BPE** |  GPT-2 (OpenAI original), GPT-3, GPT-4, LLaMA |  BPE encoded directly at the byte level, can handle arbitrary characters |  No need for additional character sets, avoid OOV completely |  More tokens, heavier computation |
| **SentencePiece (Character-level)** |  ByT5, Charformer |  Convert text to UTF-8 byte sequence for modeling |  Language independent |  Semantic fragmentation |
| **Tiktoken (OpenAI special edition)** |  GPT-3.5, GPT-4 series (ChatGPT) |  Improved byte-level BPE to optimize encoding efficiency |  Fast, multi-language compatible, context window saving |  Closed implementation (not open source) |
| **SPM BPE + BERT Tokenizer Hybrid** |  DeBERTa, ERNIE, NEZHA (Chinese model) |  Fusion of SentencePiece and WordPiece/BPE, suitable for Chinese multi-granularity |  Good for Chinese |  Complicated adjustment, difficult to unify word lists |

# BPE

BPE algorithm was firstly proposed by google, literature link: https: https://arxiv.org/abs/1508.07909

The BPE (Byte Pair Encoding) algorithm first splits each text word into a sequence of letters with Char granularity, and then, starting from the character level, merges the pair of neighboring tokens with the highest frequency at each step into a new token that has not appeared in the data, so as to gradually build a longer vocabulary or phrase representation, forming a kind of hierarchical subwords based on statistical frequency. statistical frequency-based hierarchical subword structure.

This process is iterated until a predefined vocabulary size or number of merge operations is reached. After training, a merge table is generated, which records the order and rules of each token merge. This table is used to reconstruct the token sequence of the original data when encoding or decoding the text later.

The core idea of BPE is a greedy algorithm: it locally selects the most frequent token pairs for merging in each iteration, but this strategy does not necessarily guarantee global optimality; at the same time, frequency is not necessarily the best merging metric (e.g., it can sometimes break semantic boundaries).

Nevertheless, BPE is still an extremely high-performance, simple and controllable tokenization algorithm.

Not only does it significantly alleviate the OOV (Out Of Vocabulary) problem, but it can also easily keep the total number of tokens within a manually set range, striking a good balance between efficiency and effectiveness.

Therefore, BPE has become one of the default word separation algorithms used in many modern Transformer models (e.g., GPT-2, RoBERTa, MarianMT, etc.).

![Screenshot 2025-10-24 at 3.57.05 PM.png](Transformer%20-%20Dive%20deep%20-%20Tokenizer%20-%20English/Screenshot_2025-10-24_at_3.57.05_PM.png)

 (The above image is from BPE literature)

## Demonstration with an example for BPE

 Let me discuss the above algorithmic process with an example:

$$
{'low': 5, 'lowest': 2, 'new: 6, 'widest': 3}
$$

Suppose we get a result after pre- tokenization of a text according to corpus. The text contains 5 low, 2 lowest, 6 new, and 3 widest.

### Preliminary segmentation: subword + terminator

At this stage, we need to split each word in the set into its smallest unit: chars, and in order for the algorithm to know the end of each word, we need to add a termination symbol after the split $</w>$ This is our Vocabulary. This is our Vocabulary Set.

$$
{’l o w </w>’:5,’l o w e s t </w>’:2,’n e w </w>’:6,’w i d e s t </w>’:3}
$$

> Why do we need a terminator? For example, let's say we have two words, string and widest, both of which contain the letter combination st, but in string, st appears at the beginning of the word, and in widest, st is at the end. They have completely different meanings in different positions. Therefore, we must add a termination symbol to allow the algorithm to distinguish between the two types of st, so that it can correctly understand the structure of the word.
> 

### Building Subword Vocabulary Set

The original subword vocabulary set should contain all the letters that appear in the text plus the final symbol:

$$
['l', 'o', 'w', '</w>', 'e', 's', 't', 'n', 'i', 'd']
$$

The current state of the vocabulary set is too compressed to represent the meanings, so we must iterate and merge the vocabulary set with the BPE algorithm until we reach the desired vocabulary size.

### Merging and Iteration

In the Byte Pair Encoding algorithm, the next step is to identify the most frequent neighboring character pairs in the Frequency Set and merge them into new symbols. This process is repeated in an iterative manner until a predefined maximum number of tokens or iterations is reached.

By continuously merging high-frequency character pairs, BPE is able to represent the corpus efficiently with a minimum number of tokens, thus realizing its core goal - data compression. In each iteration, the algorithm first counts the frequency of each character pair in the current corpus, selects the most frequent pair for merging, and adds the newly generated token to the word list. Since merging changes the composition of the corpus, the algorithm recalculates the frequency of each token after each iteration.

This merging and updating process continues until a predefined termination condition is reached. Next, we will discuss in detail the steps that occur in each iteration.

### Step 1: calculating frequencies for each character pair

In the first iteration of the algorithm, we begin by performing a fine-grained tokenization of the original word set based on the  ****vocabulary set.

This process decomposes each word into its most fundamental character units, allowing us to count the frequency of individual characters within the corpus.

Next, we compute the co-occurrence frequency of all adjacent character pairs.

These pair frequencies serve as the foundation for subsequent merge operations in the BPE algorithm.

The detailed frequency statistics are shown below. Frequency Set: 

$$
('l', 'o'): 7, ('o', 'w'): 7, ('w', '</w>'): 11, ('w', 'e'): 2, ('e', 's'): 5, ('s', 't'): 5, ('t', '</w>'): 5, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3
$$

### Step 2: find the highest frequent character pair

After completing the statistics of the frequency of character pairs, the next step is to select the character pairs with the highest frequency and merge them. In our case, the highest frequent character pair is:

$$
('w', '</w>'): 11
$$

 We need to merge these two characters and add them to the Subword Vocabulary.

$$
['l', 'o',  'w</w>', 'e', 's', 't', 'n', 'i', 'd','w'，'</w>']
$$

 Recount and update the frequency set:

$$
('l', 'o'): 7, ('o', 'w</w>'): 5, ('o', 'w'): 2, ('w', 'e'): 2, ('e', 's'): 5, ('s', 't'): 5, ('t', '</w>'): 5, ('n', 'e'): 6, ('e', 'w</w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3
$$

Finally merge the Vocabulary set: merge low</w>andnew</w>.

$$
\texttt{{'l o w</w>': 5, 'l o w e s t </w>': 2, 'n e w</w>': 6, 'w i d e s t </w>': 3}}
$$

### Step 3: second iteration

Find the highest frequency pair according to the latest frequency set:

$$
('l', 'o'): 7
$$

Merge Subword Vocabulary set:

$$
['lo',  'w</w>', 'e', 's', 't', 'n', 'i', 'd', 'w','</w>']
$$

Update the frequency set:

$$
{('lo', 'w</w>'): 5, ('lo', 'w'): 2, ('w', 'e'): 2, ('e', 's'): 5, ('s', 't'): 5, ('t', '</w>'): 5, ('n', 'e'): 6, ('e', 'w</w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3}
$$

Merge Vocabulary set: l and o

$$
\texttt{{'lo w</w>': 5, 'lo w e s t </w>': 2, 'n e w</w>': 6, 'w i d e s t </w>': 3}}
$$

The next iteration is the same: the highest frequency pair is n and e. The final Subword Vocabulary set we get is:

$$
['est</w>', 'new</w>', 'low</w>', 'wid', 'lo', 'w']

$$

Vocabulary Set.

$$
\texttt{{'low</w>': 5, 'lo w est</w>': 2, 'new</w>': 6, 'wid est</w>': 3}}
$$

We continue to repeat the above steps until we reach a predefined vocabulary size or until the iteration condition is satisfied, or until the next most frequent character pair has a frequency of 1. 

## Implementation of BPE

 See Stanford university cs 336 assignment 1

## Other Algorithms

 WordPiece, Mini BPE, BBPE, UNILM

## Papers You may like

 Byte Latent Transformer: Patches Scale Better Than Tokens

 The Future of AI: Exploring the Potential of Large Concept Models