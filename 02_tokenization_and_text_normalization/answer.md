# Tokenization and Text Normalization

#### What is tokenization in NLP, what are the different types of tokenizers, and why is text normalization important?

Tokenization is how we are going to split our sentences or paragraphs in smaller pieces of information which are called *tokens*. A token could be from a mark punctuation to a group of n-words.

There are 7 principal types of tokenizers that will be discussed in this document:
    
    1. Whitespace Tokenizer
    2. Word Tokenizer
    3. Subword Tokenizer (Byte-Pair Encoding - BPE)
    4. WordPiece Tokenizer
    5. SentencePiece Tokenizer
    6. Unigram Language Model Tokenizer
    7. Character Tokenizer

### 1. Whitespace Tokenizer
This is a simple but effective and commonly used tokenization. It consists of split the text by its white spaces ` ` and sometimes by line breaks `\n`.  
It could be implemented via native Python (`split()` function), regex  or using the NLTK library.  
Here's an implementation with the NLTK library:

```python
>>> from nltk.tokenize import WhitespaceTokenizer

>>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
>>> WhitespaceTokenizer().tokenize(s)
['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.',
'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.']
```

### 2. Word Tokenizer
The word tokenizer as it names says is a kind of tokenization based on words, but this tokenizer has a little but important consideration. This tokenizer (at least in `NLTK` package) considers contractions such as `'s` or `I'` as a independent words.
In the next example it's going to be more clear:

```python
>>> from nltk.tokenize import word_tokenize

>>> text = 'I love this flavor! It\'s by far the best choice and my go-to whenever I go to the grocery store. I wish they would restock it more often though.'
>>> word_tokens = word_tokenize(text)
>>> print(word_tokens)
['I', 'love', 'this', 'flavor', '!', 'It', "'s", 'by', 'far', 'the', 'best', 'choice', 'and', 'my', 'go-to', 'whenever', 'I', 'go', 'to', 'the', 'grocery', 'store', '.', 'I', 'wish', 'they', 'would', 'restock', 'it', 'more', 'often', 'though', '.']
```

### 3. Subword Tokenizer (Byte-Pair Encoding - BPE)
The BPE tokenization is a kind of split a bit more complex than the previous tokenizers.
This tokenizations appeared for the need of combine another two tokenizers which are the word and character tokenizers because the first one has the problem of creating large vocabularies and the second one has the problem or loosing a lot of information. So, the need was to manage a smaller vocabulary size compared to the word-tokenizer but also keep that level of information.
This tokenization follows these steps for `n` times.

![](https://miro.medium.com/v2/resize:fit:808/1*-hWgwtevk_JY2CVePBYlXQ.png)

0. Applies a pre-tokenization that is equivalent to the whitespace tokenization (depends of the implementation) and add a special end-of-word symbol such as `_` or `</w>` to prevent confusions of the algorithm.

1. **Base Vocabulary Creation**  
Start by identifying all unique characters (symbols) from the words. These characters form the base vocabulary.

- **Example base vocabulary:**  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _)
    ```

2. **Represent Words with Base Vocabs**  
Each word is broken down into symbols from the base vocabulary.  

- **Example representation:**  
    ```python
    >> (low_: 5), (lower_: 2), (newest_: 6), (widest_: 3)
    ```

- **After symbolization:**  
    ```python
    >> ((l, o, w, _): 5), ((l, o, w, e, r, _): 2), ((n, e, w, e, s, t, _): 6), ((w, i, d, e, s, t, _): 3)
    ```

3. **Vocabulary Merging**  
Iteratively merge the most frequent pairs of symbols to reduce the vocabulary size and create meaningful chunks. For this example `n = 9`.

- **Merge 1:**  
    Merge the pair `(e, s)` (occurs 9 times) into a new symbol `es`.  
    - Update vocabulary:  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _, es)
    ```
    - After update:  
    ```python
    >> ((l, o, w, _): 5), ((l, o, w, e, r, _): 2), ((n, e, w, es, t, _): 6), ((w, i, d, es, t, _): 3)
    ```

- **Merge 2:**  
    Merge `(es, t)` to create `est` (occurs 9 times).  
    - Update vocabulary:  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _, es, est)
    ```
    - After update:  
    ```python
    >> ((l, o, w, _): 5), ((l, o, w, e, r, _): 2), ((n, e, w, est, _): 6), ((w, i, d, est, _): 3)
    ```

- **Merge 3:**  
    Merge `(est, _)` to create `est_` (occurs 9 times).  
    - Update vocabulary:  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _, es, est, est_)
    ```
    - After update:  
    ```python
    >> ((l, o, w, _): 5), ((l, o, w, e, r, _): 2), ((n, e, w, est_): 6), ((w, i, d, est_): 3)
    ```

- **Merge 4:**  
    Merge `(l, o)` to create `lo` (occurs 7 times).  
    - Update vocabulary:  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _, es, est, est_, lo)
    ```
    - After update:  
    ```python
    >> ((lo, w, _): 5), ((lo, w, e, r, _): 2), ((n, e, w, est_): 6), ((w, i, d, est_): 3)
    ```

- **Merge 5:**  
    Merge `(lo, w)` to create `low` (occurs 7 times).  
    - Update vocabulary:  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _, es, est, est_, lo, low)
    ```
    - After update:  
    ```python
    >> ((low, _): 5), ((low, e, r, _): 2), ((n, e, w, est_): 6), ((w, i, d, est_): 3)
    ```

4. **Final Vocabulary & Merge Rules**  
After several iterations, the final vocabulary consists of merged symbols, with rules defining how symbols are combined.

- **Final vocabulary:**  
    ```python
    >> vocabs = (l, o, w, e, r, n, s, t, i, d, _, es, est, est_, lo, low)
    ```

- **Merge rules:**  
    ```python
    >> (e, s) → es  
    >> (es, t) → est  
    >> (est, _) → est_  
    >> (l, o) → lo  
    >> (lo, w) → low
    ```
Finally, when this tokenization is going to be applied to new text it performs almost the same process beggining with the pre-tokenization by whitespaces, then it will apply the merging rules and returns the tokenization result.



