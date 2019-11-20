#!/bin/env python
# reference the implementation from https://github.com/joshualoehr/ngram-language-model/blob/master/preprocess.py
# rearange by sesiria 2019
import nltk
import re

SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"
# init the toktoktokenizer
toktok = nltk.tokenize.ToktokTokenizer()
word_tokenize = toktok.tokenize

def add_sentence_tokens(sentences, n):
    """Wrap each sentence in SOS and EOS tokens.
    For n >= 2, n-1 SOS tokens are added, otherwise only one is added.
    Args:
        sentences (list of str): the sentences to wrap.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        List of sentences with SOS and EOS tokens wrapped around them.
    """
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def replace_singletons(tokens):
    """Replace tokens which appear only once in the corpus with <UNK>.
    
    Args:
        tokens (list of str): the tokens comprising the corpus.
    Returns:
        The same list of tokens with each singleton replaced by <UNK>.
    
    """
    vocab = nltk.FreqDist(tokens)
    return [token if vocab[token] > 1 else UNK for token in tokens]

def sentence_clean(sentences, pattern = ''):
    """clean the characters in the input sentences.
    
    Args:
        sentences (list of str): the sentences to preprocess.
        pattern (str): the regular expression to be filted from the sentence.
    Returns:
        The cleaned sentences, filted with the specific pattern.
    """
    # by default we will filt the \n characters.
    cleaner = lambda x : x.replace('\n', '')
    if pattern != '':
        pa = re.compile(pattern)
        pattern_filter = lambda x : re.sub(pa, '', x)
        return [pattern_filter(cleaner(sent)) for sent in sentences]
    else:
        return [cleaner(sent) for sent in sentences]

def preprocess(sentences, n):
    """Add SOS/EOS/UNK tokens to given sentences and tokenize.
    Args:
        sentences (list of str): the sentences to preprocess.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        The preprocessed sentences, tokenized by words.
    """
    sentences = add_sentence_tokens(sentences, n)
    sentences = sentence_clean(sentences)
    tokens = word_tokenize(' '.join(sentences))
    tokens = replace_singletons(tokens)
    return tokens

if __name__ == '__main__':
    # test case
    sentences = ['“  一点  外语  知识  、  数理化  知识  也  没有  ，  还  攀  什么  高峰  ？', 
                '“  在  我们  做  子女  的  眼  中  ，  他  是  一个  严厉  的  父亲  ，  同时  又是  一个  充满  爱心  的  父亲  。']
    print(preprocess(sentences, 2))