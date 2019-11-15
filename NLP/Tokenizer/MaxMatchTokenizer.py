# sesiria 2019
# simple tokenizer implementation base on Max match algorithm

# Max forward Match tokenizer.
def maxMatchTokenizer(sentence, vocab, maxlen = 5):
    result = []

    while len(sentence) > 0:
        subseq = None

        if len(sentence) < maxlen:
            subseq = sentence
        else:
            subseq = sentence[0:maxlen]
        
        while len(subseq) > 0:
            if subseq in vocab or len(subseq) == 1:
                result.append(subseq)
                sentence = sentence[len(subseq):]
                break
            else:
                subseq = subseq[:-1]
    
    return result

# Max backward Match tokenizer.
def maxMatchTokenizerBW(sentence, vocab, maxlen = 5):
    result = []

    while len(sentence) > 0:
        subseq = None

        if len(sentence) < maxlen:
            subseq = sentence
        else:
            subseq = sentence[-maxlen:]
        
        while len(subseq) > 0:
            if subseq in vocab or len(subseq) == 1:
                result.append(subseq)
                sentence = sentence[:-len(subseq)]
                break
            else:
                subseq = subseq[1:]
    result.reverse()
    return result

def test_maxMatchTokenizer():
    sentence = '北京大学生前来应聘'
    vocab = {'北京', '北京大学', '大学生', '前来', '应聘'}
    words = maxMatchTokenizer(sentence, vocab)
    print('test max forward match tokenizer: ', words)

def test_maxMatchTokenizerBW():
    sentence = '北京大学生前来应聘'
    vocab = {'北京', '北京大学', '大学生', '前来', '应聘'}
    words = maxMatchTokenizerBW(sentence, vocab)
    print('test max backward match tokenizer: ', words)

def sanity_check():
    test_maxMatchTokenizer()
    test_maxMatchTokenizerBW()

sanity_check()

