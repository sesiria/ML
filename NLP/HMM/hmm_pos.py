# Hidden Markov Model for POS Tagging implementation
# author sesiria 2019
import numpy as np

# load the corpus and build the dictionary.
def buildDictionary(corpus):
    """ build the vocabulary for the Hidden Markov Model.
    Args:
        corpus (str) : the path of the corpus.
    Returns:
        tag2id (dict) : the tag to id dictionary.
        id2tag (dict) : the id to tag dictionary.
        word2id (dict) : the word to id dictionary.
        id2word (dict) : the id to word dictionary.
    """
    # init the dictionary
    # dict for post tag
    tag2id, id2tag = {}, {}
    # dict for word
    word2id, id2word = {}, {}

    with open(corpus, 'r', encoding='utf8') as f:
        lines = f.readlines()

    for line in lines:
        items = line.split('/')
        word, tag = items[0], items[1].rstrip()

        # fill the word dictionary
        if word not in word2id:
            word2id[word] = len(word2id)
            id2word[len(id2word)] = word

        # fill the tag dictionary
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
            id2tag[len(id2tag)] = tag
    
    return tag2id, id2tag, word2id, id2word


def trainHMM(corpus, tag2id, id2tag, word2id, id2word):
    """ train the Hidden Markov Model from vocabulary and the corpus.
    Args:
        corpus (str) : the path of the corpus.
        tag2id (dict) : the tag to id dictionary.
        id2tag (dict) : the id to tag dictionary.
        word2id (dict) : the word to id dictionary.
        id2word (dict) : the id to word dictionary.
    Returns:
        PI (nummpy.array) : the initial distribution vector of the hidden state of HMM.
        A (numpy.array) : the transition matrix of the HMM
        B (numpy.array) : the emission matrix of the HMM
    """
    with open(corpus, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # init the parameter matrix
    M = len(word2id)
    N = len(tag2id)

    PI = np.zeros(N) # initial
    B = np.zeros((N, M)) # emission
    A = np.zeros((N, N)) # transition

    # count the stastics matrix.
    prev_tag = ""
    for line in lines:
        items = line.split('/')
        wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
        if prev_tag == "":
            PI[tagId] += 1
        else:
            A[tag2id[prev_tag]][tagId] += 1
        B[tagId][wordId] += 1

        if items[0] == ".":
            prev_tag = ""
        else:
            prev_tag = items[1].rstrip()
    
    # normalize
    PI /= sum(PI)
    for i in range(N):
        A[i] /= sum(A[i])
        B[i] /= sum(B[i])

    return PI, A, B


def viterbi(sentence, PI, A, B):
    """ decoding of sentence of the Hidden Markov Model.
    the decoding routine is via the dynamic programing approach of viterbi algorithm.
    Args:
        sentence (str) : the input sentence
        PI (nummpy.array) : the initial distribution vector of the hidden state of HMM.
        A (numpy.array) : the transition matrix of the HMM
        B (numpy.array) : the emission matrix of the HMM
    Returns:
        pos(str) : the string of sentences with pos tag.
    """
    def log(v):
        """
        smoothing log
        """
        if v == 0:
            return np.log(v+ 10e-6)
        return np.log(v)

    w = [word2id[word] for word in sentence.split(' ')]
    T = len(w) # the number of words

    N, M = B.shape

    dp = np.zeros((T, N))
    path = np.zeros((T, N), dtype = int)

    for j in range(N):
        dp[0][j] = log(PI[j]) + log(B[j][w[0]])
    
    for i in range(1, T):
        for j in range(N):
            dp[i][j] = -9999999
            scoreEmission = log(B[j][w[i]])
            for k in range(N):
                score = dp[i - 1][k] + log(A[k][j]) + scoreEmission
                if score > dp[i][j]:
                    dp[i][j] = score
                    path[i][j] = k
    
    best_seq = [0] * T
    best_seq[T - 1] = np.argmax(dp[T - 1])

    for i in range(T - 2, -1, -1):
        best_seq[i] = path[i + 1][best_seq[i + 1]]
    
    result = []
    for i in range(len(best_seq)):
        result.append(id2tag[best_seq[i]])
    return result


if __name__ == '__main__':
    # load the dictionary
    tag2id, id2tag, word2id, id2word = buildDictionary('traindata.txt')

    print(f"tag size: {len(tag2id)}")
    print(f"tag2id:{tag2id}")

    # init the paramter
    M = len(word2id) # observation
    N = len(tag2id) # latent variable

    # train the HMM model.
    PI, A, B = trainHMM('traindata.txt', tag2id, id2tag, word2id, id2word)
    print(f"pi:{PI}")
    print(f"A[0]:{A[0]}")
    
    # test sentence
    x = 'keep new to everything'
    r = viterbi(x, PI, A, B)
    print(x)
    print(' '.join(r))