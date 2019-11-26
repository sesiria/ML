# author sesiria 2019
# simple inverted index implementation demo.
# reference Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze, 
# Introduction to Information Retrieval, Cambridge University Press. 2008.
# https://nlp.stanford.edu/IR-book/pdf/01bool.pdf
# test corpus
corpus = [
'the new home sales top forecasts ',
'the home sales rise in july',
'the increase in home sales in july',
'the Doc 4 july new home sales rise']

# create the inverted index table
def getInvertedIndex(corpus):
    vocab = {}
    for i, doc in zip(range(len(corpus)), corpus):
        for word in doc.split(' '):
            if word not in vocab:
                vocab[word] = set()    
            vocab[word].add(i)
    
    # return the sorted./
    for key in vocab:
        postList = list(vocab[key])
        postList.sort()
        vocab[key] = [len(postList), postList]
    return vocab

# do the intersection of two query with 'and'
def intersection(p1, p2):
    result = []
    i1, i2 = 0, 0
    e1, e2 = len(p1), len(p2)
    while i1 != e1 and i2 != e2:
        if p1[i1] == p2[i2]:
            result.append(p1[i1])
            i1 += 1
            i2 += 1
        elif p1[i1] < p2[i2]:
            i1 += 1
        else:
            i2 += 1
    return result

# merge n postlist with the 'and' queries
def mergeLists(t):
    assert len(t) > 2
    i, N = 1, len(t)
    t.sort(key = lambda x : x[0])
    result = t[0][1]
    while i != N:
        result = intersection(result, t[i][1])
        i += 1
    return result

invertedIndex = getInvertedIndex(corpus)
# create inverted Index table
print(invertedIndex)
# test intersection
print(intersection(invertedIndex['new'][1], invertedIndex['july'][1]))
# test merge query
print(mergeLists([invertedIndex['new'], 
                  invertedIndex['july'],
                  invertedIndex['home'],
                  invertedIndex['the']
]))

