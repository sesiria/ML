# backtracking algorithm for enumerate all type of segments.
def backTracking(input_str, dic_words, segment, segments):
    if len(input_str) == 0:
        segments.append(segment)
        return
    
    for i in range(len(input_str)):
        subseq = input_str[0:i+1]
        if (len(subseq) == 1) or (subseq in dic_words):
            segment.append(subseq)
            backTracking(input_str[len(subseq):], dic_words, segment, segments)
            segment.pop()

# sanity_check for backTracking
s = 'abc'
vocab = {'a', 'ab', 'abc'}
segment = []
segments= []
backTracking(s, vocab, segment, segments)
print(segments)