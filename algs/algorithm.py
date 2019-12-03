def algs(Str):
    vocab = {}
    for c in Str:
        if c in vocab:
            vocab[c] += 1
        else:
            vocab[c] = 1
    # create list
    array = []
    for key in vocab:
        array.append((key, vocab[key]))
    qsort(array)
    print(array)

def comp(a, b):
    return a[1] - b[1] if a[1] != b[1] else ord(a[0]) - ord(b[0])

def partition(array):
    k = 0
    i = 1
    j = len(array) - 1
    val = array[k]
    while i < j:
        while i < j and comp(array[j], val) > 0:
            j -= 1
        while i < j and comp(array[i], val) < 0:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
    if comp(array[i], val) > 0:
        array[i - 1], array[k] = array[k], array[i - 1]
    else:
        array[i], array[k] =  array[k], array[i]
    return k

def qsort(array):
    if len(array) <= 1:
        return
    k = partition(array)
    qsort(array[0:k-1])
    qsort(array[k+1:])
    
# void sort(array):
algs('aaabbsssassssaaaaasssssscccbcedddasafzzzxvzxvasfasfsaeeeasdfasf')