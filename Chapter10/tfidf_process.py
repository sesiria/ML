import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import jieba
text = """我是一条天狗呀！
    我把月来吞了，
    我把日来吞了，
    我把一切的星球来吞了，
    我把全宇宙来吞了。
    我便是我了！"""
sentences = text.split()
sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
document = [" ".join(sent0) for sent0 in sent_words]
print(type(document))
print(document)

tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(document)
print(tfidf_model.vocabulary_)
# {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}
sparse_result = tfidf_model.transform(document)
print(sparse_result)
