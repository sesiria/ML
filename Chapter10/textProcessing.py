import re
import jieba
def clean_symbols(text):
    """
    对特殊符号做一些处理，此部分已写好。如果不满意也可以自行改写，不记录分数。
    """
    text = re.sub('[!！]+', "!", text)
    text = re.sub('[?？]+', "?", text)
    text = re.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’[\\]^_`{|}~]+", " OOV ", text)
    # change all digital number into ' NUM '
    text = re.sub('\d+(\.\d+)*', ' NUM ', text)
    return re.sub("\s+", " ", text)  

ss = 'adafasw12314.12333egrdf5236qew'
num = re.findall('\d+',ss)
print(num)

ss = re.sub('\d+(\.\d+)*', ' NUM ', ss)
print(ss)

text = '请问这机不是有个遥控器的吗？'
text = clean_symbols(text)
print(text)

tokens = [x for x in jieba.cut(text, cut_all = False)]
print(tokens)
sentence = [" ".join(tokens)]
print(sentence)

text = """我是一条天狗呀！
    我把月来吞了，
    我把日来吞了，
    我把一切的星球来吞了，
    我把全宇宙来吞了。
    我便是我了！"""
sentences = text.split()
sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
document = [" ".join(sent0) for sent0 in sent_words]
print(document)

