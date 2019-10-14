# encoding-utf-8
# reference : https://github.com/fxsjy/jieba
import jieba

# base on jieba segment reference: https://github.com/fxsjy/jieba
seg_list = jieba.cut("贪心学院是国内最专业的人工智能在线教育品牌", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))


# TODO: 在jieba中加入"贪心学院"关键词, hint: 通过.add_word函数
# 
jieba.add_word("贪心学院")
seg_list = jieba.cut("贪心学院是国内最专业的人工智能在线教育品牌", cut_all=False)
tokenizer = [word for word in seg_list]
print(len(tokenizer))
print("Default Mode: " + "/ ", tokenizer) 