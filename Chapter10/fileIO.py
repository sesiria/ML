# author sesiria 2019
# simple file IO demo
import re

def readCommentLabels(fileName, label = False):
    texts = []
    list_comments = []
    list_labels = []
    pattern = re.compile(r'(?<=label=\")\d')
    with open(fileName, 'r', encoding = 'UTF-8') as f1:
        texts = f1.readlines()
    
    strTmp = ''
    for i in range(0, len(texts)):
        # find the XML start element
        if texts[i].find('<review') != -1:
            # reset the string
            strTmp = ''
            # try to extract the label
            if label == True:
                num = pattern.findall(texts[i])
                assert len(num) > 0 # the document does not have the label
                list_labels.append(int(num[0]))
        # find the XML end element
        elif texts[i].find('</review') != -1:
            list_comments.append(strTmp)
        else:
            strTmp += texts[i].strip()

    return list_comments, list_labels

def sanity_check():
    fileName = "E:/ML/ML/Greedy/Chapter10/test.txt"
    comments, labels = readCommentLabels(fileName, label = True)
    print(comments)
    print(labels)

sanity_check()


    
    

