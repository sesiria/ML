# method.1 build a stop word dictionary.
stop_words = {"the", "an", "is", "there"} # using the set as hash table for query complexity O(1)
# use the dictionary. assume word_list included the stop word
word_list = ["we", "are", "the", "students"]
filtered_words = [word for word in word_list if word not in stop_words]
print(filtered_words)

# method.2 using the stopwords from nltk
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
print(cachedStopWords)