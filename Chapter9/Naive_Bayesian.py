import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read span.csv
df = pd.read_cv("spam.csv", encoding = 'latin')
df.head()

# rename  the column of v1 and v2
df.rename(columns = {'v1' : 'Label', 'v2' : 'Text'}, inplace = True)
df.head()

# map'ham' and 'span' to 0 and 1
df['numLabel'] = df['Label'].map({'ham' : 0, 'spam' : 1})
df.head()

# count number of ham and spam
print ('# of ham : ', len(df[df.numLabel == 0]), ' # of spam: ', len(df[df.numLabel == 1]))
print ('# of total samples: ', len(df))

# count the length of  text, and plot a histogram
text_lengths = [len(df.loc[i, 'Text']) for i in range(len(df))]
plt.hist(terxt_lengths, 100, facecolor = 'blue', alpha = 0.5)
plt.xlim([0, 200])
plt.show()

# import English vocabulary
from sklearn.feature_extraction.text import CountVectorizer

# construct word vector (base on the frequency of the word)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
y = df.numLabel

# split the data into train and test data set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)
print('# of samples in the train data set: ', X_train.shape[0], '# of samples in test data set: ', X_test.shape[0])

# use the Naive Bayesian for model training
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB(alpha = 1.0, fit_prior = True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data: ", accuracy_score(y_test, y_pred))

# print confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels = [0, 1])